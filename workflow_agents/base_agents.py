"""
Base agents for the LLM Agent Toolkit.

KEY DIFFERENCES:
  Anthropic                          OpenAI
  ────────────────────────────────   ────────────────────────────────
  client.messages.create(...)        client.chat.completions.create(...)
  system= (separate parameter)       {"role": "system", ...} in messages list
  response.content[0].text           response.choices[0].message.content
  anthropic.APIError                 openai.OpenAIError
  claude-haiku / claude-sonnet       gpt-4o-mini / gpt-4o
"""

import json
import math
import os
from contextlib import nullcontext

from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from .spinner import Spinner  # oppure: from .spinner import Spinner


load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gpt-4o-mini"   # fast and cheap — good for dev
POWERFUL_MODEL = "gpt-4o"       # more capable — use for complex tasks
MAX_TOKENS = 4096


# ---------------------------------------------------------------------------
# BaseAgent — parent class with shared API logic
# ---------------------------------------------------------------------------

class BaseAgent:
    """
    Base class for all agents.

    Handles:
    - OpenAI client initialization
    - API call with error handling
    - Lightweight logging for debugging
    """

    def __init__(self, model: str = DEFAULT_MODEL, verbose: bool = True):
        self.client = OpenAI(
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.model = model
        self.verbose = verbose

    def _call_api(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = MAX_TOKENS,
        temperature: float = 0.3,
        spinner_message: str = "",
    ) -> str:
        """
        Calls the OpenAI API with the given messages and parameters.
        NOTE:
        - temperature=0.3 → more deterministic output (good for classification)
        - temperature=0.7 → more creative output (good for brainstorming)
        - max_tokens: the maximum number of tokens the model can generate
        """
        # Build the full messages list.
        # If a system prompt is provided, prepend it as a system-role message.
        full_messages: list[dict] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        label = spinner_message if spinner_message else self.__class__.__name__

        try:
            with Spinner(label):
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=full_messages,
                )
            
            return response.choices[0].message.content

        except OpenAIError as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e

    def _log(self, label: str, content: str) -> None:
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[{self.__class__.__name__}] {label}")
            print(f"{'='*60}")
            print(content[:800] + ("..." if len(content) > 800 else ""))

    def run(self, *args, **kwargs):
        raise NotImplementedError("Every agent must implement run()")


# ---------------------------------------------------------------------------
# 1. DirectPromptAgent: sends the prompt directly to the model with no modifications
# ---------------------------------------------------------------------------

class DirectPromptAgent(BaseAgent):
    """
    The simplest possible pattern: sends the prompt exactly as received,
    with no modifications. No system prompt, no added context.

    WHEN TO USE:
    - Rapid prototyping
    - Tasks where you don't want to influence model behavior
    - Baseline for comparing other agents
    """

    def run(self, prompt: str) -> str:
        """
        Args:
            prompt: The text to send directly to the model

        Returns:
            The raw model response
        """
        self._log("INPUT PROMPT", prompt)
        messages = [{"role": "user", "content": prompt}]
        result = self._call_api(messages)
        self._log("OUTPUT", result)
        return result


# ---------------------------------------------------------------------------
# 2. AugmentedPromptAgent: adds a system prompt to guide the model's behavior
# ---------------------------------------------------------------------------

class AugmentedPromptAgent(BaseAgent):
    """
    Adds a configurable system prompt to the user's prompt.
    The system prompt defines who the agent is, what it can do, and how it responds. 
    """

    def __init__(self, system_prompt: str, model: str = DEFAULT_MODEL, verbose: bool = True):
        super().__init__(model=model, verbose=verbose)
        self.system_prompt = system_prompt

    def run(self, prompt: str) -> str:
        """
        Args:
            prompt: The specific task for this agent

        Returns:
            Response guided by the system prompt
        """
        self._log("SYSTEM PROMPT", self.system_prompt)
        self._log("USER PROMPT", prompt)

        messages = [{"role": "user", "content": prompt}]
        result = self._call_api(messages, system=self.system_prompt)

        self._log("OUTPUT", result)
        return result


# ---------------------------------------------------------------------------
# 3. KnowledgeAugmentedPromptAgent: injects document context into the prompt
# ---------------------------------------------------------------------------

class KnowledgeAugmentedPromptAgent(BaseAgent):
    """
    Extends AugmentedPromptAgent by injecting document context into the prompt.
    This is simplified RAG: instead of doing real retrieval, we pass the relevant
    documents (already selected) directly into the context window.
    """

    def __init__(
        self,
        system_prompt: str,
        knowledge_docs: list[str] | None = None,
        model: str = DEFAULT_MODEL,
        verbose: bool = True,
    ):
        super().__init__(model=model, verbose=verbose)
        self.system_prompt = system_prompt
        self.knowledge_docs = knowledge_docs or []

    def _build_context_block(self) -> str:
        """Builds the context block from the provided documents."""

        if not self.knowledge_docs:
            return ""
        
        parts = ["<knowledge_base>"]

        for i, doc in enumerate(self.knowledge_docs, 1):
            parts.append(f"<document index='{i}'>\n{doc}\n</document>")
        parts.append("</knowledge_base>")

        return "\n".join(parts)

    def run(self, prompt: str) -> str:
        """
        Args:
            prompt: The task to execute using the document context

        Returns:
            Response informed by the injected context
        """
        context = self._build_context_block()
        full_prompt = f"{context}\n\n{prompt}" if context else prompt

        self._log("SYSTEM PROMPT", self.system_prompt)
        self._log("FULL USER PROMPT (with context)", full_prompt)

        messages = [{"role": "user", "content": full_prompt}]
        result = self._call_api(messages, system=self.system_prompt, temperature=0.4)

        self._log("OUTPUT", result)
        return result


# ---------------------------------------------------------------------------
# 4. RAGKnowledgePromptAgent: retrieves relevant chunks based on 
# semantic similarity and injects them into the prompt
# ---------------------------------------------------------------------------

class RAGKnowledgePromptAgent(BaseAgent):
    """
    Agent with true semantic retrieval: uses cosine similarity between embeddings
    to find the most relevant chunks from the knowledge base.
    
    SIMILARITY:
    - similarity = 1.0 → semantically identical
    - similarity = 0.0 → unrelated
    - similarity = -1.0 → opposite meanings

    NOTE:
    This agent uses a simulated bag-of-words embedding for simplicity.
    To use real OpenAI embeddings, replace _build_embeddings() with:

        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]

    Real embeddings are much more accurate but require additional API calls.
    """

    def __init__(
        self,
        system_prompt: str,
        knowledge_chunks: list[str],
        top_k: int = 3,
        model: str = DEFAULT_MODEL,
        verbose: bool = True,
    ):
        super().__init__(model=model, verbose=verbose)
        self.system_prompt = system_prompt
        self.knowledge_chunks = knowledge_chunks
        self.top_k = top_k
        self._chunk_embeddings = self._build_embeddings(knowledge_chunks)

    def _build_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Simulates embeddings using normalized bag of words.
        Replace with client.embeddings.create() for production use.
        """
        all_words: set[str] = set()
        tokenized = []
        for text in texts:
            words = text.lower().split()
            tokenized.append(words)
            all_words.update(words)
        vocab = sorted(all_words)
        vocab_idx = {w: i for i, w in enumerate(vocab)}

        embeddings = []
        for words in tokenized:
            vec = [0.0] * len(vocab)
            for w in words:
                if w in vocab_idx:
                    vec[vocab_idx[w]] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            embeddings.append([x / norm for x in vec])
        return embeddings

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Computes cosine similarity between two normalized vectors."""
        return sum(x * y for x, y in zip(a, b))

    def _retrieve(self, query: str) -> list[str]:
        """Retrieves the top_k most relevant chunks for the given query."""
        query_emb = self._build_embeddings([query])[0]
        scores = [
            (self._cosine_similarity(query_emb, chunk_emb), i)
            for i, chunk_emb in enumerate(self._chunk_embeddings)
        ]
        scores.sort(reverse=True)
        top_indices = [i for _, i in scores[: self.top_k]]
        return [self.knowledge_chunks[i] for i in top_indices]

    def run(self, prompt: str) -> str:
        """
        Args:
            prompt: The user query — also used as the retrieval query

        Returns:
            Response grounded in the semantically retrieved chunks
        """
        retrieved = self._retrieve(prompt)
        self._log("RETRIEVED CHUNKS", "\n---\n".join(retrieved))

        context_block = "<retrieved_context>\n"
        for i, chunk in enumerate(retrieved, 1):
            context_block += f"<chunk index='{i}'>{chunk}</chunk>\n"
        context_block += "</retrieved_context>"

        full_prompt = f"{context_block}\n\n{prompt}"
        messages = [{"role": "user", "content": full_prompt}]
        result = self._call_api(messages, system=self.system_prompt, temperature=0.3)

        self._log("OUTPUT", result)
        return result


# ---------------------------------------------------------------------------
# 5. EvaluationAgent  (LLM-as-judge): evaluates another agent's output against defined criteria
# ---------------------------------------------------------------------------

class EvaluationAgent(BaseAgent):
    """
    Evaluates the output of another agent against a defined set of criteria.
    Implements the "LLM-as-judge" pattern.

    OUTPUT STRUCTURE:
    Always returns a dict with:
    - passed: bool — does the output meet all criteria?
    - score: float (0-10) — numeric quality score
    - feedback: str — detailed assessment
    - criteria_results: dict — per-criterion breakdown
    """

    EVAL_SYSTEM_PROMPT = """You are a rigorous quality evaluator for AI-generated content.
Your job is to assess outputs against specific criteria and provide structured feedback.

Always respond in valid JSON with this exact structure:
{
  "passed": true or false,
  "score": <number 0-10>,
  "feedback": "<overall assessment>",
  "criteria_results": {
    "<criterion_name>": {
      "passed": true or false,
      "comment": "<specific feedback>"
    }
  }
}

Be strict but fair. A "passed: true" requires ALL criteria to be met."""

    def __init__(
        self,
        criteria: list[str],
        model: str = DEFAULT_MODEL,
        verbose: bool = True,
    ):
        super().__init__(model=model, verbose=verbose)
        self.criteria = criteria

    def run(self, content_to_evaluate: str, context: str = "") -> dict:
        """
        Args:
            content_to_evaluate: The agent output to assess
            context: Optional context (e.g. the original prompt)

        Returns:
            dict with passed, score, feedback, criteria_results
        """
        criteria_list = "\n".join(f"- {c}" for c in self.criteria)

        eval_prompt = f"""Evaluate the following content against these criteria:

CRITERIA:
{criteria_list}

{"CONTEXT: " + context if context else ""}

CONTENT TO EVALUATE:
{content_to_evaluate}

Provide your evaluation as JSON."""

        self._log("EVALUATING CONTENT", content_to_evaluate[:400])

        messages = [{"role": "user", "content": eval_prompt}]
        raw = self._call_api(messages, system=self.EVAL_SYSTEM_PROMPT, temperature=0.1)

        try:
            clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
            result = json.loads(clean)
        except json.JSONDecodeError:
            result = {
                "passed": False,
                "score": 0,
                "feedback": f"JSON parsing error: {raw[:200]}",
                "criteria_results": {},
            }

        self._log("EVALUATION RESULT", json.dumps(result, indent=2))
        return result


# ---------------------------------------------------------------------------
# 6. RoutingAgent: decides which team should receive a task, based on task 
# content and the description of available teams
# ---------------------------------------------------------------------------

class RoutingAgent(BaseAgent):
    """
    Decides which team should receive a task, based on task content and
    the description of available teams.

    OUTPUT:
    - route: str — the name of the selected team
    - confidence: float — how confident the agent is in its choice
    - reasoning: str — why this team was selected
    """

    ROUTING_SYSTEM_PROMPT = """You are an intelligent task router for a multi-agent workflow system.
Your job is to analyze a task and route it to the most appropriate team.

Always respond in valid JSON:
{
  "route": "<team_name>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation of why this team is best suited>"
}

Be decisive. If multiple teams could handle the task, pick the single best one."""

    def __init__(
        self,
        routes: dict[str, str],
        model: str = DEFAULT_MODEL,
        verbose: bool = True,
    ):
        super().__init__(model=model, verbose=verbose)
        self.routes = routes

    def run(self, task: str) -> dict:
        """
        Args:
            task: Description of the task to route

        Returns:
            dict with route, confidence, reasoning
        """
        routes_description = "\n".join(
            f"- {name}: {desc}" for name, desc in self.routes.items()
        )

        routing_prompt = f"""Available teams:
{routes_description}

Task to route:
{task}

Select the most appropriate team."""

        self._log("TASK TO ROUTE", task)
        messages = [{"role": "user", "content": routing_prompt}]
        raw = self._call_api(messages, system=self.ROUTING_SYSTEM_PROMPT, temperature=0.1)

        try:
            clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
            result = json.loads(clean)
        except json.JSONDecodeError:
            result = {
                "route": list(self.routes.keys())[0],
                "confidence": 0.5,
                "reasoning": f"Fallback routing: {raw[:100]}",
            }

        self._log("ROUTING DECISION", json.dumps(result, indent=2))
        return result


# ---------------------------------------------------------------------------
# 7. ActionPlanningAgent: decomposes a high-level goal into an ordered list 
# of concrete sub-tasks.
# ---------------------------------------------------------------------------

class ActionPlanningAgent(BaseAgent):
    """
    Decomposes a high-level goal into an ordered list of concrete sub-tasks.
    Implements the Plan-and-Execute pattern.

    OUTPUT:
    A list of dicts, each containing:
    - id: str — unique task identifier
    - description: str — what needs to be done
    - expected_output: str — what the output should look like
    - dependencies: list[str] — IDs of tasks that must complete first
    """

    PLANNING_SYSTEM_PROMPT = """You are a technical project planning expert.
Your job is to decompose high-level goals into concrete, actionable sub-tasks.

Always respond in valid JSON — a list of task objects:
[
  {
    "id": "task_1",
    "description": "<concrete action to take>",
    "expected_output": "<what the output should look like>",
    "dependencies": []
  },
  ...
]

Guidelines:
- Each task should be completable by a single specialized agent
- Tasks should be ordered logically (earlier tasks inform later ones)
- Be specific: "Write user stories for authentication feature" not "do requirements"
- Maximum 6 tasks to keep the workflow focused"""

    def run(self, goal: str, context: str = "") -> list[dict]:
        """
        Args:
            goal: The high-level goal to decompose
            context: Optional additional information (e.g. a product spec excerpt)

        Returns:
            Ordered list of sub-tasks
        """
        planning_prompt = f"""Goal to decompose:
{goal}

{"Additional context:" + context[:1000] if context else ""}

Create a structured action plan."""

        self._log("GOAL TO PLAN", goal)
        messages = [{"role": "user", "content": planning_prompt}]
        raw = self._call_api(
            messages,
            system=self.PLANNING_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=2000,
        )

        try:
            clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
            tasks = json.loads(clean)
            if not isinstance(tasks, list):
                tasks = [tasks]
        except json.JSONDecodeError:
            tasks = [
                {
                    "id": "task_1",
                    "description": goal,
                    "expected_output": "Completed goal",
                    "dependencies": [],
                }
            ]

        self._log("ACTION PLAN", json.dumps(tasks, indent=2))
        return tasks