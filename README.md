# agentflow

A modular Python toolkit for building multi-agent AI workflows powered by OpenAI. Provides seven reusable agent classes that can be composed into pipelines for planning, routing, knowledge retrieval, generation, and self-correcting evaluation.

---

## Why agentflow?

This toolkit keeps things explicit: each agent is a focused Python class with a single responsibility. You can understand every call, every prompt, every decision — and swap or extend any piece independently.

---

## Architecture

The library ships with seven agent classes and an example orchestrator that wires them into a full multi-agent workflow:

```
Input (natural language goal)
        │
        ▼
┌─────────────────────┐
│  ActionPlanningAgent │  → Decomposes the goal into ordered sub-tasks
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    RoutingAgent      │  → Assigns each sub-task to the right specialist
└──────────┬──────────┘
           │
    ┌──────┴──────┬──────────────┐
    ▼             ▼              ▼
[Team A]      [Team B]       [Team C]
    │             │              │
    ▼             ▼              ▼
[KnowledgeAug] [KnowledgeAug] [KnowledgeAug]  ← domain-aware generation
    │             │              │
    ▼             ▼              ▼
[EvalAgent]  [EvalAgent]   [EvalAgent]  ← LLM-as-judge, auto-retry on fail
    │             │              │
    └─────────────┴──────────────┘
                  │
                  ▼
         Structured JSON + Markdown report
```

---

## The 7 Agents

### 1. `DirectPromptAgent`
The simplest possible agent: sends a prompt to the model and returns the text response. No system prompt, no added context. Use it as a baseline or for rapid prototyping.

### 2. `AugmentedPromptAgent`
Extends the direct agent with a configurable system prompt that defines the model's role, output format, and constraints.

### 3. `KnowledgeAugmentedPromptAgent`
Injects one or more documents directly into the context window alongside the user prompt. This is the simplest form of RAG: no retrieval step, you decide which documents are relevant and pass them in.

### 4. `RAGKnowledgePromptAgent`
Adds a retrieval step before generation. The document is chunked, embedded, and the top-k most semantically relevant chunks are selected for the context window.

### 5. `EvaluationAgent`
Uses the LLM as a judge to evaluate the output of another agent against a set of explicit criteria. Returns a structured result with a numeric score, pass/fail status, per-criterion breakdown, and actionable feedback.

### 6. `RoutingAgent`
Given a task description and a dictionary of available teams, decides which team is best suited to handle the task. Returns the decision with a confidence score and reasoning.

### 7. `ActionPlanningAgent`
Decomposes a high-level goal into a list of concrete, ordered sub-tasks with dependencies. Implements the Plan-and-Execute pattern: plan first, then route each task to the right executor.


## Setup

**Requirements:** Python 3.11+, OpenAI API key.

```bash
git clone https://github.com/your-username/agentflow.git
cd agentflow
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-ant-..." > .env
```

---

## Usage

### Test each agent individually

```bash
python tests/test_direct_prompt.py
python tests/test_augmented_prompt.py
python tests/test_knowledge_augmented.py
python tests/test_rag_knowledge.py
python tests/test_evaluation.py
python tests/test_routing.py
python tests/test_action_planning.py
```

### Run the example orchestrator

```bash
python agentic_workflow.py
```

## License

MIT
