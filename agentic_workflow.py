import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) 

from workflow_agents.base_agents import (
    ActionPlanningAgent,
    RoutingAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
)

# ---------------------------------------------------------------------------
# Global configuration: company/product name, shared constants
# ---------------------------------------------------------------------------
COMPANY_NAME = "InnovateNext Solutions"
PRODUCT_NAME = "Email Router"

# ----------------------------------------------------------------------------
# Team identifiers (used for routing and configuration)
# In a complex workflow, these could have identifiers, metadata
# ----------------------------------------------------------------------------
PM_TEAM = "pm_team" # Product Manager team
PGM_TEAM = "pgm_team" # Program Manager team (focuses on features and roadmap)
DEV_TEAM = "dev_team" # Development team (focuses on technical implementation)

# ----------------------------------------------------------------------------
# Team configuration: route descriptions, system prompts, evaluation criteria
# It is used by the RoutingAgent to assign tasks, 
# by the KnowledgeAugmentedPromptAgent to set the system prompt, 
# and by the EvaluationAgent to evaluate outputs.
# ---------------------------------------------------------------------------
TEAM_ROUTES = {
    PM_TEAM: (
        """Product Manager team — specializes in user stories, acceptance criteria, 
        and requirements from the end-user perspective. Handles: user stories, 
        personas, use cases, acceptance tests."""
    ),
    PGM_TEAM: (
        """Program Manager team — specializes in product features, epics, and release 
        planning from a product roadmap perspective. Handles: feature definitions, 
        epics, product milestones, scope decisions."""
    ),
    DEV_TEAM: (
        """Development Engineer team — specializes in technical implementation tasks, 
        architecture, and engineering work breakdown. Handles: engineering tasks, 
        technical specs, implementation steps, API design."""
    ),
}

# ----------------------------------------------------------------------------
# System prompts for each team (KnowledgeAugmentedPromptAgent)
# Specify the expected output format and the key criteria for success for 
# each team.
# ----------------------------------------------------------------------------
TEAM_SYSTEM_PROMPTS = {
    PM_TEAM: """You are a senior Product Manager at {COMPANY_NAME}.
You write clear, well-structured user stories using the standard format:
"As a [user type], I want [action], so that [benefit]."

Each user story MUST include:
1. Title (short, descriptive)
2. User story statement (As a / I want / So that)
3. Acceptance Criteria (3-5 specific, testable criteria using Given/When/Then or bullet format)
4. Priority (High/Medium/Low)
5. Story Points estimate (Fibonacci: 1, 2, 3, 5, 8, 13)

Use the provided product specification as your source of truth.
Generate 3-5 user stories for the given scope.""",

    PGM_TEAM: """You are a senior Program Manager at {COMPANY_NAME}.
You define product features that translate business requirements into 
structured development scope.

Each feature definition MUST include:
1. Feature Name (clear, noun-based)
2. Description (2-3 sentences explaining what and why)
3. User Value (who benefits and how)
4. Dependencies (what must exist first)
5. Success Metrics (how we know it's working)
6. Scope (In Scope / Out of Scope for this release)

Use the provided product specification as your source of truth.
Generate 3-5 feature definitions for the given scope.""",

    DEV_TEAM: """You are a senior Software Engineer at {COMPANY_NAME}.
You create detailed engineering tasks that developers can pick up and execute.

Each engineering task MUST include:
1. Task Title (verb-based, e.g., "Implement OAuth2 email ingestion")
2. Description (technical explanation of what to build)
3. Technical Requirements (specific tech choices, APIs, data structures)
4. Implementation Steps (numbered, ordered list of concrete steps)
5. Definition of Done (checklist of completion criteria)
6. Estimated Effort (hours or days)
7. Dependencies (other tasks that must complete first)

Use the provided product specification as your source of truth.
Generate 3-5 engineering tasks for the given scope.""",
}

# ----------------------------------------------------------------------------
# Evaluation criteria for each team (EvaluationAgent)
# These criteria are used by the EvaluationAgent to score the outputs 
# of each team.
# Each criterion is a specific, measurable aspect of the output that 
# indicates quality.
# The EvaluationAgent will provide a score from 1 to 10 for each criterion 
# and an overall pass/fail based on whether the output meets the standards.
# ----------------------------------------------------------------------------
TEAM_EVAL_CRITERIA = {
    PM_TEAM: [
        """All user stories follow the 'As a / I want / So that' format,
        Each story has at least 3 specific, testable acceptance criteria,
        Stories reference concrete functionality from the Email Router product,
        Stories are focused on user value, not implementation details,
        Priority and story points are assigned to each story""",
    ],
    PGM_TEAM: [
        """Each feature has a clear name and description",
        User value and success metrics are explicitly stated for each feature,
        Dependencies between features are identified,
        In-scope and out-of-scope boundaries are defined,
        Features align with the product specification goals""",
    ],
    DEV_TEAM: [
        """Each task has a concrete, actionable title starting with a verb,
        Implementation steps are numbered and technically specific,
        Definition of Done is measurable and complete,
        Technical requirements reference actual technologies and APIs,
        Effort estimates are provided for each task""",
    ],
}


# ---------------------------------------------------------------------------
# ProjectManagementWorkflow: workflow multi-agent orchestrator
# ---------------------------------------------------------------------------

class ProjectManagementWorkflow:
    """ Multi-agent workflow orchestrator for technical project management."""

    def __init__(self, product_spec_path: str, verbose: bool = True):
        """ 
        Initialize agents and load product specification. 
            The product specification is used as the common knowledge base for all teams.
                - product_spec_path: path to the product specification text file
                - verbose: if True, prints detailed logs of each step and agent output
        """

        self.verbose = verbose
        self.product_spec = Path(product_spec_path).read_text(encoding="utf-8")
        self.results: dict = {}

        # Initialize agents
        self.planner = ActionPlanningAgent(verbose=verbose)
        self.router = RoutingAgent(routes=TEAM_ROUTES, verbose=verbose)

        # Initialize teams (KnowledgeAugmentedAgent with the spec as context)
        self.teams: dict[str, KnowledgeAugmentedPromptAgent] = {}
        self.evaluators: dict[str, EvaluationAgent] = {}

        for team_name, system_prompt in TEAM_SYSTEM_PROMPTS.items():
            self.teams[team_name] = KnowledgeAugmentedPromptAgent(
                system_prompt=system_prompt,
                knowledge_docs=[self.product_spec],
                verbose=verbose,
            )
            self.evaluators[team_name] = EvaluationAgent(
                criteria=TEAM_EVAL_CRITERIA[team_name],
                verbose=verbose,
            )

    def _print_phase(self, phase: str, description: str) -> None:
        print(f"\n{'='*70}")
        print(f"🔷 PHASE: {phase}")
        print(f"   {description}")
        print(f"{'='*70}")

    def _execute_team_task(
        self,
        team_name: str,
        task_description: str,
        task_id: str,
        max_retries: int = 2,
    ) -> dict:
        """ 
        Executes a single task for a given team with evaluation and retry logic.

        PATTERN: Generate → Evaluate → (Retry if failed) → Return 
        """
        team = self.teams[team_name]
        evaluator = self.evaluators[team_name]

        print(f"\n🤖 [{team_name.upper()}] Executing: {task_description[:80]}...")

        for attempt in range(1, max_retries + 1):
            print(f"   Attempt {attempt}/{max_retries}...")

            # STEP 1: the team generates the output
            output = team.run(
                f"Based on the Email Router product specification provided, {task_description}"
            )

            # STEP 2: the evaluator evaluates the output
            eval_result = evaluator.run(
                content_to_evaluate=output,
                context=f"Task: {task_description}",
            )

            if eval_result["passed"]:
                print(f"   ✅ Evaluation PASSED (score: {eval_result['score']}/10)")
                return {
                    "team": team_name,
                    "task_id": task_id,
                    "task_description": task_description,
                    "output": output,
                    "evaluation": eval_result,
                    "attempts": attempt,
                }
            else:
                print(f"   ⚠️  Evaluation FAILED (score: {eval_result['score']}/10)")
                print(f"   Feedback: {eval_result['feedback'][:150]}...")
                if attempt < max_retries:
                    print(f"   Retrying with feedback...")
                    # 
                    task_description = (
                        f"{task_description}\n\n"
                        f"IMPORTANT — Previous attempt failed. Fix these issues:\n"
                        f"{eval_result['feedback']}"
                    )

        # After max retries, return the best effort output with evaluation results
        print(f"   ⚠️  Max retries reached. Returning best effort output.")
        return {
            "team": team_name,
            "task_id": task_id,
            "task_description": task_description,
            "output": output,
            "evaluation": eval_result,
            "attempts": max_retries,
        }

    def run(self, tpm_request: str) -> dict:
        """
        Executes the complete workflow.

        Args:
            tpm_request: The high-level request from the TPM

        Returns:
            dict: A structured report containing the action plan, outputs, evaluations, and quality summary.
        """
        start_time = time.time()
        print(f"\n{'🚀'*35}")
        print(f"AGENTIC WORKFLOW — InnovateNext Solutions")
        print(f"Pilot: Email Router | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'🚀'*35}")

        # PHASE 1: PLANNING 
        self._print_phase(
            "1/4 — ACTION PLANNING",
            "ActionPlanningAgent decomposes the TPM request into a structured plan of tasks"
        )
        tasks = self.planner.run(
            goal=tpm_request,
            context=self.product_spec[:500],  # Pass a excerpt of the spec as a hint
        )
        print(f"\n📋 Plan generated: {len(tasks)} tasks")

        # PHASE 2: ROUTING
        self._print_phase(
            "2/4 — INTELLIGENT ROUTING",
            "RoutingAgent assigns each task to the most suitable team based on the description"
        )
        task_assignments: list[dict] = []
        for task in tasks:
            routing_result = self.router.run(task["description"])
            task_assignments.append({
                "task": task,
                "assigned_team": routing_result["route"],
                "routing_confidence": routing_result["confidence"],
                "routing_reasoning": routing_result["reasoning"],
            })
            print(
                f"   [{task['id']}] → {routing_result['route']} "
                f"(confidence: {routing_result['confidence']:.2f})"
            )

        # PHASE 3: EXECUTION
        self._print_phase(
            "3/4 — TEAM EXECUTION + EVALUATION",
            "Each team generates the output, the EvaluationAgent validates it"
        )

        # Group tasks by team and force at least one task per team "key"
        # (in a real system the router decides everything; here we ensure
        # that all three teams produce output for the pilot)
        team_tasks: dict[str, list] = {
            PM_TEAM: [],
            PGM_TEAM: [],
            DEV_TEAM: [],
        }

        for assignment in task_assignments:
            team = assignment["assigned_team"]
            if team in team_tasks:
                team_tasks[team].append(assignment)
            else:
                # Fallback: sends to the team with fewer tasks assigned so far
                min_team = min(team_tasks, key=lambda k: len(team_tasks[k]))
                team_tasks[min_team].append(assignment)

        # If any team has no tasks assigned, add a fallback task to ensure they produce output for the pilot
        fallback_descriptions = {
            PM_TEAM: "generate user stories for the core email classification and routing features",
            PGM_TEAM: "define the product features and epics for the Email Router MVP",
            DEV_TEAM: "create engineering tasks for the email ingestion and AI classification modules",
        }
        for team_name, task_list in team_tasks.items():
            if not task_list:
                task_list.append({
                    "task": {
                        "id": f"fallback_{team_name}",
                        "description": fallback_descriptions[team_name],
                        "expected_output": "Structured deliverables",
                        "dependencies": [],
                    },
                    "assigned_team": team_name,
                    "routing_confidence": 1.0,
                    "routing_reasoning": "Fallback assignment",
                })

        # Execute tasks team by team
        all_outputs: list[dict] = []
        for team_name, assignments in team_tasks.items():
            print(f"\n{'─'*60}")
            print(f"📂 TEAM: {team_name.upper()} ({len(assignments)} task)")
            print(f"{'─'*60}")
            for assignment in assignments:
                result = self._execute_team_task(
                    team_name=team_name,
                    task_description=assignment["task"]["description"],
                    task_id=assignment["task"]["id"],
                )
                all_outputs.append(result)

        # ── PHASE 4: FINAL REPORT ─────────────────────────────────────────
        self._print_phase(
            "4/4 — FINAL REPORT",
            "Aggregation of all outputs in a structured report"
        )

        elapsed = time.time() - start_time
        report = self._build_report(tpm_request, tasks, all_outputs, elapsed)

        return report

    def _build_report(
        self,
        tpm_request: str,
        plan: list[dict],
        outputs: list[dict],
        elapsed: float,
    ) -> dict:
        """Assemble the final structured report."""

        # Group output by team
        by_team: dict[str, list] = {}
        for output in outputs:
            team = output["team"]
            by_team.setdefault(team, []).append(output)

        # Quality statistics
        scores = [o["evaluation"]["score"] for o in outputs]
        avg_score = sum(scores) / len(scores) if scores else 0
        all_passed = all(o["evaluation"]["passed"] for o in outputs)

        report = {
            "metadata": {
                "project": "Email Router — InnovateNext Solutions",
                "generated_at": datetime.now().isoformat(),
                "elapsed_seconds": round(elapsed, 1),
                "tpm_request": tpm_request,
            },
            "action_plan": plan,
            "quality_summary": {
                "overall_passed": all_passed,
                "average_score": round(avg_score, 1),
                "total_tasks_executed": len(outputs),
                "tasks_passed_first_attempt": sum(1 for o in outputs if o["attempts"] == 1),
            },
            "deliverables": {
                team: [
                    {
                        "task_id": o["task_id"],
                        "task": o["task_description"][:100],
                        "output": o["output"],
                        "score": o["evaluation"]["score"],
                        "passed": o["evaluation"]["passed"],
                    }
                    for o in team_outputs
                ]
                for team, team_outputs in by_team.items()
            },
        }

        return report

    def save_report(self, report: dict, output_dir: str = "output") -> str:
        """Save the report in JSON and Markdown format."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON (structured for integration with other systems)
        json_path = f"{output_dir}/report_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        # Markdown (leggibile dagli stakeholder)
        md_path = f"{output_dir}/report_{timestamp}.md"
        with open(md_path, "w") as f:
            f.write(self._render_markdown(report))

        print(f"\n💾 Report salvati:")
        print(f"   JSON: {json_path}")
        print(f"   Markdown: {md_path}")
        return md_path

    def _render_markdown(self, report: dict) -> str:
        """Converte il report in Markdown leggibile."""
        md = []
        meta = report["metadata"]
        qual = report["quality_summary"]

        md.append(f"# Email Router — Project Management Report")
        md.append(f"\n**Generated:** {meta['generated_at']}")
        md.append(f"**Elapsed:** {meta['elapsed_seconds']}s")
        md.append(f"**TPM Request:** {meta['tpm_request']}\n")

        md.append(f"## Quality Summary")
        md.append(f"- **Overall Passed:** {'✅' if qual['overall_passed'] else '⚠️'}")
        md.append(f"- **Average Score:** {qual['average_score']}/10")
        md.append(f"- **Tasks Executed:** {qual['total_tasks_executed']}")
        md.append(f"- **First-Attempt Pass Rate:** {qual['tasks_passed_first_attempt']}/{qual['total_tasks_executed']}\n")

        md.append(f"## Action Plan")
        for task in report["action_plan"]:
            md.append(f"- **[{task['id']}]** {task['description']}")
        md.append("")

        md.append(f"## Deliverables by Team")
        for team, deliverables in report["deliverables"].items():
            md.append(f"\n### {team.upper().replace('_', ' ')}")
            for d in deliverables:
                md.append(f"\n**Task [{d['task_id']}]** — Score: {d['score']}/10 {'✅' if d['passed'] else '⚠️'}")
                md.append(f"\n{d['output']}\n")
                md.append("---")

        return "\n".join(md)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Request for the TPM workflow (can be customized for different pilots)
    TPM_REQUEST = (
        """Plan the complete technical project management deliverables for the  
        Email Router product. We need: (1) user stories for the PM team, 
        (2) product feature definitions for the Program Manager team, and 
        (3) engineering tasks for the Development team. Use the provided 
        "product specification as the source of truth."""
    )

    # Path alla product spec
    spec_path = os.path.join(os.path.dirname(__file__), "Product-Spec-Email-Router.txt")

    if not os.path.exists(spec_path):
        print(f"❌ Product spec non trovata: {spec_path}")
        sys.exit(1)

    # Esegui il workflow
    workflow = ProjectManagementWorkflow(
        product_spec_path=spec_path,
        verbose=False,  # Metti True per vedere ogni chiamata API in dettaglio
    )

    report = workflow.run(tpm_request=TPM_REQUEST)

    # Stampa il summary
    print(f"\n{'='*70}")
    print("📊 WORKFLOW COMPLETATO — SUMMARY")
    print(f"{'='*70}")
    qual = report["quality_summary"]
    print(f"Overall Passed: {'✅ YES' if qual['overall_passed'] else '⚠️  PARTIAL'}")
    print(f"Average Quality Score: {qual['average_score']}/10")
    print(f"Tasks Executed: {qual['total_tasks_executed']}")
    print(f"Elapsed: {report['metadata']['elapsed_seconds']}s")

    print(f"\n📁 DELIVERABLES:")
    for team, deliverables in report["deliverables"].items():
        print(f"\n  {team.upper()}:")
        for d in deliverables:
            status = "✅" if d["passed"] else "⚠️ "
            print(f"    {status} [{d['task_id']}] score={d['score']}/10")
            print(f"       {d['output'][:120]}...")

    # Salva il report
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    md_path = workflow.save_report(report, output_dir=output_dir)

    print(f"\n✅ Done! Report Markdown: {md_path}")


if __name__ == "__main__":
    main()