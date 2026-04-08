"""
test_routing.py — Test del RoutingAgent
========================================
CONCETTI (Fase 4.1):
- Pattern orchestratore: il routing smista il lavoro
- temperature=0.1 per routing deterministico
- Output: route + confidence + reasoning
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workflow_agents import RoutingAgent

ROUTES = {
    "pm_team": "Handles user stories, acceptance criteria, and end-user requirements",
    "pgm_team": "Handles product features, epics, and release planning",
    "dev_team": "Handles engineering tasks, technical specs, and implementation",
}

TEST_TASKS = [
    "Write user stories for the email priority detection feature",
    "Define the epics and milestones for the Email Router v1.0 release",
    "Design the database schema for storing routing decisions and audit logs",
]

def main():
    print("=" * 60)
    print("TEST: RoutingAgent")
    print("=" * 60)
    print("Concetto: il router decide a chi mandare ogni task.\n")

    router = RoutingAgent(routes=ROUTES, verbose=False)  # verbose=False: output pulito

    for task in TEST_TASKS:
        print(f"\nTASK: {task}")
        result = router.run(task)
        print(f"  → ROUTE: {result['route']}")
        print(f"  → CONFIDENCE: {result['confidence']:.2f}")
        print(f"  → REASONING: {result['reasoning']}")

    print("\n✅ RoutingAgent: test completato")

if __name__ == "__main__":
    main()
