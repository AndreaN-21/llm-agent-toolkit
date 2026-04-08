"""
test_action_planning.py — Test dell'ActionPlanningAgent
========================================================
CONCETTI (Fase 4.5):
- Plan-and-Execute: pianifica prima, esegui dopo
- Task decomposition: goal → sub-task atomici
- Output strutturato con dipendenze
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workflow_agents import ActionPlanningAgent

def main():
    print("=" * 60)
    print("TEST: ActionPlanningAgent")
    print("=" * 60)
    print("Concetto: scompone un goal complesso in sub-task eseguibili.\n")

    planner = ActionPlanningAgent(verbose=True)

    goal = (
        "Plan the technical project management deliverables for the Email Router: "
        "user stories, product features, and engineering tasks."
    )
    print(f"GOAL: {goal}\n")

    tasks = planner.run(goal=goal)

    print(f"\nPIANO GENERATO: {len(tasks)} task")
    for t in tasks:
        print(f"\n  [{t['id']}] {t['description'][:80]}")
        print(f"       Output atteso: {t.get('expected_output', 'N/A')[:60]}")
        deps = t.get('dependencies', [])
        if deps:
            print(f"       Dipendenze: {deps}")

    print("\n✅ ActionPlanningAgent: test completato")

if __name__ == "__main__":
    main()
