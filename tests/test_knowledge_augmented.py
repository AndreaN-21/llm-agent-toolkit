"""
test_knowledge_augmented.py — Test del KnowledgeAugmentedPromptAgent 
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workflow_agents import KnowledgeAugmentedPromptAgent

SPEC_EXCERPT = """
EMAIL ROUTER — Product Spec (excerpt)
Core Features:
1. Auto-classification: Support, Billing, Sales, Legal, Spam, Internal
2. Priority: P1 Critical, P2 High, P3 Medium, P4 Low
3. Sender Enrichment via CRM (Salesforce)
4. Duplicate Detection (Zendesk integration)
Target: < 2s processing, > 92% accuracy, handle 5000 emails/day
"""

SYSTEM_PROMPT = """You are a senior Product Manager who writes user stories.
Format: As a [user], I want [action], so that [benefit].
Always include 3 acceptance criteria per story."""

def main():
    print("=" * 60)
    print("TEST: KnowledgeAugmentedPromptAgent")
    print("=" * 60)
    print("Concetto: l'agent conosce il tuo prodotto grazie alla knowledge base.\n")

    agent = KnowledgeAugmentedPromptAgent(
        system_prompt=SYSTEM_PROMPT,
        knowledge_docs=[SPEC_EXCERPT],
        verbose=True,
    )
    result = agent.run(
        "Write 2 user stories for the email classification feature."
    )
    print(f"\nRISULTATO:\n{result}")
    print("\n✅ KnowledgeAugmentedPromptAgent: test completato")

if __name__ == "__main__":
    main()