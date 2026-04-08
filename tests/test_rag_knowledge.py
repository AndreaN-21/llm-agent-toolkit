"""
test_rag_knowledge.py — Test del RAGKnowledgePromptAgent
=========================================================
CONCETTI (Fase 1.6 + 4.3):
- Embeddings simulati (bag-of-words normalizzato)
- Cosine similarity per retrieval
- Solo i chunk rilevanti entrano nel contesto
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workflow_agents import RAGKnowledgePromptAgent

KNOWLEDGE_CHUNKS = [
    "The Email Router classifies emails into: Support, Billing, Sales, Legal, Spam, Internal categories.",
    "Priority levels: P1 Critical (< 2h SLA), P2 High, P3 Medium, P4 Low. P1 triggers Slack alert.",
    "Sender enrichment: system looks up sender in Salesforce CRM and attaches customer tier and health score.",
    "Duplicate detection: system checks if email is part of an existing Zendesk thread.",
    "Performance: process each email in < 2 seconds. Target accuracy > 92%. Handle 5000 emails/day.",
    "Out of scope for v1.0: auto-reply generation, sentiment analysis, multi-language support.",
]

SYSTEM_PROMPT = "You are a helpful assistant that answers questions about the Email Router product."

def main():
    print("=" * 60)
    print("TEST: RAGKnowledgePromptAgent")
    print("=" * 60)
    print("Concetto: retrieval semantico — solo i chunk più rilevanti.\n")

    agent = RAGKnowledgePromptAgent(
        system_prompt=SYSTEM_PROMPT,
        knowledge_chunks=KNOWLEDGE_CHUNKS,
        top_k=2,
        verbose=True,
    )

    query = "What are the performance requirements for the email router?"
    print(f"QUERY: {query}\n")
    result = agent.run(query)
    print(f"\nRISULTATO:\n{result}")
    print("\n✅ RAGKnowledgePromptAgent: test completato")

if __name__ == "__main__":
    main()
