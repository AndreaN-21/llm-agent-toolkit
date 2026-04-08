"""
test_direct_prompt.py — Test del DirectPromptAgent
===================================================
CONCETTI (Fase 1):
- Prima chiamata all'API Anthropic
- System prompt vs user message (qui: solo user message)
- temperature e max_tokens
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workflow_agents import DirectPromptAgent

def main():
    print("=" * 60)
    print("TEST: DirectPromptAgent")
    print("=" * 60)
    print("Concetto: prompt diretto, nessun contesto aggiunto.")
    print("L'agent è il più semplice possibile.\n")

    agent = DirectPromptAgent(verbose=True)

    prompt = (
        "List 3 key challenges of building an email routing system "
        "that uses AI for classification. Be concise."
    )
    print(f"PROMPT: {prompt}\n")
    result = agent.run(prompt)
    print(f"\nRISULTATO:\n{result}")
    print("\n✅ DirectPromptAgent: test completato")

if __name__ == "__main__":
    main()
