"""
test_augmented_prompt.py — Test dell'AugmentedPromptAgent 
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workflow_agents import AugmentedPromptAgent

SYSTEM_PROMPT = """You are a concise technical writer.
Always respond in this exact JSON format:
{
  "summary": "<one sentence summary>",
  "key_points": ["<point 1>", "<point 2>", "<point 3>"],
  "recommendation": "<one actionable recommendation>"
}
Never include explanations outside the JSON."""

def main():
    print("=" * 60)
    print("TEST: AugmentedPromptAgent")
    print("=" * 60)
    print("Concetto: system prompt configura il ruolo e il formato.\n")

    agent = AugmentedPromptAgent(system_prompt=SYSTEM_PROMPT, verbose=True)
    result = agent.run(
        "Analyze the importance of email routing automation for customer support teams."
    )
    print(f"\nRISULTATO:\n{result}")
    print("\n✅ AugmentedPromptAgent: test completato")

if __name__ == "__main__":
    main()