from models import GPTWrapper, ClaudeWrapper, GeminiWrapper
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY

def test_models():
    prompt = "Explain what machine learning is in exactly 10 words."
    print(f"Testing Prompt: '{prompt}'\n")

    if OPENAI_API_KEY:
        print("Testing GPT-4o-mini...")
        gpt = GPTWrapper(model_name="gpt-4o-mini")
        print("Response:", gpt.generate(prompt))
        print("-" * 40)
    else:
        print("Skipping GPT test (OPENAI_API_KEY missing)\n")

    if ANTHROPIC_API_KEY:
        print("Testing Claude Haiku...")
        claude = ClaudeWrapper(model_name="claude-3-5-haiku-latest") # or claude-haiku-4-5-20251001
        print("Response:", claude.generate(prompt))
        print("-" * 40)
    else:
        print("Skipping Claude test (ANTHROPIC_API_KEY missing)\n")

    if GOOGLE_API_KEY:
        print("Testing Gemini Flash...")
        gemini = GeminiWrapper(model_name="gemini-1.5-flash")
        print("Response:", gemini.generate(prompt))
        print("-" * 40)
    else:
        print("Skipping Gemini test (GOOGLE_API_KEY missing)\n")

if __name__ == "__main__":
    test_models()
