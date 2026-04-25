from models import GPTWrapper, ClaudeWrapper, GeminiWrapper, GroqWrapper, HFWrapper
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY, HF_TOKEN

def test_models():
    prompt = "Explain what machine learning is in exactly 10 words."
    print(f"Testing Prompt: '{prompt}'\n")

    test_configs = [
        ("GPT-4o-mini", GPTWrapper, "gpt-4o-mini", OPENAI_API_KEY),
        ("Claude Haiku", ClaudeWrapper, "claude-3-5-haiku-latest", ANTHROPIC_API_KEY),
        ("Gemini Flash", GeminiWrapper, "gemini-1.5-flash", GOOGLE_API_KEY),
        ("Groq Llama-3", GroqWrapper, "llama3-8b-8192", GROQ_API_KEY),
        ("HF Mistral-7B", HFWrapper, "mistralai/Mistral-7B-v0.1", HF_TOKEN)
    ]

    for name, wrapper_class, model_name, api_key in test_configs:
        if api_key or name == "HF Mistral-7B": # HF might work without token (rate limited)
            print(f"Testing {name}...")
            try:
                wrapper = wrapper_class(model_name=model_name)
                response = wrapper.generate(prompt)
                print("Response:", response)
            except Exception as e:
                print(f"Error testing {name}: {e}")
            print("-" * 40)
        else:
            print(f"Skipping {name} test (API key missing)\n")

if __name__ == "__main__":
    test_models()
