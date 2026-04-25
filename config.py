import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Models to evaluate
MODELS = {
    "gpt-4o-mini": "openai",
    "claude-haiku": "anthropic",  # Maps to claude-haiku-4-5-20251001 in wrapper
    "gemini-flash": "google"      # Maps to gemini-1.5-flash in wrapper
}

# Datasets
DATASETS = ["sst2", "mmlu", "gsm8k"]
NUM_SAMPLES = 100

# Attacks
ATTACKS = [
    "textbugger",
    "deepwordbug",
    "textfooler",
    "checklist",
    "stresstest"
]

# Evaluation Settings
TEMPERATURE = 0.0
CACHE_DIR = "data/.cache"

# Paths
RESULTS_FILE = "data/results.csv"
CUSTOM_RESULTS_FILE = "data/custom_results.csv"
CHARTS_DIR = "charts"

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("attacks", exist_ok=True)
os.makedirs("evaluation", exist_ok=True)
