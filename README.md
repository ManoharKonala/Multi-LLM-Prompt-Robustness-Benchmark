# Multi-LLM Prompt Robustness Benchmark

A Python-based benchmarking system that tests how robustly different LLMs (GPT-4o-mini, Claude Haiku, Gemini Flash) handle prompt variations and adversarial attacks.

## Modes

The system supports TWO modes:

1. **Benchmark Mode** — Uses built-in datasets (SST2, MMLU, GSM8K) with automatic scoring.
2. **Custom Prompt Mode** — User types their own prompt, the system attacks it, sends all versions to all LLMs, and shows how each model responds side by side.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **API Keys**:
   Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add:
   ```env
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   GOOGLE_API_KEY=your_key_here
   ```

3. **Verify API Connections**:
   Run the test script to make sure your API keys are working:
   ```bash
   python test_models.py
   ```

## Usage

### Custom Prompt Mode
Type your own prompt and see how models react to typos, paraphrasing, and adversarial attacks.

```bash
python main.py --mode custom
```

### Benchmark Mode
Run the full benchmark across 3 datasets and 5 attack types.

```bash
python main.py --mode benchmark
```

## Results

- `data/custom_results.csv` and `data/results.csv` contain the raw data.
- `charts/` will contain bar charts and heatmaps visualizing the robustness of each model.
