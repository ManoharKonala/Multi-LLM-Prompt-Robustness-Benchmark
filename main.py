"""
Multi-LLM Prompt Robustness Benchmark
======================================
Entry point supporting two modes:
  - benchmark : automated evaluation on SST2, MMLU, GSM8K
  - custom    : interactive prompt testing with side-by-side comparison
"""

import argparse
import sys
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

from config import MODELS, DATASETS, NUM_SAMPLES, ATTACKS, RESULTS_FILE, CUSTOM_RESULTS_FILE
from models import GPTWrapper, ClaudeWrapper, GeminiWrapper, GroqWrapper, HFWrapper
from attacks import generate_attacked_prompts
from evaluation.scorer import calculate_robustness, evaluate_custom_prompt
from evaluation.visualizer import (
    print_custom_mode_table,
    plot_accuracy_comparison,
    plot_robustness_heatmap,
)


# ──────────────────────────────────────────────────────────────────────────────
# Model Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_models() -> dict:
    """Initializes and returns the model wrappers."""
    wrappers = {}
    if "gpt-4o-mini" in MODELS:
        wrappers["gpt-4o-mini"] = GPTWrapper(model_name="gpt-4o-mini")
    if "claude-haiku" in MODELS:
        wrappers["claude-haiku"] = ClaudeWrapper(model_name="claude-3-5-haiku-latest")
    if "gemini-flash" in MODELS:
        wrappers["gemini-flash"] = GeminiWrapper(model_name="gemini-2.0-flash")
    if "llama-3-8b" in MODELS:
        wrappers["llama-3-8b"] = GroqWrapper(model_name="llama-3.3-70b-versatile")
    if "mistral-7b" in MODELS:
        wrappers["mistral-7b"] = HFWrapper(model_name="mistralai/Mistral-7B-Instruct-v0.2")
    return wrappers


def _model_is_available(wrapper) -> bool:
    """Check if a model wrapper has valid credentials."""
    # GPT & Claude wrappers store a `client`; Gemini stores a `model`
    return (getattr(wrapper, "client", None) is not None
            or getattr(wrapper, "model", None) is not None)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_dataset_samples(dataset_name: str, n: int):
    """
    Load *n* samples from a HuggingFace dataset and return a list of dicts
    with keys  { "prompt", "label" }.
    """
    samples = []

    if dataset_name == "sst2":
        ds = load_dataset("glue", "sst2", split="validation")
        label_map = {0: "negative", 1: "positive"}
        for row in ds.select(range(min(n, len(ds)))):
            samples.append({
                "prompt": (
                    f'Classify the sentiment of the following sentence as '
                    f'"positive" or "negative". '
                    f'Sentence: "{row["sentence"]}"'
                ),
                "label": label_map[row["label"]],
            })

    elif dataset_name == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        choices_letters = ["A", "B", "C", "D"]
        for row in ds.select(range(min(n, len(ds)))):
            choices_text = "\n".join(
                f"  {letter}. {choice}"
                for letter, choice in zip(choices_letters, row["choices"])
            )
            samples.append({
                "prompt": (
                    f"Answer the following multiple-choice question. "
                    f"Reply with ONLY the letter (A, B, C, or D).\n\n"
                    f"Question: {row['question']}\n{choices_text}"
                ),
                "label": choices_letters[row["answer"]],
            })

    elif dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        for row in ds.select(range(min(n, len(ds)))):
            # Ground truth is the final numeric answer after "####"
            answer_text = row["answer"].split("####")[-1].strip()
            samples.append({
                "prompt": (
                    f"Solve the following math problem step by step. "
                    f"End your answer with the final number only.\n\n"
                    f"{row['question']}"
                ),
                "label": answer_text,
            })
    else:
        print(f"Unknown dataset: {dataset_name}")

    return samples


def _check_answer(response: str | None, label: str, dataset_name: str) -> bool:
    """
    Check whether the model response contains the correct answer.
    Simple substring / normalised matching per dataset type.
    """
    if response is None:
        return False

    response_lower = response.lower().strip()
    label_lower = label.lower().strip()

    if dataset_name == "sst2":
        return label_lower in response_lower

    elif dataset_name == "mmlu":
        # Look for the letter at the start or as a standalone token
        # e.g. "A", "A.", "(A)", "The answer is A"
        import re
        pattern = rf"(?:^|[\s(])({re.escape(label_lower)})(?:[\s).]|$)"
        return bool(re.search(pattern, response_lower))

    elif dataset_name == "gsm8k":
        # BUG FIX: Use regex to match the exact number or check for word boundaries
        # to prevent "5" from matching "15".
        import re
        pattern = rf"\b{re.escape(label_lower)}\b"
        return bool(re.search(pattern, response_lower))

    return label_lower in response_lower


# ──────────────────────────────────────────────────────────────────────────────
# Custom Prompt Mode
# ──────────────────────────────────────────────────────────────────────────────

def run_custom_mode():
    print("========================================")
    print("  Multi-LLM Prompt Robustness Benchmark")
    print("========================================")

    prompt = input("\nEnter your prompt: ")
    if not prompt.strip():
        print("Prompt cannot be empty.")
        return

    print(f"\nGenerating {len(ATTACKS)} attacked versions of your prompt...")
    attacked_prompts = generate_attacked_prompts(prompt, ATTACKS)

    # Show generated attacks
    for name, text in attacked_prompts.items():
        if name != "Original":
            print(f"  {name:15s}: {text[:80]}{'...' if len(text) > 80 else ''}")

    models = get_models()
    results: dict[str, dict[str, float]] = {m: {} for m in models}
    all_responses: list[dict] = []

    for model_name, wrapper in models.items():
        if not _model_is_available(wrapper):
            print(f"Skipping {model_name} (API key missing)")
            continue

        print(f"Running on {model_name}...", end="", flush=True)

        # Baseline
        orig_response = wrapper.generate(prompt)
        if orig_response is None:
            print(" Skipping (baseline failed)")
            continue
            
        results[model_name]["Original"] = 1.0
        all_responses.append({
            "Model": model_name,
            "Attack Type": "Original",
            "Prompt": prompt,
            "Response": orig_response,
            "Similarity": 1.0,
        })

        # Attacked variants
        for attack_name, attack_prompt in attacked_prompts.items():
            if attack_name == "Original":
                continue
            response = wrapper.generate(attack_prompt)
            score = evaluate_custom_prompt(orig_response, response)
            results[model_name][attack_name] = score
            all_responses.append({
                "Model": model_name,
                "Attack Type": attack_name,
                "Prompt": attack_prompt,
                "Response": response,
                "Similarity": score,
            })

        print(" Done")

    # Save
    print(f"\nSaving results to {CUSTOM_RESULTS_FILE}...")
    df = pd.DataFrame(all_responses)
    df.to_csv(CUSTOM_RESULTS_FILE, index=False)

    # Display
    print_custom_mode_table(prompt, results)
    print("Done!")


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark Mode
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark_mode(num_samples_override: int = None):
    """
    Runs the automated benchmark across all configured datasets and models.
    Saves results to a CSV and generates visual heatmaps/charts.
    """
    print("========================================")
    print("  Multi-LLM Prompt Robustness Benchmark")
    print("       ── Benchmark Mode ──")
    print("========================================\n")

    models = get_models()
    all_results: list[dict] = []
    
    samples_to_load = num_samples_override or NUM_SAMPLES

    for dataset_name in DATASETS:
        print(f"\n{'─'*50}")
        print(f"Loading dataset: {dataset_name}")
        samples = _load_dataset_samples(dataset_name, samples_to_load)
        if not samples:
            print(f"  ⚠ No samples loaded for {dataset_name}, skipping.")
            continue
        print(f"  Loaded {len(samples)} samples.")

        for model_name, wrapper in models.items():
            if not _model_is_available(wrapper):
                print(f"  Skipping {model_name} (API key missing)")
                continue

            print(f"\n  Evaluating {model_name} on {dataset_name}...")

            correct_clean = 0
            correct_attacked = {a: 0 for a in ATTACKS}

            for sample in tqdm(samples, desc=f"    {model_name}", leave=False):
                base_prompt = sample["prompt"]
                label = sample["label"]

                # ── Clean baseline ──
                orig_resp = wrapper.generate(base_prompt)
                if _check_answer(orig_resp, label, dataset_name):
                    correct_clean += 1

                # ── Attacked variants ──
                attacked = generate_attacked_prompts(base_prompt, ATTACKS)
                for attack_name in ATTACKS:
                    att_prompt = attacked.get(attack_name, base_prompt)
                    att_resp = wrapper.generate(att_prompt)
                    if _check_answer(att_resp, label, dataset_name):
                        correct_attacked[attack_name] += 1
                print(" Done")
            total = len(samples)
            clean_acc = correct_clean / total if total else 0

            for attack_name in ATTACKS:
                att_acc = correct_attacked[attack_name] / total if total else 0
                robustness, drop = calculate_robustness(clean_acc, att_acc)

                row = {
                    "Model": model_name,
                    "Dataset": dataset_name,
                    "Attack Type": attack_name,
                    "Clean Accuracy": round(clean_acc * 100, 1),
                    "Attacked Accuracy": round(att_acc * 100, 1),
                    "Drop Rate": round(drop * 100, 1),
                    "Robustness Score": round(robustness, 1),
                }
                all_results.append(row)

                print(
                    f"    {attack_name:15s}  "
                    f"clean={clean_acc*100:5.1f}%  "
                    f"attacked={att_acc*100:5.1f}%  "
                    f"robustness={robustness:5.1f}%"
                )

            # Save progressively after each model
            pd.DataFrame(all_results).to_csv(RESULTS_FILE, index=False)

    print(f"\n{'─'*50}")
    print(f"Benchmark complete. Results saved to {RESULTS_FILE}.")

    # ── Charts ──
    if all_results:
        print("Generating charts...")
        df = pd.DataFrame(all_results)
        plot_accuracy_comparison(df)
        plot_robustness_heatmap(df)
        print("Charts saved to charts/ directory.")
    else:
        print("No results to chart (check API keys).")


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-LLM Prompt Robustness Benchmark (PRB)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["benchmark", "custom"],
        required=True,
        help="Run mode: 'benchmark' for datasets, 'custom' for interactive prompt",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples per dataset (overrides config.py)",
    )
    args = parser.parse_args()

    if args.mode == "custom":
        run_custom_mode()
    elif args.mode == "benchmark":
        run_benchmark_mode(num_samples_override=args.samples)
