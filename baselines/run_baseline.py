"""Baseline sentiment analysis using Azure OpenAI API."""
import sys
import os
import argparse
from typing import Callable, Optional
import pandas as pd
from openai import AzureOpenAI
from sklearn.metrics import accuracy_score, f1_score

from prompts import get_zero_shot_prompt, get_few_shot_prompt, get_cot_prompt


def init_azure_client() -> AzureOpenAI:
    """Initialize Azure OpenAI client from environment variables."""
    api_key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

    if not api_key or not endpoint:
        raise ValueError(
            "Please set AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT environment variables"
        )

    return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)


def clean_sentiment_output(raw: Optional[str]) -> str:
    """Extract sentiment label from model output."""
    if raw is None:
        return "neutral"

    raw = raw.strip().lower()
    if "positive" in raw:
        return "positive"
    if "negative" in raw:
        return "negative"
    return "neutral"


def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV data file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    return pd.read_csv(csv_path)


def save_results(df: pd.DataFrame, output_path: str):
    """Save predictions to CSV file."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


def get_domain_name(dataset_name: str) -> str:
    """Extract domain name from dataset filename."""
    dataset_lower = dataset_name.lower()
    if "laptop" in dataset_lower:
        return "laptop"
    elif "restaurant" in dataset_lower:
        return "restaurant"
    return "general"


PROMPT_GETTERS = {
    "zero-shot": get_zero_shot_prompt,
    "few-shot": get_few_shot_prompt,
    "cot": get_cot_prompt,
}


def predict_sentiment(
    client: AzureOpenAI,
    text: str,
    target: str,
    domain: str,
    prompt_getter: Callable,
    model: str = "gpt-4o",
) -> str:
    """Call API to predict sentiment."""
    prompt = prompt_getter(text=text, target=target, domain=domain)

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful sentiment analysis engine."},
                {"role": "user", "content": prompt}
            ]
        )
        raw_output = response.choices[0].message.content
        return clean_sentiment_output(raw_output)

    except Exception as e:
        print(f"Error during API call: {e}")
        return "neutral"


def run_baseline(
    input_file: str,
    output_file: str,
    prompt_type: str,
    model: str = "gpt-4o",
):
    """Run baseline experiment with specified prompting strategy."""
    print(f"\n{'='*60}")
    print(f"Running {prompt_type.upper()} Baseline")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")

    df = load_data(input_file)
    domain = get_domain_name(input_file)
    prompt_getter = PROMPT_GETTERS[prompt_type]

    client = init_azure_client()
    predictions = []

    for idx, row in df.iterrows():
        print(f"[{idx+1}/{len(df)}] Processing review...", end="\r")

        text = row.get("text", "")
        target = row.get("aspect", "")

        sentiment = predict_sentiment(
            client=client,
            text=text,
            target=target,
            domain=domain,
            prompt_getter=prompt_getter,
            model=model
        )
        predictions.append(sentiment)

    df["predicted_sentiment"] = predictions
    save_results(df, output_file)

    print(f"\nCompleted {len(predictions)} predictions")

    if "sentiment" in df.columns:
        print(f"\n{'='*60}")
        print("EVALUATION METRICS")
        print(f"{'='*60}")

        y_true = df["sentiment"].str.lower()
        y_pred = df["predicted_sentiment"].str.lower()

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-score (macro): {f1:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Baseline sentiment analysis using LLM prompting strategies"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file with 'text' and 'aspect' columns"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV file for predictions"
    )
    parser.add_argument(
        "--prompt-type",
        choices=["zero-shot", "few-shot", "cot"],
        default="zero-shot",
        help="Prompting strategy to use (default: zero-shot)"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model name to use (default: gpt-4o)"
    )

    args = parser.parse_args()

    try:
        run_baseline(
            input_file=args.input,
            output_file=args.output,
            prompt_type=args.prompt_type,
            model=args.model
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
