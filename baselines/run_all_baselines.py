"""Batch runner for all baseline experiments."""
import os
import argparse

from run_baseline import run_baseline


def main():
    parser = argparse.ArgumentParser(
        description="Run all three baseline experiments on multiple datasets"
    )
    parser.add_argument(
        "--data-dir",
        default="./data/reasoning_augmented",
        help="Directory containing input CSV files (default: ./data/reasoning_augmented)"
    )
    parser.add_argument(
        "--output-dir",
        default="./baseline_results",
        help="Directory for output CSV files (default: ./baseline_results)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["laptop_test_reasoning.csv", "restaurant_test_reasoning.csv"],
        help="CSV files to process"
    )
    parser.add_argument(
        "--prompt-types",
        nargs="+",
        default=["zero-shot", "few-shot", "cot"],
        help="Prompting strategies to run"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model to use"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for dataset in args.datasets:
        input_path = os.path.join(args.data_dir, dataset)

        if not os.path.exists(input_path):
            print(f"Warning: Dataset not found: {input_path}")
            continue

        dataset_name = dataset.split("_")[0]

        for prompt_type in args.prompt_types:
            output_file = os.path.join(
                args.output_dir,
                f"{dataset_name}_{prompt_type}_predictions.csv"
            )

            try:
                run_baseline(
                    input_file=input_path,
                    output_file=output_file,
                    prompt_type=prompt_type,
                    model=args.model
                )
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
            except Exception as e:
                print(f"Error running {prompt_type} on {dataset}: {e}")
                continue


if __name__ == "__main__":
    main()
