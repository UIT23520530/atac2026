# Reasoning-Augmented Fine-Tuning of Gemma 7B for Implicit Sentiment Analysis

This repository contains the source code and experimental results for our research on **Implicit Sentiment Analysis (ISA)** using **Gemma 7B** with LoRA fine-tuning. We propose a **Reasoning-Augmented Fine-Tuning** approach that leverages GPT-4o-generated Chain-of-Thought (CoT) reasoning to improve implicit sentiment detection.

## Overview

Implicit Sentiment Analysis is the task of identifying sentiment when no explicit opinion words are present (e.g., *"The laptop weighs 5 pounds"* → negative toward **weight**). This is significantly harder than explicit sentiment analysis.

We investigate two fine-tuning strategies:
1. **Standard Fine-Tuning**: Direct `Sentence + Aspect → Sentiment` mapping
2. **Reasoning-Augmented Fine-Tuning (CoT)**: `Sentence + Aspect + GPT-4o Expert Analysis → Sentiment`, where GPT-4o provides step-by-step reasoning about implicit cues

Both approaches use **Gemma 7B** as the base model with **LoRA adapters** for parameter-efficient fine-tuning.

## Repository Structure

```
├── ISA_Standard_FineTuning_Gemma.ipynb     # Standard fine-tuning pipeline
├── ISA_Reasoning_FineTuning_Gemma.ipynb    # Reasoning-augmented (CoT) fine-tuning pipeline
├── baselines/                              # Baseline implementations (zero-shot, few-shot, CoT)
│   ├── run_baseline.py                     # Main baseline runner
│   ├── run_all_baselines.py                # Batch runner for all experiments
│   └── prompts.py                          # Prompt templates
├── data/
│   ├── laptop/                             # SemEval-style XML data (Laptop domain)
│   ├── restaurant/                         # SemEval-style XML data (Restaurant domain)
│   └── reasoning_augmented/                # CSV data with GPT-4o reasoning annotations
├── evaluation_results/                     # Evaluation metrics (Standard & Reasoning)
├── inference_results/                      # Model predictions on test sets (CSV)
├── LICENSE
└── README.md
```

## Methodology

### Standard Fine-Tuning
- Fine-tune Gemma 7B with LoRA on aspect-level sentiment classification
- Input format: `Instruction + Sentence + Aspect → Sentiment`

### Reasoning-Augmented Fine-Tuning (CoT)
1. **Reasoning Generation**: Use GPT-4o to generate expert linguistic analysis for each sample, following a structured 3-step guideline:
   - *Contextual Anchoring & Domain Standards*
   - *Evidence Extraction (Explicit & Implicit cues)*
   - *Synthesis & Knowledge Transfer*
2. **Fine-Tuning**: Train Gemma 7B with the enriched input: `Sentence + Aspect + Expert Analysis → Sentiment`
3. **Inference**: Evaluate on test sets that also include GPT-4o reasoning

### Training Configuration

Both approaches share the same hyperparameters:

| Parameter | Value |
|-----------|-------|
| Base Model | `google/gemma-7b` (BFloat16) |
| LoRA Rank | 64 |
| LoRA Alpha | 128 |
| LoRA Dropout | 0.1 |
| Learning Rate | 2e-4 |
| LR Scheduler | Cosine |
| Epochs | 4 |
| Batch Size | 8 |
| Gradient Accumulation Steps | 2 |
| Max Sequence Length | 1024 |

## Getting Started

### Prerequisites
- Python 3.10+
- NVIDIA GPU with >= 40GB VRAM (A100 recommended)
- Google Colab Pro (recommended)
- Hugging Face account with Gemma model access

### Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft bitsandbytes trl accelerate scikit-learn openai tqdm
```

### Usage

1. **Standard Fine-Tuning**: Open and run `ISA_Standard_FineTuning_Gemma.ipynb`
2. **Reasoning-Augmented Fine-Tuning**: Open and run `ISA_Reasoning_FineTuning_Gemma.ipynb`

> **Note**: Both notebooks were originally designed for Google Colab with Google Drive integration. You may need to adjust file paths for local execution.

### Required Credentials
- Set your **Hugging Face token** in the notebook configuration cell (requires Gemma model access)
- For reasoning generation: Set your **OpenAI / Azure OpenAI API key** in the Reasoning notebook

## Data

The datasets are based on **SemEval 2014 Task 4** (Laptop and Restaurant reviews), extended with **implicit sentiment annotations**. Each aspect-term pair is labeled with:
- `sentiment`: positive / negative / neutral
- `is_implicit`: whether the sentiment is expressed implicitly

The `reasoning_augmented/` directory contains the same data enriched with GPT-4o-generated expert analysis.

## Baseline Experiments

We provide three baseline implementations using Azure OpenAI API (GPT-4o) to evaluate different prompting strategies:

### Baseline Methods

1. **Zero-shot**: Simple task instruction without examples
2. **Few-shot**: Task instruction with 3 domain-specific examples
3. **Chain-of-Thought (CoT)**: Step-by-step reasoning instruction

### Prerequisites

- Python 3.10+
- Azure OpenAI API access

### Installation

```bash
pip install pandas scikit-learn openai
```

### Configuration

Set the following environment variables:

```bash
export AZURE_OPENAI_KEY="your_api_key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_VERSION="2024-05-01-preview"
```

### Usage

**Run a specific baseline:**

```bash
python baselines/run_baseline.py \
    --input ./data/reasoning_augmented/laptop_test_reasoning.csv \
    --output ./baseline_results/laptop_zero_shot.csv \
    --prompt-type zero-shot
```

**Run all three baselines on all datasets:**

```bash
python baselines/run_all_baselines.py \
    --data-dir ./data/reasoning_augmented \
    --output-dir ./baseline_results
```

### Arguments

**run_baseline.py:**
- `--input` (required): Path to input CSV with 'text' and 'aspect' columns
- `--output` (required): Path for output CSV with predictions
- `--prompt-type` (required): One of `zero-shot`, `few-shot`, `cot`
- `--model` (optional): Model name (default: `gpt-4o`)

**run_all_baselines.py:**
- `--data-dir`: Directory with input CSVs (default: `./data/reasoning_augmented`)
- `--output-dir`: Directory for outputs (default: `./baseline_results`)
- `--datasets`: CSV files to process
- `--prompt-types`: Strategies to run

### Output Format

Results are saved as CSV with:
```
text, aspect, sentiment, is_implicit, reasoning, predicted_sentiment
```

### Evaluation Metrics

When ground truth labels are present, the following metrics are displayed:
- **Accuracy**
- **F1-score** (macro-averaged)

### Prompt Designs

**Zero-shot:** Simple instruction without examples

**Few-shot:** Instruction with 3 domain-specific examples (different for laptop vs restaurant domains)

**CoT:** Instruction to reason step-by-step:
1. Analyze text for implicit opinions and pragmatic cues
2. Consider contextual signals (sarcasm, understatement, contrast)
3. Infer underlying sentiment toward target


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
