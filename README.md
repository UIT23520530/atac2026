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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
