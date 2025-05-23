# Learned Retrieval for Code Completion

This repository contains the implementation of a retrieval-augmented code completion system that uses learned retrieval to select the most relevant context files for improving code completion accuracy.

## Project Overview

In our work, we use the [Long Code Arena's Project-Level Code Completion dataset for Python language](https://huggingface.co/datasets/JetBrains-Research/lca-project-level-code-completion). LCA is a benchmark suite designed specifically for evaluating long-context code models on various software engineering tasks. We utilize this dataset because it provides realistic scenarios where completing code in one file often requires understanding dependencies and patterns from other files in the same repository. This dataset is particularly well-suited for our research as it preserves the entire repository structure, allowing us to test how different retrieval strategies affect completion quality when models have access to cross-file context. LCA also includes helpful baselines like no-context and path-distance retrieval, enabling direct comparison with our approaches.

We evaluate completion quality using the Exact Match metric that assesses whether a generated completion line perfectly matches the ground truth line. We focus exclusively on the performance of the [deepseek-coder](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base) model for generating full line code completion.

We explore two key aspects of retrieval-augmented code completion:

1. **Estimating the Exact Match Upper Bound**: We use a brute-force approach to measure how much we can improve completion Exact Match (EM) by adding a single relevant context file to the completion prefix. By exhaustively testing all possible context files, we establish an upper bound on the potential performance gain when the system has access to the optimal context.

2. **Learned Retrieval**: We develop and evaluate a **Small Approximation Model** specifically trained to select the most relevant context file for a given completion prefix. We use the `Salesforce/codet5p-220m` encoder as our base model and train it to predict the similarity between potential context files and the completion prefix, learning to identify which files contain the information most helpful for generating correct completions.

### Small Approximation Model

We created two distinct datasets derived from the original LCA dataset: one based on the `large_context` portion for training purposes, and another based on the `medium_context` portion for validation and final evaluation.

#### Model Parameters

| Parameter | Value |
|-----------|-------|
| Base model | Salesforce/codet5p-220m encoder |
| Maximum sequence length | 512 tokens |

#### Training Parameters

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Optimizer | Adam |
| Scheduler | Cosine schedule with warmup |
| Accumulation steps | 8 |
| Learning rate | 2e-4 |
| Warmup steps | 512 |
| Number of epochs | 2 |
| Loss functions | MSE, Cross-Entropy |

## Getting Started

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (recommended for faster inference and training)

### Installation

1. Clone the repository:
```bash
# SSH
git clone ssh://git@git.jetbrains.team/ml-4-se-lab/learned-retrieval.git
# or HTTPS
git clone https://git.jetbrains.team/ml-4-se-lab/learned-retrieval.git
```

2. Create and activate a virtual environment (optional but recommended):
```bash
# Using venv
python -m venv venv
source venv/bin/activate 

# Or using conda
conda create -n learned-retrieval python=3.9
conda activate learned-retrieval
```

3. Install the package and its dependencies:
```bash
pip install -e .
```

This will install all the required dependencies listed in `requirements.txt`.

## Estimating the Exact Match Upper Bound

To calculate the upper-bound of completion Exact Match (EM) improvement with optimal context selection, run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python -m learned_retrieval.collect_generated_data.run \
    --model_name deepseek-ai/deepseek-coder-1.3b-base \
    --device cuda \
    --with_context_files True \
    --config_name medium_context \
    --max_seq_len 16356 \
    --max_completion_len 100 \
    --wandb_project_name lca-eval \
    --vllm True
```

This command:
1. Uses the deepseek-coder model to generate completions
2. Evaluates completions both with and without context files
3. For each completion point, tries all possible context files to find the one that maximizes accuracy
4. Reports the Exact Match (EM) metric for completions with and without context

The upper-bound is calculated by exhaustively testing all possible context files for each completion point and selecting the one that produces the best result. This establishes the maximum potential improvement that can be achieved with perfect context selection.

## Finetuning the Retrieval Model

To finetune the retrieval model that selects the most relevant context file:

1. First, preprocess the data:
```bash
python -m learned_retrieval.scripts.preprocess_data
```

2. Split the data into train/validation/test sets:
```bash
python -m learned_retrieval.scripts.split_data
```

3. Train the retrieval model:
```bash
TOKENIZERS_PARALLELISM=true \
CUDA_VISIBLE_DEVICES=0 \
python3 -m learned_retrieval.learned_retrieval.run  \
                --model_name Salesforce/codet5p-220m \
                --model_type bi_encoder \
                --wandb_project_name train-lca \
                --dataset_type levenshtein \
                --device cuda \
                --data_path /home/kolomyttseva/Git/learned-retrieval/data/split \
                --learning_rate 2e-4 \
                --batch_size 32 \
                --num_workers 4 \
                --num_epochs 2 \
                --max_length 512 \
                --accumulation_steps 8 \
                --validation_steps 128 \
                --warmup_steps 512 \
                --normalize_strategy mean_std_sigmoid \
                --loss CrossEntropyLoss
```

The training process:
1. Uses the `Salesforce/codet5p-220m` encoder as the base model
2. Trains it to predict the similarity between completion prefixes and potential context files
3. Evaluates the model on the validation set during training
4. Tracks metrics using Weights & Biases

### Experiments

| Context Retrieval Approach | Label Type | EM Infile, % | EM Inproject, % |
|----------------------------|------------|--------------|-----------------|
| No Context | - | 33.91 | 30.52 |
| Path Distance | - | 34.73 | 31.33 |
| Learned Retrieval | Likelihood | 37.80 | 36.40 |
| **Learned Retrieval** | **Levenshtein** | **41.70** | **37.53** |
| Learned Retrieval | ChrF | 41.10 | 33.81 |
| Brute-Force (Upper Bound) | - | 50.66 | 63.73 |

## Project Structure

- `learned_retrieval/`
  - `collect_generated_data/`: Code for generating completions and calculating the Exact Match upper-bound
  - `collect_logit_data/`: Code for collecting logit data for analysis
  - `learned_retrieval/`: Code for training and evaluating the retrieval model
    - `dataset/`: Dataset classes for the retrieval model
    - `model/`: Model definition and loss functions
    - `train/`: Training loop and utilities
  - `scripts/`: Utility scripts for data preprocessing and splitting
