# ml-4-se-lab/learned-retrieval



## Getting Started

Download links:

SSH clone URL: ssh://git@git.jetbrains.team/ml-4-se-lab/learned-retrieval.git

HTTPS clone URL: https://git.jetbrains.team/ml-4-se-lab/learned-retrieval.git



These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Usage
```
python eval.py --model_name deepseek-ai/deepseek-coder-1.3b-base
               --device cuda
               --with_context_files True
               --config_name small_context
               --max_seq_len 16000
               --wandb_project_name lca-eval
               --limit_samples 10 # to score on a small portion of the entire datset
               --vllm True
```
