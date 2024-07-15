# ml-4-se-lab/learned-retrieval



## Getting Started

Download links:

SSH clone URL: ssh://git@git.jetbrains.team/ml-4-se-lab/learned-retrieval.git

HTTPS clone URL: https://git.jetbrains.team/ml-4-se-lab/learned-retrieval.git



These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Setup
```
pip install -e .
```


## Usage
```
cd eval_lca
CUDA_VISIBLE_DEVICES=6 python3 eval_run.py --model_name deepseek-ai/deepseek-coder-1.3b-base \
                                           --device cuda \
                                           --with_context_files True \
                                           --config_name medium_context \
                                           --max_seq_len 16000 \
                                           --wandb_project_name lca-eval \
                                           --composer path_distance \
                                           --vllm True
```
