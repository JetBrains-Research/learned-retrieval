'''
Usage: 
CUDA_VISIBLE_DEVICES=6 python3 run.py   --model_name deepseek-ai/deepseek-coder-1.3b-base \
                                        --device cuda \
                                        --with_context_files True \
                                        --config_name large_context \
                                        --max_seq_len 16000 \
                                        --wandb_project_name lca-collect-logit-data \
                                        --composer brute_force
'''

from warnings import warn
from pathlib import Path
import pandas as pd
import wandb
from tqdm.auto import tqdm
from dataclasses import asdict

from fire import Fire

import torch

from learned_retrieval.collect_logit_data.model import get_model, get_tokenizer, get_ground_truth_logits
from learned_retrieval.collect_logit_data.utils import set_seed, get_results_path
from learned_retrieval.collect_logit_data.dataset import LcaPythonCompletionDataset
from learned_retrieval.collect_logit_data.data_classes import ModelConfig, DatasetConfig

def run(model_name: str | Path,
        device: str | torch.DeviceObjType,
        config_name: str,
        composer: str,
        wandb_project_name: str,
        max_seq_len: int,
        with_context_files: bool = False,
        seed: int = 42,
        ) -> dict:

    set_seed(seed)

    model_config = ModelConfig(model_name,
                               device,
                               model_name,
                               max_seq_len,
                               seed=seed)

    dataset_config = DatasetConfig(config_name,
                                   with_context_files,
                                   composer=composer)

    completion_dataset = LcaPythonCompletionDataset(dataset_config)

    wb_run = wandb.init(
        project=wandb_project_name,
        name=f'_{model_config.model}_{dataset_config.config_name}_{dataset_config.with_context_files}',
        config=asdict(model_config) | asdict(dataset_config)
    )
    num_samples = len(completion_dataset)
    wb_run.log({"num_samples": num_samples})

    results_path = get_results_path(wb_run, dataset_config)
    model = get_model(model_config)
    tokenizer = get_tokenizer(model_config)

    data = []

    for n in tqdm(range(num_samples)):
        s = completion_dataset[n]

        cross_entropy, avg_cross_entropy, perplexity = get_ground_truth_logits(s['model_inputs'],
                                                                               s['ground_truth'],
                                                                               model,
                                                                               tokenizer)

        s['cross_entropy'] = cross_entropy
        s['avg_cross_entropy'] = avg_cross_entropy
        s['perplexity'] = perplexity

        data.append(s)

    df = pd.DataFrame(data)
    df.to_json(results_path, orient='records', lines=True)

    wb_run.save(results_path)
    wb_run.finish()

if __name__ == '__main__':
    Fire(run)
