'''
Usage: 
CUDA_VISIBLE_DEVICES=6 python3 run.py   --model_name deepseek-ai/deepseek-coder-1.3b-base \
                                        --device cuda \
                                        --with_context_files True \
                                        --config_name medium_context \
                                        --max_seq_len 16356 \
                                        --max_completion_len 100 \
                                        --wandb_project_name lca-eval \
                                        --composer path_distance \
                                        --vllm True

--limit-samples is to score on a small portion of the entire datset, scoring
				on the entire dataset can take time
'''

from warnings import warn
from pathlib import Path
import pandas as pd
import json
import wandb
from tqdm.auto import tqdm
from dataclasses import asdict

from fire import Fire

import torch
from transformers.generation import StoppingCriteriaList
from transformers import AutoTokenizer

from learned_retrieval.collect_generated_data.model import get_model, get_tokenizer, generate_completions_batch, generate_completion
from learned_retrieval.collect_generated_data.utils import set_seed, get_results_path, exact_match, StopOnNewLine
from learned_retrieval.collect_generated_data.dataset import LcaPythonCompletionDataset
from learned_retrieval.collect_generated_data.data_classes import ModelConfig, DatasetConfig

def run(model_name: str | Path,
        device: str | torch.DeviceObjType,
        config_name: str | None,
        composer: str,
        wandb_project_name: str,
        max_seq_len: int,
        max_completion_len: int = 128,
        max_context_len: int | None = None,
        with_context_files: bool = False,
        vllm: bool = False,
        seed: int = 42,
        ) -> dict:

    set_seed(seed)

    tokenizer = get_tokenizer(model_name)

    if vllm:
        stopping_criteria = ['\n']
    else:
        stopping_criteria = StoppingCriteriaList([StopOnNewLine(tokenizer)])

    model_config = ModelConfig(model_name,
                               device,
                               model_name,
                               max_seq_len,
                               max_context_len,
                               max_completion_len,
                               seed,
                               stopping_criteria)

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
    model = get_model(model_config, vllm)

    data = []
    # model_inputs = [completion_dataset[n]['model_inputs'] for n in range(num_samples)]

    # predictions = generate_completions_batch(model_inputs,
    #                                          model,
    #                                          tokenizer,
    #                                          model_config,
    #                                          vllm)

    # for i, pred in tqdm(enumerate(predictions)):
    #     s = completion_dataset[i]
    #     s['preds'] = pred
    #     s['EMs'] = int(pred == s['ground_truth'])
    #     data.append(s)

    for n in tqdm(range(num_samples)):
        s = completion_dataset[n]

        pred = generate_completion(s['model_inputs'],
                                   model,
                                   tokenizer,
                                   model_config,
                                   vllm)

        s['preds'] = pred
        s['EMs'] = int(pred == s['ground_truth'])
        data.append(s)

    em = exact_match(data)

    print(em)

    with open(results_path, 'w', encoding='utf-8') as f:
        df = pd.DataFrame(data)
        df.to_json(results_path, orient='records', lines=True)

    wb_run.log(em)
    wb_run.save(results_path)
    wb_run.finish()

if __name__ == '__main__':
    Fire(run)
