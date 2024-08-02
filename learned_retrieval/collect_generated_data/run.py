'''
Usage: 
CUDA_VISIBLE_DEVICES=7 python3 run.py   --model_name deepseek-ai/deepseek-coder-1.3b-base \
                                        --device cuda \
                                        --with_context_files False \
                                        --config_name medium_context \
                                        --max_seq_len 16000 \
                                        --max_completion_len 128 \
                                        --wandb_project_name lca-eval \
                                        --composer brute_force \
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

from learned_retrieval.collect_generated_data.model import get_model, generate_completion
from learned_retrieval.collect_generated_data.utils import set_seed, exact_match, StopOnNewLine
from learned_retrieval.collect_generated_data.dataset import LcaPythonCompletionDataset
from learned_retrieval.collect_generated_data.data_classes import ModelConfig, DatasetConfig

def run(model_name: str | Path,
        device: str | torch.DeviceObjType,
        config_name: str,
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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = 'left'

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
        name='_'.join([model_config.model, dataset_config.config_name]) + f'_{dataset_config.with_context_files}',
        config=asdict(model_config) | asdict(dataset_config)
    )

    base_path = f'/home/kolomyttseva/Git/learned-retrieval/jsonl/{wb_run.id}/generated_data'
    Path(base_path).mkdir(parents=True, exist_ok=True)

    results_path = f'{base_path}/pred_{config_name}_{with_context_files}.jsonl'

    model = get_model(model_config, vllm)

    data = []

    num_samples = len(completion_dataset)
    wb_run.log({"num_samples": num_samples})

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
        # json.dump(data, f, ensure_ascii=False, indent=4)

    wb_run.log(em)
    wb_run.save(results_path)
    wb_run.finish()

if __name__ == '__main__':
    Fire(run)
