'''
Usage: 
CUDA_VISIBLE_DEVICES=6 python3 eval_lca.py --model_name deepseek-ai/deepseek-coder-1.3b-base \
                                           --device cuda \
                                           --with_context_files True \
                                           --config_name medium_context \
                                           --max_seq_len 16000 \
                                           --wandb_project_name lca-eval \
                                           --composer path_distance \
                                           --vllm True

--limit-samples is to score on a small portion of the entire datset, scoring
				on the entire dataset can take time
'''

from warnings import warn
from pathlib import Path
import json
import wandb

from fire import Fire
import torch
from transformers.generation import StoppingCriteria, StoppingCriteriaList
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm.auto import tqdm
from dataclasses import asdict

from eval_lca.model import get_model, generate_completion
from eval_lca.utils import set_seed, exact_match, StopOnNewLine
from eval_lca.dataset import LcaPythonCompletionDataset
from eval_lca.data_classes import ModelConfig, DatasetConfig

def eval_lca(model_name: str | Path,
             device: str | torch.DeviceObjType,
             config_name: str,
             composer: str,
             wandb_project_name: str,
             max_seq_len: int,
             max_completion_len: int = 128,
             max_context_len: int | None = None,
             with_context_files: bool = False,
             limit_samples: int | None = None,
             vllm: bool = False,
             seed: int = 42,
             ) -> dict:

    if limit_samples is not None:
        warn(f'Limiting the number of samples to evaluate on to {limit_samples}!')

    set_seed(seed)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
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
    
    ds_test = LcaPythonCompletionDataset(dataset_config)

    base_path = '/home/kolomyttseva/Git/learned-retrieval/eval/generated_data'
    results_path = f'{base_path}/pred_{config_name}_{with_context_files}_{limit_samples}.jsonl'

    wb_run = wandb.init(
        project=wandb_project_name,
        name='_'.join([model_config.model, dataset_config.config_name]) + f'_{dataset_config.with_context_files}',
        config=asdict(model_config) | asdict(dataset_config)
    )

    model = get_model(model_config, vllm)

    gts = []
    preds = []
    data = []
    
    num_samples = ds_test.get_limited_len(limit_samples) if limit_samples is not None else len(ds_test)
    wb_run.log({"num_samples": num_samples})

    for n in tqdm(range(num_samples)):
        s = ds_test[n]
        
        pred = generate_completion(s['model_input'],
                                   model,
                                   tokenizer,
                                   model_config,
                                   vllm)
        
        s['pred'] = pred
        data.append(s)

        preds.append(pred)
        gts.append(s['sample']['gt'])
    
    em = exact_match(gts, preds, ds_test.get_repo_snapshot_lens(limit_samples))

    results = {
        'EM': em
    }

    print(results)

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    wb_run.log(results)
    wb_run.save(results_path)
    wb_run.finish()
    


if __name__ == '__main__':
    Fire(eval_lca)
