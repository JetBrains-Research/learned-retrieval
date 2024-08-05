import random
import torch
import os
import numpy as np
import pandas as pd
from pathlib import Path

from transformers.generation import StoppingCriteria
from typing import Dict, List

def exact_match(data: List[Dict[str, str]]):
    df = pd.DataFrame(data)

    grouped_df = df.groupby(["completion_content", "ground_truth", "completion_filename", "completion_line", "completion_line_type"], as_index=False)
    grouped_df = grouped_df.agg(list)

    grouped_df['EM'] = grouped_df['EMs'].apply(lambda x: max(x))
    grouped_by_line_df = grouped_df.groupby("completion_line_type", as_index=False)['EM'].mean()

    em = {"EM_all": grouped_df['EM'].mean()}

    # print(grouped_by_line_df)

    for _, row in grouped_by_line_df.iterrows():
        em[f"EM_{row['completion_line_type']}"] = row['EM']

    return em

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def get_results_path(wb_run, dataset_config):
    base_path = f'/home/kolomyttseva/Git/learned-retrieval/jsonl/{wb_run.id}/generated_data'
    Path(base_path).mkdir(parents=True, exist_ok=True)

    results_path = f'{base_path}/pred_{dataset_config.config_name}_{dataset_config.with_context_files}.jsonl'
    return results_path

class StopOnNewLine(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_ids = set()
        for k, tok_id in tokenizer.vocab.items():
            s = tokenizer.convert_tokens_to_string([k])
            if '\n' in s:
                self.stop_ids.add(tok_id)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert input_ids.shape[0] == 1  # only batch_size 1 is supported
        if input_ids[0, -1].item() in self.stop_ids:
            return True
        else:
            return False

def path_distance(path_from, path_to):
    divided_path_from = os.path.normpath(path_from).split(os.path.sep)
    divided_path_to = os.path.normpath(path_to).split(os.path.sep)
    common_len = 0
    for el1, el2 in zip(divided_path_from, divided_path_to):
        if el1 == el2:
            common_len += 1
        else:
            break
        # return len(divided_path_from) - common_len - 1
    return (len(divided_path_from) - common_len - 1) + (len(divided_path_to) - common_len - 1)

def sort_filepathes(path_from, repo_snapshot):
    # sorts with increase of distances
    max_len = max([len(os.path.normpath(path).split(os.path.sep)) for path in repo_snapshot['filename']])
    max_len += len(os.path.normpath(path_from).split(os.path.sep))
    paths_by_distance = [list() for _ in range(max_len)]

    for path_to, context_content in zip(repo_snapshot['filename'], repo_snapshot['content']):
        dist = path_distance(path_from, path_to)
        paths_by_distance[dist].append((path_to, context_content))
        return [(path_to, context_content) for path_group in paths_by_distance for (path_to, context_content) in path_group]