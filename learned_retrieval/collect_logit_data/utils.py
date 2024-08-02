from pathlib import Path
import random
import torch
import os
import numpy as np

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def get_results_path(wb_run, dataset_config):
    base_path = f'/home/kolomyttseva/Git/learned-retrieval/jsonl/{wb_run.id}/logit_data'
    Path(base_path).mkdir(parents=True, exist_ok=True)

    results_path = f'{base_path}/logit_{dataset_config.config_name}_{dataset_config.with_context_files}.jsonl'
    return results_path

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