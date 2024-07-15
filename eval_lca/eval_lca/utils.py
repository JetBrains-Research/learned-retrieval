import random
import torch
import os
import numpy as np

from transformers.generation import StoppingCriteria

def exact_match(gts, preds, repo_snapshot_lens):
    start = 0
    em = 0

    for l in repo_snapshot_lens:
        em_curr = sum(gt == pr for gt, pr in zip(gts[start:start+l], preds[start:start+l]))
        em += (em_curr > 0)
        start += l
    
    em /= len(repo_snapshot_lens)

    return em

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

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