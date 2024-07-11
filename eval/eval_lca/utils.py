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