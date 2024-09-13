'''
medium_context
7ontbo2s -- em
c4l60bhb -- logit

large_context
yqmklyyb -- em
wyez0ger -- logit

Usage: 
python3 preprocess_data.py --wandb_project_name_logit lca-collect-logit-data \
                           --wandb_run_id_logit c4l60bhb \
                           --wandb_project_name_em lca-eval \
                           --wandb_run_id_em 7ontbo2s \
                           --base_path /home/kolomyttseva/Git/learned-retrieval/jsonl \
                           --save_file /home/kolomyttseva/Git/learned-retrieval/data/raw/medium_context_data.jsonl

python3 preprocess_data.py --wandb_project_name_logit lca-collect-logit-data \
                           --wandb_run_id_logit wyez0ger \
                           --wandb_project_name_em lca-eval \
                           --wandb_run_id_em yqmklyyb \
                           --base_path /home/kolomyttseva/Git/learned-retrieval/jsonl \
                           --save_file /home/kolomyttseva/Git/learned-retrieval/data/raw/large_context_data.jsonl
'''

from pathlib import Path
from fire import Fire
import pandas as pd
import wandb
import os
import evaluate

class WandbFileManager:
    def __init__(self, project_name: str, run_id: str, base_path: Path, dataset_type: str):
        self.project_name = project_name
        self.run_id = run_id
        self.base_path = Path(base_path)
        self.dataset_type = dataset_type
        self.folder_name = 'generated_data' if self.dataset_type == 'em' else 'logit_data'

        self.path = self.base_path / self.run_id

    def load_file(self):
        api = wandb.Api()
        run = api.run(f"{self.project_name}/{self.run_id}")
        files = run.files()

        for file in files:
            if self.folder_name in file.name:
                file_name = file.name
                break

        run.file(file_name).download(root=str(self.path), replace=True)

        print(f"Downloaded to {self.path / file_name}")
        return self.path / file_name

    def get_file(self):
        file_path = next(self.path.joinpath(self.folder_name).iterdir())
        return file_path

    def __call__(self):
        if self.path.joinpath(self.folder_name).is_dir():
            path = self.get_file()
        else:
            path = self.load_file()
        
        return path

def load_data(path: Path):
    print('>> Load data')
    with path.open() as f:
        data = pd.read_json(f, orient='records', lines=True)
    return data

def save_data_from_wandb(wandb_project_name: str,
                         wandb_run_id: str,
                         dataset_type: str, # ["em", "logit"]
                         base_path: Path):
    
    file_manager = WandbFileManager(wandb_project_name, wandb_run_id, base_path, dataset_type)
    file_path = file_manager()
    return file_path

def run(wandb_project_name_logit: str,
        wandb_run_id_logit: str,
        wandb_project_name_em: str,
        wandb_run_id_em: str,
        base_path: str,
        save_file: str):

    base_path = Path(base_path)
    save_file = Path(save_file)

    logit_path = save_data_from_wandb(wandb_project_name_logit, wandb_run_id_logit, 'logit', base_path)
    em_path = save_data_from_wandb(wandb_project_name_em, wandb_run_id_em, 'em', base_path)

    logit_data = load_data(logit_path)
    em_data = load_data(em_path)

    logit_data['EMs'] = em_data['EMs']
        
    levenshtein_metric = evaluate.load("levenshtein")
    chrf_metric = evaluate.load("chrf")

    logit_data['levenshtein'] = em_data.apply(lambda x: levenshtein_metric.compute(predictions=[x['preds']], references=[x['ground_truth']])['score'], axis=1)
    logit_data['chrf'] = em_data.apply(lambda x: chrf_metric.compute(predictions=[x['preds']], references=[x['ground_truth']])['score'], axis=1)

    logit_data['context_content'] = logit_data['context_files'].apply(lambda x: x[0]['content'])
    logit_data['context_filename'] = logit_data['context_files'].apply(lambda x: x[0]['filename'])

    logit_data = logit_data.drop(columns=['cross_entropy', 'context_files']).drop_duplicates()

    logit_data.to_json(save_file, orient='records', lines=True)

if __name__ == '__main__':
    Fire(run)