'''
Usage: 
TOKENIZERS_PARALLELISM=true \
CUDA_VISIBLE_DEVICES=7 \
python3 run.py  --model_name Salesforce/codet5p-220m \
                --device cuda \
                --wandb_project_name train-lca \
                --wandb_run_id kh5ifmom \
                --learning_rate 2e-3 \
                --batch_size 32 \
                --num_workers 4 \
                --num_epochs 20 \
                --max_length 128 \
                --normalize_strategy mean_std_sigmoid
'''

import os
import wandb
from pathlib import Path
import pandas as pd

from fire import Fire

from transformers import AutoTokenizer
import torch
from torch import nn

from learned_retrieval.oracle_logit.dataset.utils import prepare_dataset, prepare_dataloader
from learned_retrieval.oracle_logit.model.model import BiEncoderModel
from learned_retrieval.oracle_logit.model.loss import CrossEntropyLoss
from learned_retrieval.oracle_logit.train.utils import save_checkpoint
from learned_retrieval.oracle_logit.train.train import train, validate, evaluate

def run(model_name: str | Path,
        device: str | torch.DeviceObjType,
        wandb_project_name: str,
        wandb_run_id: str,
        learning_rate = 2e-4,
        batch_size = 32,
        num_workers = 4,
        num_epochs = 5,
        max_length: int = 128,
        normalize_strategy: str = 'mean_std_sigmoid', # ["mean_std", "mean_std_clip", "mean_std_sigmoid", "min_max_clip"]
        ) -> dict:

    base_path = '/home/kolomyttseva/Git/learned-retrieval/jsonl'
    folder_path = f'{base_path}/{wandb_run_id}/logit_data'

    file_name = os.listdir(folder_path)[0]
    path = f'{folder_path}/{file_name}'

    print('>>Load data')
    with open(path) as f:
        data = pd.read_json(f, orient='records', lines=True)

    print('>>Preparing data')
    train_dataset, val_dataset, test_dataset = prepare_dataset(data, normalize_strategy)
    train_loader, val_loader, test_loader = prepare_dataloader(train_dataset, val_dataset, test_dataset, batch_size, num_workers)

    wandb_config = {
        'model': model_name,
        'loss': 'CrossEntropyLoss',
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'max_length': max_length,
        'data': 'logits',
        'normalize_strategy': normalize_strategy,
    }

    print('>>Init model')
    model = nn.DataParallel(BiEncoderModel(model_name)).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = 'left'

    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb_config['learning_rate'])
    criterion = CrossEntropyLoss()

    wandb_run = wandb.init(project=wandb_project_name, name=f"{wandb_config['model']}_{wandb_config['loss']}", config=wandb_config)

    for epoch in range(wandb_config['num_epochs']):
        print(f"Epoch {epoch+1}/{wandb_config['num_epochs']}")
        train_loss = train(model, tokenizer, train_loader, optimizer, criterion, device, max_length)
        val_loss = validate(model, tokenizer, val_loader, criterion, device, max_length)
        print(f'Epoch {epoch+1} completed, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

        checkpoint_filename = f'checkpoint_epoch_{epoch+1}.pth'

    save_checkpoint(model, optimizer, epoch, val_loss, filename=checkpoint_filename)

    em_without_context, em_with_context = evaluate(model, tokenizer, test_dataset, device, max_length)
    print(em_without_context, em_with_context)
    wandb_run.log({
        "em_without_context": em_without_context,
        "em_with_context": em_with_context,
    })

    wandb_run.finish()

if __name__ == '__main__':
    Fire(run)