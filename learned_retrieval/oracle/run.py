'''
Usage: 
CUDA_VISIBLE_DEVICES=6 python3 run.py   --model_name Salesforce/codet5p-220m \
                                        --device cuda \
                                        --wandb_project_name train-lca \
                                        --wandb_run_id w5v72frx \
                                        --learning_rate 2e-3 \
                                        --batch_size 32 \
                                        --num_workers 4 \
                                        --num_epochs 10 \
                                        --max_length 128
'''

import os
import wandb
from pathlib import Path
import pandas as pd

from fire import Fire

from transformers import AutoTokenizer
import torch

from learned_retrieval.oracle.dataset.utils import prepare_dataset
from learned_retrieval.oracle.model.model import BiEncoderModel
from learned_retrieval.oracle.model.loss import CrossEntropyLoss, MarginContrastiveLoss
from learned_retrieval.oracle.train.utils import save_checkpoint
from learned_retrieval.oracle.train.train import train, validate, evaluate


def run(model_name: str | Path,
        device: str | torch.DeviceObjType,
        wandb_project_name: str,
        wandb_run_id: str,
        learning_rate = 2e-4,
        batch_size = 32,
        num_workers = 4,
        num_epochs = 5,
        max_length: int = 128,
        ) -> dict:

    base_path = '/home/kolomyttseva/Git/learned-retrieval/jsonl'
    folder_path = f'{base_path}/{wandb_run_id}/generated_data'

    file_name = os.listdir(folder_path)[0]
    path = f'{folder_path}/{file_name}'

    print('>>Load data')
    with open(path) as f:
        data = pd.read_json(f, orient='records', lines=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('>>Preparing data')
    train_loader, val_loader, test_loader = prepare_dataset(data, tokenizer, batch_size, num_workers, max_length)
    
    model = BiEncoderModel(model_name).to(device)

    wandb_config = {
        'model': model_name,
        'loss': 'MarginContrastiveLoss',
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'max_length': max_length,
        'data': 'pos_neg_pairs'
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb_config['learning_rate'])
    # criterion = ContrastiveLoss()
    criterion = MarginContrastiveLoss()
    # criterion = CrossEntropyLoss()

    wandb.init(project=wandb_project_name, name=f"{wandb_config['model']}_{wandb_config['loss']}", config=wandb_config)

    for epoch in range(wandb_config['num_epochs']):
        print(f"Epoch {epoch+1}/{wandb_config['num_epochs']}")
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1} completed, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

        checkpoint_filename = f'checkpoint_epoch_{epoch+1}.pth'

    save_checkpoint(model, optimizer, epoch, val_loss, filename=checkpoint_filename)

    accuracy = evaluate(model, test_loader, device)
    print(accuracy)
    wandb.log({'accuracy': accuracy})

    wandb.finish()

if __name__ == '__main__':
    Fire(run)