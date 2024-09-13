'''
--normalize_strategy mean_std_sigmoid \

Usage: 
TOKENIZERS_PARALLELISM=true \
CUDA_VISIBLE_DEVICES=3 \
python3 run.py  --model_name Salesforce/codet5p-220m \
                --model_type bi_encoder \
                --wandb_project_name train-lca \
                --dataset_type pos_neg_pairs \
                --device cuda \
                --data_path /home/kolomyttseva/Git/learned-retrieval/data/split \
                --learning_rate 2e-4 \
                --batch_size 32 \
                --num_workers 4 \
                --num_epochs 2 \
                --max_length 512 \
                --accumulation_steps 4 \
                --validation_steps 128 \
                --warmup_steps 512 \
                --loss CrossEntropyLoss
'''


from pathlib import Path
from fire import Fire
import torch
from transformers import get_cosine_schedule_with_warmup

from learned_retrieval.oracle.dataset.utils import prepare_dataset, prepare_dataloader
from learned_retrieval.oracle.model.utils import get_model, get_tokenizer
from learned_retrieval.oracle.model.loss import BaseLoss
from learned_retrieval.oracle.train.data_classes import Config
from learned_retrieval.oracle.train.train import train_loop

def run(model_name: str | Path,
        device: str,
        data_path: str | Path,
        wandb_project_name: str,
        learning_rate: float = 2e-4,
        batch_size: int = 32,
        num_workers: int = 4,
        num_epochs: int = 5,
        max_length: int = 128,
        normalize_strategy: str | None = None,  # ["mean_std", "mean_std_clip", "mean_std_sigmoid", "min_max_clip"]
        accumulation_steps: int = 1,
        validation_steps: int = 128,
        warmup_steps: int = 500,
        limit_samples: int | None = None,
        loss: str = 'CrossEntropyLoss',
        dataset_type: str = 'em',  # ["em0to1", "em_per_file", "pos_neg_pairs", "logit"]
        model_type: str = 'bi_encoder'  # ["bi_encoder", "cross_encoder"]
        ) -> dict:

    data_path = Path(data_path)

    config = Config(
        model_name=model_name,
        model_type=model_type,
        loss=loss,
        device=device,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        accumulation_steps=accumulation_steps,
        validation_steps=validation_steps,
        warmup_steps=warmup_steps,
        dataset_type=dataset_type,
        normalize_strategy=normalize_strategy,
        limit_samples=limit_samples
    )

    data_split = {
        'train': data_path / 'train_split.jsonl',
        'val': data_path / 'val_split.jsonl',
        'test': data_path / 'test_split.jsonl'
    }

    datasets = prepare_dataset(data_split, dataset_type, normalize_strategy=normalize_strategy, limit_samples=limit_samples)
    dataloaders = prepare_dataloader(datasets, batch_size, num_workers)
    
    model = get_model(config)
    tokenizer = get_tokenizer(model_name)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = BaseLoss.create_instance(config.loss, config.dataset_type)

    # Calculate total training steps
    steps_per_epoch = len(dataloaders.train) // config.accumulation_steps
    total_steps = steps_per_epoch * config.num_epochs
    warmup_steps = config.warmup_steps

    # Initialize the scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    train_loop(wandb_project_name, model, tokenizer, optimizer, scheduler, criterion, dataloaders, datasets, config)

if __name__ == '__main__':
    Fire(run)