import torch
import wandb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from learned_retrieval.oracle.train.data_classes import Config
from learned_retrieval.oracle.train.utils import save_checkpoint, calculate_loss
from learned_retrieval.oracle.dataset.data_classes import DatasetsClass, DataLoadersClass

def train(wandb_run, model, tokenizer, optimizer, criterion, dataloaders: DataLoadersClass, datasets: DatasetsClass, config: Config):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(dataloaders.train, desc='Train')):
        loss = calculate_loss(batch, model, tokenizer, criterion, config)
        loss = loss / config.accumulation_steps

        loss.backward()

        if (batch_idx + 1) % config.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * config.accumulation_steps

        # Validation step
        if config.validation_steps and (batch_idx + 1) % config.validation_steps == 0:
            val_loss = validate(model, tokenizer, criterion, dataloaders.val, config)
            em = evaluate(model, tokenizer, datasets.val, config)
            print(f"Validation Loss at step {batch_idx + 1}: {val_loss:.4f}")
            print(f"Evaluation: EM Without Context: {em['em_without_context']:.4f}, EM With Context: {em['em_with_context']:.4f}")
            wandb_run.log({
                'val_loss': val_loss,
                'step': batch_idx + 1
            })
            wandb_run.log(em)

            checkpoint_filename = f'checkpoint.pth'
            save_checkpoint(model, optimizer, batch_idx + 1, val_loss, filename=checkpoint_filename)

        # # Test step
        # if config.validation_steps and (batch_idx + 1) % (config.validation_steps * 8) == 0:
        #     em = evaluate(model, tokenizer, datasets.test, config, 'test')
        #     print(f"Evaluation: EM Without Context: {em['test_em_without_context']:.4f}, EM With Context: {em['test_em_with_context']:.4f}")
        #     wandb_run.log({
        #         'step': batch_idx + 1
        #     })
        #     wandb_run.log(em)

    if (batch_idx + 1) % config.accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    average_loss = total_loss / len(dataloaders.train)
    return average_loss

def validate(model, tokenizer, criterion, dataloader, config: Config):
    model.eval()

    total_val_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Validate')):
            loss = calculate_loss(batch, model, tokenizer, criterion, config)

            total_val_loss += loss.item()

    average_val_loss = total_val_loss / len(dataloader)
    return average_val_loss

def evaluate(model, tokenizer, dataset, config: Config, split: str | None = None):
    model.eval()  # Set the model to evaluation mode
    
    grouped_data = dataset.data.groupby(["completion", "completion_line_type"], as_index=False)
    grouped_data = grouped_data.agg(list)

    results = []
    
    with torch.no_grad():
        for _, item in tqdm(grouped_data.iterrows(), total=len(grouped_data), desc='Evaluate'):
            completion = item['completion']
            contexts = item['context']
            em = item['EM']

            preds = []

            for context in contexts:
                # Tokenize completion and context
                completion_encoding = tokenizer(completion, return_tensors='pt', max_length=config.max_length, padding='max_length', truncation=True)
                completion_encoding = {k: v.to(config.device) for k, v in completion_encoding.items()}

                context_encoding = tokenizer(context, return_tensors='pt', max_length=config.max_length, padding='max_length', truncation=True)
                context_encoding = {k: v.to(config.device) for k, v in context_encoding.items()}

                # Forward pass through model
                completion_embeds, context_embeds = model(completion_encoding, context_encoding)
                context_embeds = context_embeds[0]

                # Calculate similarity (logits) and apply sigmoid
                hidden_size = completion_embeds.shape[1]
                logit = torch.einsum('bh, bh -> b', completion_embeds, context_embeds) / np.sqrt(hidden_size)
                pred = torch.sigmoid(logit)

                preds.append(pred.item())

            preds = np.asarray(preds)
            max_pred_idx = np.argmax(preds)

            result = {
                'completion': completion,
                'completion_line_type': item['completion_line_type'],
                'em_without_context': em[0],
                'em_with_context': em[max_pred_idx]
            }
            results.append(result)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate the overall mean EM scores
    em = {
        "em_without_context": results_df['em_without_context'].mean(),
        "em_with_context": results_df['em_with_context'].mean()
    }

    # Calculate the mean EM scores grouped by 'completion_line_type'
    grouped_by_line_data = results_df.groupby(["completion_line_type"], as_index=False)[['em_without_context', 'em_with_context']].mean()

    for _, row in grouped_by_line_data.iterrows():
        em[f"em_without_context_{row['completion_line_type']}"] = row['em_without_context']
        em[f"em_with_context_{row['completion_line_type']}"] = row['em_with_context']

    if split is not None:
        em = {f'{split}_{k}': v for k, v in em.items()}

    print(em)
    return em

def train_loop(wandb_project_name, model, tokenizer, optimizer, criterion, dataloaders: DataLoadersClass, datasets: DatasetsClass, config: Config):
    wandb_run = wandb.init(project=wandb_project_name, name=f"{config.model_name}_{config.loss}", config=config)
    
    # Main training loop
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        train_loss = train(wandb_run, model, tokenizer, optimizer, criterion, dataloaders, datasets, config)
        val_loss = validate(model, tokenizer, criterion, dataloaders.val, config)
        em = evaluate(model, tokenizer, datasets.val, config)
        print(f'Epoch {epoch+1} completed, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        print(f"Evaluation: EM Without Context: {em['em_without_context']:.4f}, EM With Context: {em['em_with_context']:.4f}")

        wandb_run.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
        })
        
        wandb_run.log(em)
        checkpoint_filename = f'checkpoint_epoch_{epoch+1}.pth'

    save_checkpoint(model, optimizer, epoch, val_loss, filename=checkpoint_filename)

    em = evaluate(model, tokenizer, datasets.test, config, 'test')
    print(f"Evaluation: EM Without Context: {em['test_em_without_context']:.4f}, EM With Context: {em['test_em_with_context']:.4f}")
    
    wandb_run.log(em)

    wandb_run.finish()