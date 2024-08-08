import torch
import wandb
import numpy as np
from tqdm.auto import tqdm

from learned_retrieval.oracle.train.data_classes import Config
from learned_retrieval.oracle.train.utils import save_checkpoint, calculate_loss
from learned_retrieval.oracle.dataset.data_classes import DatasetsClass, DataLoadersClass

def train(model, tokenizer, optimizer, criterion, dataloaders: DataLoadersClass, datasets: DatasetsClass, config: Config):
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
            em_without_context, em_with_context = evaluate(model, tokenizer, datasets.val, config)
            print(f"Validation Loss at step {batch_idx + 1}: {val_loss:.4f}")
            print(f"Evaluation: EM Without Context: {em_without_context:.4f}, EM With Context: {em_with_context:.4f}")
            wandb.log({
                'val_loss': val_loss,
                'em_without_context': em_without_context,
                'em_with_context': em_with_context,
                'step': batch_idx + 1
            })

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

def evaluate(model, tokenizer, dataset, config: Config):
    model.eval()  # Set the model to evaluation mode
    
    groped_data = dataset.data.groupby(["completion"], as_index=False)
    groped_data = groped_data.agg(list)

    em_without_context = 0
    em_with_context = 0

    with torch.no_grad():
        for _, item in tqdm(groped_data.iterrows(), total=len(groped_data), desc='Evaluate'):
            completion = item['completion']
            contexts = item['context']
            em = item['EM']

            preds = []

            for context in contexts:
                completion_encoding = tokenizer(completion, return_tensors='pt', max_length=config.max_length, padding='max_length', truncation=True)
                completion_encoding = {k: v.to(config.device) for k, v in completion_encoding.items()}

                context_encoding = tokenizer(context, return_tensors='pt', max_length=config.max_length, padding='max_length', truncation=True)
                context_encoding = {k: v.to(config.device) for k, v in context_encoding.items()}

                completion_embeds, context_embeds = model(completion_encoding, context_encoding)
                context_embeds = context_embeds[0]

                hidden_size = completion_embeds.shape[1]
                logit = torch.einsum('bh, bh -> b', completion_embeds, context_embeds) / np.sqrt(hidden_size)
                pred = torch.sigmoid(logit)

                preds.append(pred.item())

            preds = np.asarray(preds)
            max_pred_idx = np.argmax(preds)

            em_without_context += em[0]
            em_with_context += em[max_pred_idx]

    em_without_context /= len(groped_data)
    em_with_context /= len(groped_data)

    return em_without_context, em_with_context

def train_loop(wandb_project_name, model, tokenizer, optimizer, criterion, dataloaders: DataLoadersClass, datasets: DatasetsClass, config: Config):
    wandb_run = wandb.init(project=wandb_project_name, name=f"{config.model_name}_{config.loss}", config=config)
    
    # Main training loop
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        train_loss = train(model, tokenizer, optimizer, criterion, dataloaders, datasets, config)
        val_loss = validate(model, tokenizer, criterion, dataloaders.val, config)
        em_without_context, em_with_context = evaluate(model, tokenizer, datasets.val, config)
        print(f'Epoch {epoch+1} completed, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        print(f'Evaluation: EM Without Context: {em_without_context:.4f}, EM With Context: {em_with_context:.4f}')

        wandb_run.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'em_without_context': em_without_context,
            'em_with_context': em_with_context
        })

        checkpoint_filename = f'checkpoint_epoch_{epoch+1}.pth'

    save_checkpoint(model, optimizer, epoch, val_loss, filename=checkpoint_filename)

    em_without_context, em_with_context = evaluate(model, tokenizer, datasets.test, config)
    print(f'Evaluation: EM Without Context: {em_without_context:.4f}, EM With Context: {em_with_context:.4f}')
    
    wandb_run.log({
        "em_without_context": em_without_context,
        "em_with_context": em_with_context,
    })

    wandb_run.finish()