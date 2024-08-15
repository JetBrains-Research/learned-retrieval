import wandb
import torch
from learned_retrieval.oracle.train.data_classes import Config

def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filename)
    wandb.save(filename)

def calculate_loss(batch, model, tokenizer, criterion, config: Config):
    if config.dataset_type == 'pos_neg_pairs':
        loss = process_pairs_batch(batch, model, tokenizer, criterion, config)
    else:
        loss = process_batch(batch, model, tokenizer, criterion, config)
    return loss

def process_batch(batch, model, tokenizer, criterion, config: Config):
    completion, context, label = batch

    completion_encoding = tokenizer(completion, return_tensors='pt', max_length=config.max_length, padding='max_length', truncation=True)
    completion_encoding = {k: v.to(config.device) for k, v in completion_encoding.items()}

    context_encoding = tokenizer(context, return_tensors='pt', max_length=config.max_length, padding='max_length', truncation=True)
    context_encoding = {k: v.to(config.device) for k, v in context_encoding.items()}

    label = label.to(config.device)

    completion_embeds, context_embeds = model(completion_encoding, context_encoding)
    loss = criterion(completion_embeds, context_embeds[0], label)
    
    return loss

def process_pairs_batch(batch, model, tokenizer, criterion, config: Config):
    completion, positive_context, negative_context = batch

    completion_encoding = tokenizer(completion, return_tensors='pt', max_length=config.max_length, padding='max_length', truncation=True)
    completion_encoding = {k: v.to(config.device) for k, v in completion_encoding.items()}

    positive_context_encoding = tokenizer(positive_context, return_tensors='pt', max_length=config.max_length, padding='max_length', truncation=True)
    positive_context_encoding = {k: v.to(config.device) for k, v in positive_context_encoding.items()}

    negative_context_encoding = tokenizer(negative_context, return_tensors='pt', max_length=config.max_length, padding='max_length', truncation=True)
    negative_context_encoding = {k: v.to(config.device) for k, v in negative_context_encoding.items()}

    completion_embeds, context_embeds = model(completion_encoding, positive_context_encoding, negative_context_encoding)
    loss = criterion(completion_embeds, context_embeds[0], context_embeds[1])
    
    return loss