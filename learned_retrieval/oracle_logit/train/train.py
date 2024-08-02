import torch
import numpy as np
from tqdm.auto import tqdm

def train(model, tokenizer, dataloader, optimizer, criterion, device, max_length=128):
    model.train()
    total_loss = 0    
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        completion, context, norm_cross_entropy = batch

        completion_encoding = tokenizer(completion, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
        completion_encoding = {k: v.to(device) for k, v in completion_encoding.items()}

        context_encoding = tokenizer(context, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
        context_encoding = {k: v.to(device) for k, v in context_encoding.items()}

        norm_cross_entropy = norm_cross_entropy.to(device)

        optimizer.zero_grad()
        completion_embeds, context_embeds = model(completion_encoding, context_encoding)
        loss = criterion(completion_embeds, context_embeds, norm_cross_entropy)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    average_loss = total_loss / len(dataloader)
    return average_loss

def validate(model, tokenizer, dataloader, criterion, device, max_length=128):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            completion, context, norm_cross_entropy = batch

            completion_encoding = tokenizer(completion, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
            completion_encoding = {k: v.to(device) for k, v in completion_encoding.items()}

            context_encoding = tokenizer(context, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
            context_encoding = {k: v.to(device) for k, v in context_encoding.items()}
            
            norm_cross_entropy = norm_cross_entropy.to(device)

            completion_embeds, context_embeds = model(completion_encoding, context_encoding)
            loss = criterion(completion_embeds, context_embeds, norm_cross_entropy)
            total_loss += loss.item()
    
    average_loss = total_loss / len(dataloader)
    return average_loss

def evaluate(model, tokenizer, dataset, device, max_length=128):
    model.eval()  # Set the model to evaluation mode
    
    groped_data = dataset.data.groupby(["completion_content"], as_index=False)
    groped_data = groped_data.agg(list)

    em_without_context = 0
    em_with_context = 0

    with torch.no_grad():
        for _, item in tqdm(groped_data.iterrows(), total=len(groped_data)):
            completion = item['completion_content']
            contexts = item['context_files']
            em = item['EMs']

            preds = []

            for context in contexts:
                completion_encoding = tokenizer(completion, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
                completion_encoding = {k: v.to(device) for k, v in completion_encoding.items()}

                context_encoding = tokenizer(context, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
                context_encoding = {k: v.to(device) for k, v in context_encoding.items()}

                completion_embeds, context_embeds = model(completion_encoding, context_encoding)
            
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