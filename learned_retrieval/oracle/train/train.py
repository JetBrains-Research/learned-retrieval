import torch
import wandb
import numpy as np
from tqdm.auto import tqdm

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        completion, positive_context, negative_context = batch
        completion = {k: v.squeeze().to(device) for k, v in completion.items()}
        positive_context = {k: v.squeeze().to(device) for k, v in positive_context.items()}
        negative_context = {k: v.squeeze().to(device) for k, v in negative_context.items()}

        optimizer.zero_grad()
        completion_embeds, positive_context_embeds, negative_context_embeds = model(completion, positive_context, negative_context)
        loss = criterion(completion_embeds, positive_context_embeds, negative_context_embeds)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            completion, positive_context, negative_context = batch
            completion = {k: v.squeeze().to(device) for k, v in completion.items()}
            positive_context = {k: v.squeeze().to(device) for k, v in positive_context.items()}
            negative_context = {k: v.squeeze().to(device) for k, v in negative_context.items()}

            completion_embeds, positive_context_embeds, negative_context_embeds = model(completion, positive_context, negative_context)
            loss = criterion(completion_embeds, positive_context_embeds, negative_context_embeds)

            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss

def evaluate(model, dataloader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            completion, positive_context, negative_context = batch

            completion = {k: v.squeeze().to(device) for k, v in completion.items()}
            positive_context = {k: v.squeeze().to(device) for k, v in positive_context.items()}
            negative_context = {k: v.squeeze().to(device) for k, v in negative_context.items()}

            completion_embeds, positive_context_embeds, negative_context_embeds = model(completion, positive_context, negative_context)

            hidden_size = completion_embeds.shape[1]
            positive_logit = torch.einsum('bh, bh -> b', completion_embeds, positive_context_embeds) / np.sqrt(hidden_size)
            negative_logit = torch.einsum('bh, bh -> b', completion_embeds, negative_context_embeds) / np.sqrt(hidden_size)

            logits = torch.cat((positive_logit, negative_logit))
            predictions = torch.round(torch.sigmoid(logits))

            labels = torch.cat((torch.ones_like(positive_logit), torch.zeros_like(negative_logit)))

            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    return accuracy