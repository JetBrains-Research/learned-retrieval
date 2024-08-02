import torch
from torch import nn
import numpy as np

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, completion, positive_context, negative_context):
        hidden_size = completion.shape[1]

        positive_logit = torch.einsum('bh, bh -> b', completion, positive_context) / np.sqrt(hidden_size)
        negative_logit = torch.einsum('bh, bh -> b', completion, negative_context) / np.sqrt(hidden_size)

        logits = torch.cat((positive_logit, negative_logit), 0)
        labels = torch.cat((torch.ones_like(positive_logit), torch.zeros_like(negative_logit)), 0)

        loss = self.bce_loss(logits, labels)

        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.similarity = nn.CosineSimilarity(dim=-1, eps=1e-7)
        self.mse_loss = nn.MSELoss()

    def forward(self, completion, positive_context, negative_context):
        positive_score = self.similarity(completion, positive_context)
        negative_score = self.similarity(completion, negative_context)

        score_difference = positive_score - negative_score

        target = torch.ones_like(score_difference)
        loss = self.mse_loss(score_difference, target)

        return loss

class MarginContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(MarginContrastiveLoss, self).__init__()
        self.similarity = nn.CosineSimilarity(dim=-1, eps=1e-7)
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, completion, positive_context, negative_context):
        positive_score = torch.abs(self.similarity(completion, positive_context))
        negative_score = torch.abs(self.similarity(completion, negative_context))

        target = torch.ones_like(positive_score)
        loss = self.ranking_loss(positive_score, negative_score, target)

        return loss
