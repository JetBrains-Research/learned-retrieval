import torch
from torch import nn
import numpy as np

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, completion, context, norm_cross_entropy):
        hidden_size = completion.shape[1]

        logits = torch.einsum('bh, bh -> b', completion, context) / np.sqrt(hidden_size)

        loss = self.bce_loss(logits, norm_cross_entropy)

        return loss