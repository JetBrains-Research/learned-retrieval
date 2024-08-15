import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
import numpy as np

class BaseMSELoss(nn.Module):
    def __init__(self):
        super(BaseMSELoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the forward method")
    
    @staticmethod
    def create_instance(dataset_type: str):
        if dataset_type == 'pos_neg_pairs':
            return PairsMSELoss()
        else:
            return MSELoss()

class MSELoss(BaseMSELoss):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, completion, context, label):
        hidden_size = completion.shape[1]

        logits = torch.einsum('bh, bh -> b', completion, context) / np.sqrt(hidden_size)

        loss = self.mse_loss(logits, label)

        return loss

class PairsMSELoss(BaseMSELoss):
    def __init__(self):
        super(PairsMSELoss, self).__init__()

    def forward(self, completion, positive_context, negative_context):
        hidden_size = completion.shape[1]

        positive_logit = torch.einsum('bh, bh -> b', completion, positive_context) / np.sqrt(hidden_size)
        negative_logit = torch.einsum('bh, bh -> b', completion, negative_context) / np.sqrt(hidden_size)

        logits = torch.cat((positive_logit, negative_logit), 0)
        labels = torch.cat((torch.ones_like(positive_logit), torch.zeros_like(negative_logit)), 0)

        loss = self.mse_loss(logits, labels)

        return loss    

class BaseCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BaseCrossEntropyLoss, self).__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the forward method")
    
    @staticmethod
    def create_instance(dataset_type: str):
        if dataset_type == 'pos_neg_pairs':
            return PairsCrossEntropyLoss()
        else:
            return CrossEntropyLoss()
        
class CrossEntropyLoss(BaseCrossEntropyLoss):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, completion, context, label):
        hidden_size = completion.shape[1]

        logits = torch.einsum('bh, bh -> b', completion, context) / np.sqrt(hidden_size)

        loss = self.bce_loss(logits, label)

        return loss

class PairsCrossEntropyLoss(BaseCrossEntropyLoss):
    def __init__(self):
        super(PairsCrossEntropyLoss, self).__init__()

    def forward(self, completion, positive_context, negative_context):
        hidden_size = completion.shape[1]

        positive_logit = torch.einsum('bh, bh -> b', completion, positive_context) / np.sqrt(hidden_size)
        negative_logit = torch.einsum('bh, bh -> b', completion, negative_context) / np.sqrt(hidden_size)

        logits = torch.cat((positive_logit, negative_logit), 0)
        labels = torch.cat((torch.ones_like(positive_logit), torch.zeros_like(negative_logit)), 0)

        loss = self.bce_loss(logits, labels)

        return loss