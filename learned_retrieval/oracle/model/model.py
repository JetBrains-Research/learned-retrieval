from transformers import AutoModel
from torch import nn
import torch

class BiEncoderModel(nn.Module):
    def __init__(self, model_name):
        super(BiEncoderModel, self).__init__()
        self.completion_encoder = AutoModel.from_pretrained(model_name).encoder
        self.context_encoder = AutoModel.from_pretrained(model_name).encoder

    def forward(self, completion, positive_context, negative_context):
        # completion_embeds = self.completion_encoder(**completion).pooler_output
        # positive_context_embeds = self.context_encoder(**positive_context).pooler_output
        # negative_context_embeds = self.context_encoder(**negative_context).pooler_output

        completion_embeds = self.completion_encoder(**completion).last_hidden_state[:, 0]
        positive_context_embeds = self.context_encoder(**positive_context).last_hidden_state[:, 0]
        negative_context_embeds = self.context_encoder(**negative_context).last_hidden_state[:, 0]

        return completion_embeds, positive_context_embeds, negative_context_embeds
