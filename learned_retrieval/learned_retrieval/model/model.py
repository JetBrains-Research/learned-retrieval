from transformers import AutoModel
from torch import nn
import torch

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the forward method.")

    @staticmethod
    def create_instance(model_name: str, model_type: str):
        if model_type == 'bi_encoder':
            return BiEncoderModel(model_name)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class BiEncoderModel(BaseModel):
    def __init__(self, model_name):
        super(BiEncoderModel, self).__init__()
        self.completion_encoder = AutoModel.from_pretrained(model_name).encoder
        self.context_encoder = AutoModel.from_pretrained(model_name).encoder

    def forward(self, completion, *contexts):
        completion_embeds = self.completion_encoder(**completion).last_hidden_state[:, 0]

        context_embeds_list = []
        for context in contexts:
            context_embeds = self.context_encoder(**context).last_hidden_state[:, 0]
            context_embeds_list.append(context_embeds)
        
        return completion_embeds, context_embeds_list