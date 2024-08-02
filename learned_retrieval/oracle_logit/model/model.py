from transformers import AutoModel
from torch import nn

class BiEncoderModel(nn.Module):
    def __init__(self, model_name):
        super(BiEncoderModel, self).__init__()
        self.completion_encoder = AutoModel.from_pretrained(model_name).encoder
        self.context_encoder = AutoModel.from_pretrained(model_name).encoder

    def forward(self, completion, context):
        completion_embeds = self.completion_encoder(**completion).last_hidden_state[:, 0]
        context_embeds = self.context_encoder(**context).last_hidden_state[:, 0]

        return completion_embeds, context_embeds