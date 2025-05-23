from learned_retrieval.learned_retrieval.model.model import BaseModel
from transformers import AutoTokenizer
from learned_retrieval.learned_retrieval.train.data_classes import Config

def get_model(config: Config):
    print('>>Init model')
    model = BaseModel.create_instance(config.model_name, config.model_type).to(config.device)
    return model

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = 'left'
    return tokenizer