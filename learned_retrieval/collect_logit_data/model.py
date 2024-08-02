from vllm import LLM, SamplingParams
from learned_retrieval.collect_logit_data.data_classes import ModelConfig

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_model(model_config: ModelConfig):
    model = AutoModelForCausalLM.from_pretrained(model_config.model,
                                                 torch_dtype=torch.bfloat16).to(model_config.device)
    model.eval()
    
    return model

def get_tokenizer(model_config: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(model_config.model)
    tokenizer.truncation_side = 'left'

    return tokenizer 

def get_ground_truth_logits(model_input, ground_truth, model, tokenizer, model_config: ModelConfig):    
    '''

    - log-probs of gt vector
    - log-probs sum of gt
    - perplexity


    - map to [0, 1]
    - sort list -- take indices divided by length

    '''
    model_input += ground_truth

    model_input_encoding = tokenizer(model_input, return_tensors='pt', truncation=True).to(model_config.device)
    ground_truth_encoding = tokenizer(ground_truth, return_tensors='pt', truncation=True).to(model_config.device)
    
    with torch.no_grad():
        output = model(**model_input_encoding)
        
    logits = output.logits

    ground_truth_tokens = ground_truth_encoding['input_ids'].squeeze(0)
    ground_truth_logits = logits[0, -ground_truth_tokens.shape[0]:, :]  

    targets = ground_truth_tokens
    cross_entropy = - torch.nn.functional.cross_entropy(ground_truth_logits, targets, reduction='none')
    avg_cross_entropy = cross_entropy.mean()
    perplexity = torch.exp(avg_cross_entropy)

    cross_entropy = cross_entropy.cpu().tolist()
    avg_cross_entropy = avg_cross_entropy.cpu().item()
    perplexity = perplexity.cpu().item()

    return cross_entropy, avg_cross_entropy, perplexity