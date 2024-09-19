from vllm import LLM, SamplingParams
from learned_retrieval.collect_generated_data.data_classes import ModelConfig

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_model(model_config: ModelConfig, vllm: bool = False):
    if vllm:
        model = LLM(model=model_config.model,
                    tokenizer=model_config.tokenizer,
                    gpu_memory_utilization=0.8,
                    max_model_len=model_config.max_seq_len,
                    max_seq_len_to_capture=model_config.max_context_len,
                    dtype='bfloat16')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_config.model,
                                                     torch_dtype=torch.bfloat16).to(model_config.device)
        model.eval()
    return model

def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = 'left'

    return tokenizer 

def generate_completions_batch(model_inputs, model, tokenizer, model_config: ModelConfig, vllm: bool = False):
    input_tokens = tokenizer(model_inputs, return_tensors='pt', max_length=model_config.max_context_len, truncation=True, padding=True)

    if vllm:
        sampling_params = SamplingParams(temperature=0,
                                         top_p=1.0,
                                         stop=model_config.stopping_criteria,
                                         max_tokens=model_config.max_completion_len)

        input_ids = input_tokens['input_ids']

        predictions = []
        out = model.generate(prompt_token_ids=input_ids.tolist(),
                             sampling_params=sampling_params,
                             use_tqdm=True)

        for output in out:
            pred = output.outputs[0].text
            predictions.append(pred)
    else:
        input_tokens = {k: v.to(model_config.device) for k, v in input_tokens.items()}

        with torch.no_grad():
            out = model.generate(**input_tokens,
                                 max_new_tokens=model_config.max_completion_len,
                                 stopping_criteria=model_config.stopping_criteria,
                                 pad_token_id=tokenizer.eos_token_id
                                 )

        predictions = []
        for i in range(len(model_inputs)):
            out_tokens = out[i, len(input_tokens['input_ids'][i]) - 1:]
            pred = tokenizer.decode(out_tokens).strip('\n')
            predictions.append(pred)

    return predictions

def generate_completion(model_input, model, tokenizer, model_config: ModelConfig, vllm: bool = False):
    input_tokens = tokenizer(model_input, return_tensors='pt', max_length=model_config.max_context_len, truncation=True)

    if vllm:
        sampling_params = SamplingParams(temperature=0,
                                         top_p=1.0,
                                         stop=model_config.stopping_criteria,
                                         max_tokens=model_config.max_completion_len)

        input_ids = input_tokens['input_ids']

        with torch.no_grad():
            out = model.generate(prompt_token_ids=input_ids[0].tolist(),
                                sampling_params=sampling_params,
                                use_tqdm=False)
        pred = out[0].outputs[0].text
    else:
        input_tokens = {k: v.to(model_config.device) for k, v in input_tokens.items()}

        with torch.no_grad():
            out = model.generate(**input_tokens,
                                max_new_tokens=model_config.max_completion_len,
                                stopping_criteria=model_config.stopping_criteria,
                                pad_token_id=tokenizer.eos_token_id
                                )
        out_tokens = out[0, len(input_tokens['input_ids'][0]) - 1:]
        pred = tokenizer.decode(out_tokens).strip('\n')

    return pred