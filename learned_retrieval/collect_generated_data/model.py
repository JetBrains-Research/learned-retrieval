from vllm import LLM, SamplingParams
from learned_retrieval.collect_generated_data.data_classes import ModelConfig

import torch
from transformers import AutoModelForCausalLM

def get_model(model_config: ModelConfig, vllm: bool = False):
    if vllm:
        model = LLM(model=model_config.model,
                    tokenizer=model_config.tokenizer,
                    gpu_memory_utilization=0.4,
                    max_model_len=model_config.max_seq_len,
                    max_seq_len_to_capture=model_config.max_context_len,
                    # distributed_executor_backend='mp',
                    # tensor_parallel_size=2,
                    dtype='bfloat16')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_config.model,
                                                     torch_dtype=torch.bfloat16).to(model_config.device)

    return model

def generate_completion(model_input, model, tokenizer, model_config: ModelConfig, vllm: bool = False):
    if vllm:
        sampling_params = SamplingParams(temperature=0,
                                        stop=model_config.stopping_criteria,
                                        max_tokens=model_config.max_completion_len)

        input_tokens = tokenizer(model_input, return_tensors='pt')
        input_ids = input_tokens['input_ids'][:, -model_config.max_context_len:]

        with torch.no_grad():
            out = model.generate(prompt_token_ids=input_ids[0].tolist(),
                                sampling_params=sampling_params,
                                use_tqdm=False)
        pred = out[0].outputs[0].text
    else:
        input_tokens = tokenizer(model_input, return_tensors='pt')
        input_ids = input_tokens['input_ids'][:, -model_config.max_context_len:].to(model_config.device)

        att_mask = input_tokens['attention_mask'][:, -model_config.max_context_len:].to(model_config.device)

        with torch.no_grad():
            out = model.generate(input_ids,
                                attention_mask=att_mask,
                                max_new_tokens=model_config.max_completion_len,
                                stopping_criteria=model_config.stopping_criteria,
                                pad_token_id=tokenizer.eos_token_id
                                )
        out_tokens = out[0, len(input_ids[0]) - 1:]
        pred = tokenizer.decode(out_tokens).strip('\n')

    return pred