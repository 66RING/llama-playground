import numpy as np
import torch
import argparse
import time
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="meta/Llama-2-7b-chat-hf"
    )
    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    # NOTE: add pad_token to use padding
    tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"trainable params {params}\n")

    model.eval()

    past_key_values = None
    prompt = [
        "One day, Lily met a Shoggoth.",
        "Once upon a time, there was a dragon.",
    ]
    batch_size = len(prompt)
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids
    input_ids = input_ids.to(model.device)

    MAX_GEN_LENGTH = 128
    generation_output = model.generate(
        input_ids=input_ids,
        max_new_tokens=MAX_GEN_LENGTH,
        use_cache=True,
        return_dict_in_generate=True)

    outputs = []
    for bid in range(batch_size):
        # NOTE: has padding
        outputs.append(tokenizer.decode(generation_output.sequences[bid,input_ids.shape[1]:]))

    for bid, o in enumerate(outputs):
        print(f"> {bid}: {o}")


if __name__ == "__main__":
    main()

