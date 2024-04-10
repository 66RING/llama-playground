import numpy as np
import torch
import argparse
import time
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)

def inference(model, tokenizer, input_ids, past_key_values, max_gen_len):
    prefill_time = 0
    decode_time = []
    batch_size = input_ids.size(0)

    # prefill phase
    start = time.time()
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    torch.cuda.synchronize()
    end = time.time()
    prefill_time = (end - start)

    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    for _ in range(max_gen_len - 1):
        start = time.time()
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        torch.cuda.synchronize()
        end = time.time()
        decode_time.append(end - start)

        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        if now > pos:
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    print(" ".join(generated_text[pos:]), flush=True)
    return past_key_values, prefill_time, decode_time


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
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    print(config)

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
    prompt = "One day, Lily met a Shoggoth."
    batch_size = 1
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)

    past_key_values, prefill_time, decode_time = inference(
        model, tokenizer, input_ids, past_key_values, max_gen_len=128
    )

    # number of tokens in context / time for processing context * batch size
    prefill_tokens_per_second = input_ids.shape[1] / prefill_time * batch_size
    # 1 second / median time per token in seconds * batch size
    decode_tokens_per_second = 1 / np.median(decode_time) * batch_size

    device = next(model.parameters()).device
    memory_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    memory_pct = memory_used / (torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)) * 100

    print(f" ** Speed (Prefill): {prefill_tokens_per_second:.2f} tokens/second")
    print(f" ** Speed (Decode): {decode_tokens_per_second:.2f} tokens/second")
    print(f" ** Max Memory (VRAM): {memory_used:.2f} GB ({memory_pct:.2f}%)")



if __name__ == "__main__":
    main()

