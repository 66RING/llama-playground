import torch
from torch.functional import F
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import argparse
import time
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)


def plot_attention(layers_attn_out, token_chars, num_figs_per_row=4, save_path="all_layers.jpg"):
    '''
    layers_attn_out: 
        list of attention weights from each layer
    '''
    num_layers = len(layers_attn_out)
    num_rows = num_layers // num_figs_per_row
    fig, axes = plt.subplots(num_rows, num_figs_per_row, figsize=(20, 20))
    for layer_idx in tqdm(range(num_layers)):
        row_idx = layer_idx // num_figs_per_row
        col_idx = layer_idx % num_figs_per_row
        # NOTE: batch 0 only
        # mean at heads dim
        # ave_score.shape = (seqlen, seqlen)
        ave_score = layers_attn_out[layer_idx][0].mean(dim=0)
        mask = torch.triu(torch.ones_like(ave_score, dtype=torch.bool), diagonal=1)
        # set axis title
        sns.heatmap(ave_score.numpy(), mask=mask.numpy(), ax=axes[row_idx, col_idx], cmap="YlGnBu", square=True, xticklabels=token_chars, yticklabels=token_chars)
        axes[row_idx, col_idx].set_title(f"Layer {layer_idx}")
        axes[row_idx, col_idx].set_ylabel("query")
        axes[row_idx, col_idx].set_xlabel("key")
    plt.suptitle("all layers ave")
    plt.savefig(save_path)
    plt.close()


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

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        device_map="auto",
        # NOTE: use LlamaAttention to return attention weights 
        attn_implementation="eager",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    prompt = "One day, Lily met a Shoggoth."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    token_chars = tokenizer.convert_ids_to_tokens(input_ids[0])

    outputs = model(input_ids, output_attentions=True)
    attn_scores = outputs.attentions
    # [layer_num][bs, heads, seqlen, seqlen]
    layers_attn_out = [layer_attn.detach().cpu() for layer_attn in attn_scores]

    plot_attention(layers_attn_out, token_chars=token_chars)

if __name__ == "__main__":
    main()

