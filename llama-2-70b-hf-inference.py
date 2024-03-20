# https://github.com/notrichardren/llama-2-70b-hf-inference/blob/main/inference.py

import os
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download, snapshot_download
import torch

MODEL_NAME = f"meta-llama/Llama-2-70b-hf"
# model path OR checkpoint path
WEIGHTS_DIR = f"{os.getcwd()}/llama-weights-70b"

checkpoint_location = WEIGHTS_DIR

# Load model

with init_empty_weights():
    config = AutoConfig.from_pretrained(checkpoint_location, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    # model = LlamaForCausalLM.from_pretrained(checkpoint_location)

model = load_checkpoint_and_dispatch(
    model,
    checkpoint_location,
    device_map="auto",
    offload_folder=WEIGHTS_DIR,
    dtype=torch.float16,
    no_split_module_classes=["LlamaDecoderLayer"],
)
tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(checkpoint_location)


# Use model
print(tokenizer.decode(model.generate(**({ k: torch.unsqueeze(torch.tensor(v), 0) for k,v in tokenizer("Hi there, how are you doing?").items()}), max_new_tokens = 20).squeeze()))
