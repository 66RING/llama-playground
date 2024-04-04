# A tiny llama environment for expermental use 

## roadmap

- [x] basic autoregressive inference
- [x] basic TFLOPS, TPS(token per second) benchmark
- [x] multi gpu inference
- [ ] sequence parallel inference for very large scale input sequence
- [ ] perplexity compute

- generate empty llama of any size with LlamaConfig
- train with random data
- train and inference tinyllama


## Demo1: Run GQA in any size

Generate a llama2 model with GQA:

```bash
python ./zero_init.py \
  --num_key_value_heads 16 \
  --num_attention_heads 128 \
  --num_hidden_layers 32 \
  --hidden_size 128 \
```

run GQA model

```bash
config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
# check config
print(config)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    config=config,
    device_map="auto",
    # TODO:
    # turns out that GQA not work in flash attention as GQA 128 16
    # but work with GQA 64 8 (llama2-70B)

    # attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
```

