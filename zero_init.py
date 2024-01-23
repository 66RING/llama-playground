from transformers import AutoModelForCausalLM
from transformers.models.llama import LlamaConfig
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_key_value_heads", type=int, default=8
    )
    parser.add_argument(
        "--num_attention_heads", type=int, default=8
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=8
    )
    parser.add_argument(
        "--hidden_size", type=int, default=8
    )
    parser.add_argument(
        "--max_position_embeddings", type=int, default=128
    )
    parser.add_argument(
        "--output", type=str, default="./llama-zero"
    )
    args = parser.parse_args()

    config = LlamaConfig()
    config.architectures = "LlamaForCausalLM"
    config.bos_token_id = 1
    config.eos_token_id = 2
    config.hidden_act = "silu"
    config.initializer_range = 0.02
    config.intermediate_size = 108
    config.model_type = "llama"
    config.max_position_embeddings = args.max_position_embeddings
    config.hidden_size = args.hidden_size
    config.num_hidden_layers = args.num_hidden_layers
    config.num_attention_heads = args.num_attention_heads
    config.num_key_value_heads = args.num_key_value_heads
    config.pad_token_id = 0
    config.pretraining_tp = 1
    config.rms_norm_eps = 1e-06
    config.rope_scaling = None
    config.tie_word_embeddings = False
    config.torch_dtype = "float16"
    config.transformers_version = "4.31.0"
    config.use_cache = True
    config.vocab_size = 32000

    print(f'config to save: {config} {type(config)}')
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(args.output)
    print("Model size: ", sum(p.numel() for p in model.parameters()))

if __name__ == '__main__':
    main()
