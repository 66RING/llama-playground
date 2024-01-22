from transformers import AutoModelForCausalLM
from transformers.models.llama import LlamaConfig

def main():
    config = LlamaConfig()
    print(f'config: {config} {type(config)}')

    config.architectures = "LlamaForCausalLM"
    config.bos_token_id = 1
    config.eos_token_id = 2
    config.hidden_act = "silu"
    config.hidden_size = 8
    config.initializer_range = 0.02
    config.intermediate_size = 108
    config.max_position_embeddings = 128
    config.model_type = "llama"
    config.num_hidden_layers = 8
    config.num_attention_heads = 8
    config.num_key_value_heads = 8
    config.pad_token_id = 0
    config.pretraining_tp = 1
    config.rms_norm_eps = 1e-06
    config.rope_scaling = None
    config.tie_word_embeddings = False
    config.torch_dtype = "float16"
    config.transformers_version = "4.31.0"
    config.use_cache = True
    config.vocab_size = 32000

    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained('./llama-zero')
    print("Model size: ", sum(p.numel() for p in model.parameters()))

if __name__ == '__main__':
    main()
