from typing import List, Optional, Set, Tuple, Union
import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)

class TreeDecoding:
    def __init__(self, model):
        self.model = model
        self.device = model.device

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        return self.model.parameters()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        batch_size, seq_length = input_ids.shape[:2]

        # NOTE: manually call forward

        # NOTE: call LlamaModel
        llama_model = self.model.model

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        # do embedding
        # (bs, seqlen, vocab_size)
        inputs_embeds = llama_model.embed_tokens(input_ids)

        # TODO: prepare TREE attention mask for causal lm
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None

        # embed positions
        hidden_states = inputs_embeds

        for decoder_layer in llama_model.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]

        # NOTE:
        # layer_outputs[0]: hidden_states
        # layer_outputs[1]: output_attentions
        # next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        hidden_states = llama_model.norm(hidden_states)


        # TODO: review logits and understand it
        logits = self.model.lm_head(hidden_states)
        logits = logits.float()

        # TODO: review loss compute
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # TODO: the shape of logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )




