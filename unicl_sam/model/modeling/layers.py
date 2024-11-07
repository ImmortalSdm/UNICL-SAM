import torch
import torch.nn as nn
from transformers.models.blip_2.modeling_blip_2 import Blip2QFormerIntermediate, Blip2QFormerAttention, Blip2QFormerOutput
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.pytorch_utils import apply_chunking_to_forward

class PromptLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Blip2QFormerAttention(config)
        if config.add_cross_attn:
            self.crossattention = Blip2QFormerAttention(config, is_cross_attention=True)
        self.layer_idx = layer_idx

        self.intermediate_query = Blip2QFormerIntermediate(config)
        self.output_query = Blip2QFormerOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_length=0,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        cross_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must be given for cross-attention layers")
        
        if self.config.add_cross_attn:
            cross_attention_outputs = self.crossattention(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
                past_key_value=cross_attn_past_key_value,
            )
            cross_attention_output = cross_attention_outputs[0]

            outputs = cross_attention_outputs[1:-1]
            present_key_value = cross_attention_outputs[-1]
        else:
            cross_attention_output = hidden_states
            outputs = ()
            present_key_value = ()

        self_attention_outputs = self.attention(
            cross_attention_output,
            attention_mask,
            head_mask,
            output_attentions=output_attentions
        )
        attention_output = self_attention_outputs[0]

        outputs = outputs + self_attention_outputs[1:-1]
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk_query,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )

        outputs = (layer_output,) + outputs
        outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(self, attention_output):
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output

class PromptEncoder(nn.Module):
    def __init__(self, config, use_fp16: bool = False):
        super(PromptEncoder, self).__init__()
        self.config = config
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.hidden_size))

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer = nn.ModuleList(
            [PromptLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        attention_mask=None,
        head_mask=None,
        support_encoder_hidden_states=None,
        support_encoder_attention_mask=None,
        query_encoder_hidden_states=None,
        query_encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        query_length=0,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        next_decoder_cache = () if use_cache else None

        query_tokens = self.query_tokens.expand(support_encoder_hidden_states.shape[0], -1, -1)
        hidden_states = self.layernorm(query_tokens)
        hidden_states = self.dropout(hidden_states)

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if i == 0:
                encoder_hidden_states = support_encoder_hidden_states
                encoder_attention_mask = support_encoder_attention_mask
            else:
                encoder_hidden_states = query_encoder_hidden_states
                encoder_attention_mask = query_encoder_attention_mask

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    query_length,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_cross_attentions = all_cross_attentions + (layer_outputs[1],)
                all_self_attentions = all_self_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
