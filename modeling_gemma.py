# from itertools import repeat

import torch
# from click.core import batch
# from dask.cli import config_get
# from llama.model import apply_rotary_emb, repeat_kv
# from tomlkit import key_value
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from transformers import PreTrainedModel, PretrainedConfig
from torch.nn.functional import dropout

from modeling_siglip import SiglipVisionConfig, SiglipVisionModel


class KVCache:
    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self):
        if len(self.key_cache) == 0:
            return 0
        else:
            return self.key_cache[0].shape[-2]

    def update(self, key_states, value_states, layer_idx):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def clear_trace(self):
        for i in range(len(self.key_cache)):
            self.key_cache[i] = self.key_cache[i].detach()
        for i in range(len(self.value_cache)):
            self.value_cache[i] = self.value_cache[i].detach()


class GemmaConfig(PretrainedConfig):
    model_type = "gemma"
    def __init__(
            self,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim=256,
            max_position_embeddings=8192,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            attention_bias=False,
            attention_dropout=0.0,
            pad_token_id=None, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class PaliGemmaConfig(PretrainedConfig):
    model_type = "paligemma"
    def __init__(self, vision_config=None, text_config=None, ignore_index=-100,
                 image_token_index=256000, vocab_size=257152, projection_dim=2048, hidden_size=2048,
                 pad_token_id=None, **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoConfig
        import json, os

        # 加载 config.json
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_path, "r") as f:
            raw_config = json.load(f)

        text_config_dict = raw_config.pop("text_config", None)
        vision_config_dict = raw_config.pop("vision_config", None)

        config = cls(
            vision_config=vision_config_dict,
            text_config=text_config_dict,
            ignore_index=raw_config.get("ignore_index", -100),
            image_token_index=raw_config.get("image_token_index", 256000),
            projection_dim=raw_config.get("projection_dim", 2048),
            hidden_size=raw_config.get("hidden_size", 2048),
            pad_token_id=raw_config.get("pad_token_id", 0),
            **kwargs,
        )

        # 构建子配置
        config.text_config = GemmaConfig(**text_config_dict)
        config.vision_config = SiglipVisionConfig(**vision_config_dict)

        # 自动推导字段
        config.vocab_size = config.text_config.vocab_size
        config.text_config.num_image_tokens = (
            config.vision_config.image_size // config.vision_config.patch_size
        ) ** 2
        config.vision_config.projection_dim = config.projection_dim
        return config
        
        
class GemmaRotaryEmbeddings(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # θi = 10000^(-2i/d)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer('inv_freq', tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len):
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # x1 * cosθ1 - x2 * sinθ2
    q_embed = (q * torch.transpose(cos,0,2)) + (rotate_half(q) * torch.transpose(sin,0,2))
    k_embed = (k * torch.transpose(cos,0,2)) + (rotate_half(k) * torch.transpose(sin,0,2))
    return q_embed, k_embed


def rotate_half(x):  # 重要！自己整理的embeddings可能也需要经过某种重新排序
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0

        # 分组查询
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = GemmaRotaryEmbeddings(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta
        )

    def forward(self, hidden_states, attention_mask, position_ids, kv_cache, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = torch.transpose(query_states.view(bsz, q_len, self.num_heads, self.head_dim), 1, 2)
        key_states = torch.transpose(key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim), 1, 2)
        value_states = torch.transpose(value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim), 1, 2)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, torch.transpose(key_states, 2, 3)) / math.sqrt(self.head_dim)

        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float16).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
            # Make sure the sequence length is the second dimension. # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim] -> [Batch_Size, Seq_Len_Q, Num_Heads_Q, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Concatenate all the heads together. [Batch_Size, Seq_Len_Q, Num_Heads_Q, Head_Dim] -> [Batch_Size, Seq_Len_Q, Num_Heads_Q * Head_Dim]
        attn_output = attn_output.view(bsz, q_len, -1)
        # Multiply by W_o. [Batch_Size, Seq_Len_Q, Hidden_Size]

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # gate projection WHY?
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate='tanh') * self.up_proj(x))


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask, position_ids, kv_cache):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attention_weight = self.self_attn(hidden_states=hidden_states,
                                          attention_mask=attention_mask,
                                          position_ids=position_ids,
                                          kv_cache=kv_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attention_weight


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class GemmaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, attention_mask, position_ids, inputs_embeds, kv_cache):
        hidden_states = inputs_embeds
        normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        attn_weights_list = []
        for i, decoder_layer in enumerate(self.layers):
            hidden_states, attention_weight = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache
            )
            attn_weights_list.append(attention_weight)
        hidden_states = self.norm(hidden_states)
        return hidden_states, attn_weights_list


class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)


    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        kv_cache=None,
        labels=None,
        past_key_values=None,
        **kwargs,
    ):
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        outputs, attention_weight_list = self.model(attention_mask=attention_mask, position_ids=position_ids, inputs_embeds=inputs_embeds,
                             kv_cache=kv_cache)
        hidden_states = outputs
#         print(hidden_states.dtype)
#         print(torch.isnan(hidden_states).sum())
#         print(hidden_states)
        logits = self.lm_head(hidden_states) ### 
#         logits = logits.float()
        returning_data = {'logits': logits}
        if kv_cache is not None:
            returning_data['kv_cache'] = kv_cache
        return returning_data, attention_weight_list[0], outputs
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear(image_features)
        return hidden_states


class PaliGemmaForConditionalGeneration(PreTrainedModel):
    config_class = PaliGemmaConfig
    _no_split_modules = ["language_model"] 
    def __init__(self, config: PaliGemmaConfig):
        super().__init__(config)
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size
        # self.image_feature_to_color = nn.Linear(config.hidden_size, 17)
        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
            self, image_features, inputs_embeds, input_ids, attention_mask, kv_cache=None):
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        # [bs,seq_len,hidden_size]
        scaled_image_features = image_features / (self.config.hidden_size ** 0.5)

        # combined embeddings
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype,
                                      device=inputs_embeds.device)
        # [bs, seq_len]
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # [bs, seq_len]
        image_mask = input_ids == self.config.image_token_index
        # [bs, seq_len]
        pad_mask = input_ids == self.pad_token_id

        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embeddings
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # Insert image embeddings. We can't use torch.where because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
#         print(image_mask_expanded.dtype)
#         print(scaled_image_features.dtype)
#         print(final_embedding.dtype)
        final_embedding = final_embedding.half().masked_scatter(image_mask_expanded,scaled_image_features).type(torch.cuda.FloatTensor)
#         final_embedding = final_embedding.half().masked_scatter(image_mask_expanded, scaled_image_features) ###

        # Zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        #### CREATE THE ATTENTION MASK ####
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            causal_mask = torch.full((batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device)
        else:
            # assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            causal_mask = torch.full((batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device)

        # generate head dimension
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None or kv_cache.num_items() > 0:
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)
        return final_embedding, causal_mask, position_ids

    def forward(self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None, kv_cache=None):
        # assert torch.all(attention_mask == 1), 'The input cannot be padded'

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        selected_image_feature, attention_weight_list = self.vision_tower(pixel_values)

        image_features = self.multi_modal_projector(selected_image_feature)

        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, kv_cache)

        outputs, gemma_attention_weight,_ = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache)
        return outputs, attention_weight_list, gemma_attention_weight

    def forward_partly_gradient(self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None, kv_cache=None):
        # assert torch.all(attention_mask == 1), 'The input cannot be padded'

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        selected_image_feature, attention_weight_list = self.vision_tower(pixel_values)

        selected_image_feature = selected_image_feature.detach()
        attention_weight_list = attention_weight_list.detach()

        image_features = self.multi_modal_projector(selected_image_feature)

        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, kv_cache)

        outputs, gemma_attention_weight = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache
        )
        return outputs, attention_weight_list

    def distract_image_features(self,
                input_ids=None,
                pixel_values=None):

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
#         selected_image_feature = self.vision_tower.vision_model.embeddings(pixel_values.to(inputs_embeds.dtype))
#         return selected_image_feature
        selected_image_feature, attention_weight_list = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        return selected_image_feature

    def multi_forward(self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None, kv_cache=None):

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        image_embeddings, selected_image_feature, _ \
            = self.vision_tower.multi_forward(pixel_values)

        image_features = self.multi_modal_projector(selected_image_feature)

        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, kv_cache)

        outputs, _, features = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache)
        return outputs, image_embeddings, selected_image_feature, features
    
    

