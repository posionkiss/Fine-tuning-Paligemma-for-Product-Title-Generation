from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import PretrainedConfig

class SiglipVisionConfig(PretrainedConfig):
    model_type = "siglip_vision"
    def __init__(
            self,
            hidden_size=768,  # size of embedded vector
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_channels=3,
            image_size=224,
            patch_size=16,
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            num_image_tokens: int = None,
            **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size  # size of embedding vector
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class SiglipVisionModel(nn.Module):
    """
        Take an image, give a batch of embedded vector, each vector correspond to a path of an image
    Input: [batch_size, channels, height, width]
    Output: [batch_size, num_patches, embed_dimension]
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values):
        return self.vision_model(pixel_values=pixel_values)

    def multi_forward(self, pixel_values):
        return self.vision_model.multi_forward(pixel_values=pixel_values)

    # def vision_model(self):


class SiglipVisionTransformer(nn.Module):
    """
    Input: [batch_size, channels, height, width]
    Output: [batch_size, num_patches, embed_dimension]
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)  # backbone of a transformer
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state, attention_weights_list = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state, attention_weights_list

    def multi_forward(self, pixel_values):
        image_embeddings = self.embeddings(pixel_values)
        last_hidden_state, attention_weights_list = self.encoder(inputs_embeds=image_embeddings)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return image_embeddings, last_hidden_state, attention_weights_list


class SiglipVisionEmbeddings(nn.Module):
    """
    Extract patches from original input, and apply positional encoding
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding='valid'
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )

    def forward(self, pixel_values):
        _, _, height, width = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2)  # [bs,embed_dim,num_patches_H,num_patches_W]->[bs,embed_dim,num_patches]
        embeddings = embeddings.transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attention_weights = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states, attention_weights


class SiglipEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds):
        attention_weights_list = []
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states, attn_w = encoder_layer(hidden_states)
            attention_weights_list.append(attn_w)
        return hidden_states, attention_weights_list


class SiglipAttention(nn.Module):
    """Attention is All You Need"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.size()  # [bs,num_patches,embedded_dim]
        query_states = self.q_proj(hidden_states)  # [bs,num_patches,embedded_dim]
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,
                                                                                                       2)  # [bs,num_heads,num_patches, head_dim]
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,
                                                                                                   2)  # [bs,num_heads,num_patches, head_dim]
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,
                                                                                                       2)  # [bs,num_heads,num_patches, head_dim]
        # calculate attention weight
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)
        assert attn_weights.size() == (batch_size, self.num_heads, seq_len, seq_len)

        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float16)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        assert attn_output.size() == (batch_size, self.num_heads, seq_len, self.head_dim)

        attn_output = attn_output.transpose(1, 2).contiguous()  # [bs,num_patches,num_heads, head_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')
        hidden_states = self.fc2(hidden_states)
        return hidden_states
