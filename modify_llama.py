from typing import Optional, Tuple

import os
import sys
import pdb
import math
import copy
import time 
import types
import numpy as np 
from scipy.stats import entropy

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    LlamaLinearScalingRotaryEmbedding,
    LlamaForCausalLM,
)


__all__ = ['OrPLMsPoEH2OLlamaForCausalLM', 'OrPLMsPoEH2OLlamaAttention']

from transformers.configuration_utils import PretrainedConfig

LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

class LlamaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
            Llama 2 up to 4096, CodeLlama up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import LlamaModel, LlamaConfig

    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = LlamaConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")
def _make_causal_mask(
    bsz: int, tgt_len: int, past_key_values_length: int, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def apply_rotary_pos_emb_single_scaling(x, cos, sin, position_ids):
    cos = cos[:,position_ids]  # [head, bs, seq_len, dim]
    sin = sin[:,position_ids]  # [head, bs, seq_len, dim]

    cos = cos.transpose(0, 1)  # [bs, head, seq_len, dim]
    sin = sin.transpose(0, 1)  # [bs, head, seq_len, dim]

    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def sample_rotary_emb(cos, sin, num_key_value_groups):
    cos = cos[::num_key_value_groups,...]  # [head, bs, seq_len, dim]
    sin = sin[::num_key_value_groups,...]  # [head, bs, seq_len, dim]
    return cos, sin

class H2OKVCache_LayerWise:
    def __init__(
        self,
        heavy_ratio=0.1,
        recent_ratio=0.1,
        hh_size=None,
        recent_size=None,
        k_seq_dim=2,
        v_seq_dim=2,
        not_accumulated=False,
        num_accumulated=None,
    ):
        print(f"H2OKVCache-LayerWise: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.heavy_ratio = heavy_ratio
        self.recent_ratio = recent_ratio
        if self.hh_size is not None:
            self.cache_size = hh_size + recent_size
        else:
            self.cache_size = None
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None
        self.not_accumulated = not_accumulated
        self.num_accumulated = num_accumulated

    def __call__(self, past_key_values, attn_score_cache):

        self._update_hh_score(attn_score_cache)

        if past_key_values is None:
            return None
        seq_len = past_key_values[0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values

        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values[0].shape

        select_hh_scores = self.hh_score[:, :seq_len - self.recent_size]
        _, keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values

        # keep_recent = torch.arange(seq_len - self.recent_size, seq_len).expand(keep_topk.shape[0], 1).to(keep_topk.device)
        keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
        # print(self.hh_score.size(), keep_topk.size(), keep_recent.size())
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

        inf_attn_weights = torch.full_like(attn_score_cache, float('-inf'))
        # 设置对应列的值为0
        for head in range(num_heads):
            inf_attn_weights[:, head, :, keep_idx[head]] = 0

        # 如果attn_score_cache的第二个维度大于1，将对角线上的元素设置为0
        if attn_score_cache.size(2) > 1:
            eye_mask = torch.eye(seq_len, dtype=torch.bool, device=attn_score_cache.device)
            inf_attn_weights.masked_fill_(eye_mask.unsqueeze(0).unsqueeze(1), 0)
            if self.not_accumulated:
                inf_attn_weights[:,:,:-1,:] = 0.
                # print(inf_attn_weights)
        # print(inf_attn_weights[:,:,:10,:10], inf_attn_weights.size(), keep_idx)
        # sys.exit(0)
        # print(inf_attn_weights, inf_attn_weights.size(), keep_idx)
        # sys.exit(0)
        return inf_attn_weights


    def _update_hh_score(self, attn_score_cache):

        num_new_tokens = attn_score_cache.shape[2]
        # print(f""attn_score_cache.size())
        # print(f"num_new_tokens: {num_new_tokens}")
        if self.hh_score is None:
            # set-up cache size
            if self.hh_size is None:
                self.hh_size = int(self.heavy_ratio * num_new_tokens)
                self.recent_size = int(self.recent_ratio * num_new_tokens)
                self.cache_size = self.hh_size + self.recent_size
            
            if not self.not_accumulated:
                if self.num_accumulated is not None:
                    self.hh_score = attn_score_cache[:,:,-self.num_accumulated:,:].sum(0).sum(1)
                else:
                    self.hh_score = attn_score_cache.sum(0).sum(1)
            else:
                self.hh_score = attn_score_cache[:,:,-1,:][:,:,None,:].sum(0).sum(1)
        else:
            # print(attn_score_cache.size(), self.hh_score.size())
            attn_score_cache = attn_score_cache.sum(0).sum(1)
            if not self.not_accumulated:
                attn_score_cache[:, :-num_new_tokens] += self.hh_score
            self.hh_score = attn_score_cache
            # print(self.hh_size, self.recent_size, self.cache_size)

    def _clean_scores(self):
        self.hh_score = None
### Positional Scaling
class MsPoELlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, min_cratio=1, max_cratio=3, num_heads=32, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.min_ratio = min_cratio
        self.max_ratio = max_cratio
        self.num_heads = num_heads

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        min_ratio = self.min_ratio
        max_ratio = self.max_ratio
        num_heads = self.num_heads
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype).repeat(num_heads,1)
        compress_ratio = torch.arange(num_heads, device=device, dtype=self.inv_freq.dtype)
        compress_ratio = min_ratio + (max_ratio - min_ratio) * (compress_ratio / num_heads)
        compress_ratio = compress_ratio.unsqueeze(-1)

        t = t / compress_ratio
        freqs = torch.einsum("ki,j->kij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:,:seq_len].to(dtype=x.dtype),
            self.sin_cached[:,:seq_len].to(dtype=x.dtype),
        )


class OrPLMsPoEH2OLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: LlamaConfig, layer_idx: int, layer_decoding_query: dict):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.compress_ratio_min = config.compress_ratio_min
        self.compress_ratio_max = config.compress_ratio_max

        self.enable_head_metrics = True
        self.head_type = config.head_type
        self.head_order = None

        self._init_rope()
        self.kv_cache = H2OKVCache_LayerWise(
            heavy_ratio=config.heavy_ratio,
            recent_ratio=config.recent_ratio,
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            not_accumulated=config.not_accumulated,
            num_accumulated=config.num_accumulated
        )
        self.layer_decoding_query = layer_decoding_query
        self.layer_idx = layer_idx
        self.use_cycle = config.use_cycle
        self.less_dim = config.less_dim

    def _head_wise_statistics(self, query_states, key_states, q_len, kv_seq_len, bsz, attention_mask):

        query_states_new = query_states
        key_states_new = repeat_kv(key_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states_new, key_states_new.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        if len(attn_weights.shape) == 4:
            attn_weights = attn_weights.squeeze(0)

        head_orders = self._calculate_outlier(attn_weights)

        return head_orders


    def _calculate_outlier(self, attn_weights):
        # attn_weights: [num_heads, q_len, kv_seq_len]
        average = attn_weights.mean(-1).unsqueeze(-1)
        outlier = - (attn_weights > 3 * average).float().mean(-1)[:,-1]
        head_orders = outlier.argsort()

        if self.head_type == "normal":
            head_orders = np.arange(self.num_heads)
            head_orders = self.num_heads - head_orders - 1

        return head_orders


    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = MsPoELlamaRotaryEmbedding(
                self.head_dim,
                min_cratio=self.compress_ratio_min,
                max_cratio=self.compress_ratio_max,
                num_heads=self.num_heads,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                assert False # not implemented
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    min_cratio=self.compress_ratio_min,
                    max_cratio=self.compress_ratio_max,
                    num_heads=self.num_heads,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                assert False # not implemented
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def _clean_cache(self):
        self.kv_cache._clean_scores()
        self.layer_decoding_query[self.layer_idx] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # remake causal mask
        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=past_key_value[0].shape[-2] if past_key_value is not None else 0,
            dtype=query_states.dtype,
            device=query_states.device,
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        position_length = kv_seq_len
        if not position_ids.nelement() > 1:
            if position_length < position_ids.item()+1:
                position_length = position_ids.item()+1

        cos, sin = self.rotary_emb(value_states, seq_len=position_length)

        if self.enable_head_metrics:
            self.head_order = self._head_wise_statistics(query_states, key_states, q_len, kv_seq_len, bsz, attention_mask)
            self.enable_head_metrics = False

        cos = cos[self.head_order, :, :]
        sin = sin[self.head_order, :, :]
        query_states = apply_rotary_pos_emb_single_scaling(query_states, cos, sin, position_ids)

        cos, sin = sample_rotary_emb(cos, sin, self.num_key_value_groups)
        key_states = apply_rotary_pos_emb_single_scaling(key_states, cos, sin, position_ids)

        prefill=True
        if past_key_value is not None:
            prefill=False
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # key/value are already rotated
        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )
        #############################
        # 计算当前认为重要的kv 的mask
        #############################
        # print(prefill)
        if prefill:
            if self.use_cycle:
                self.layer_decoding_query[self.layer_idx] = hidden_states[:,-1:,:].detach().clone()
                # print(prefill, hidden_states[:,-1:,:].size())
            inf_attn_mask = self.kv_cache(past_key_value, attn_weights.detach().clone())
        else:
            self.layer_decoding_query[self.layer_idx] = hidden_states.detach().clone()
            
            if self.layer_idx==0 and self.use_cycle:
                _query_states = self.q_proj(self.layer_decoding_query[max(self.layer_decoding_query.keys())])
                _query_states = _query_states.view(
                                bsz, q_len, self.num_heads, self.head_dim
                            ).transpose(1, 2)
                _query_states = apply_rotary_pos_emb_single_scaling(_query_states, cos, sin, position_ids)
                attn_weights_for_prefetch = torch.matmul(_query_states, key_states.transpose(2, 3)) / math.sqrt(
                                                self.head_dim
                                            )
                inf_attn_mask = self.kv_cache(past_key_value, attn_weights_for_prefetch)
                attn_weights = attn_weights + inf_attn_mask
                # print(prefill, hidden_states.size(),_query_states.size(), attn_weights_for_prefetch.size())
            elif self.layer_idx==0 and not self.use_cycle:
                pass
            else:
                # print(prefill, self.layer_idx, self.layer_decoding_query[self.layer_idx - 1].size(),past_key_value[0].size(), past_key_value[1].size())
                _query_states = self.q_proj(self.layer_decoding_query[self.layer_idx - 1])
                _query_states = _query_states.view(
                                bsz, q_len, self.num_heads, self.head_dim
                            ).transpose(1, 2)
                # print("*******",_query_states.size())
                _query_states = apply_rotary_pos_emb_single_scaling(_query_states, cos, sin, position_ids)
                # print("*******",_query_states.size(), key_states.size())

                
                #######################
                # 对attn 进行近似计算
                #######################
                # 获取query的dim维度的绝对值
                _query_abs = torch.abs(_query_states)

                # 对query的dim维度的绝对值进行topk筛选
                _, topk_indices = torch.topk(_query_abs, self.less_dim, dim=-1)  # 1, 32, 1, 16

                # 根据topk索引选择query的对应维度
                _query_states_topk = torch.gather(_query_states, -1, topk_indices)

                # 根据topk索引选择key_states的对应维度
                _key_states_topk = torch.gather(key_states, -1, topk_indices.repeat(1, 1, key_states.size(2), 1))
                # print(_query_states_topk.size(), _key_states_topk.size(),topk_indices)
                attn_weights_for_prefetch = torch.matmul(_query_states_topk, _key_states_topk.transpose(2, 3)) / math.sqrt(
                                                self.less_dim
                                            )
                inf_attn_mask = self.kv_cache(past_key_value, attn_weights_for_prefetch)
                ####################################
                
                attn_weights = attn_weights + inf_attn_mask
                # print(prefill, hidden_states.size(),_query_states.size(), attn_weights_for_prefetch.size())
            # attn_weights = attn_weights + inf_attn_mask

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
class OrPLH2OLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int, layer_decoding_query: dict):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()
        
        self.kv_cache = H2OKVCache_LayerWise(
            heavy_ratio=config.heavy_ratio,
            recent_ratio=config.recent_ratio,
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            not_accumulated=config.not_accumulated,
            num_accumulated=config.num_accumulated
        )
        self.layer_decoding_query = layer_decoding_query
        self.layer_idx = layer_idx
        self.use_cycle = config.use_cycle
        self.less_dim = config.less_dim
        self.first_3_hidden = config.first_3_hidden

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _clean_cache(self):
        self.kv_cache._clean_scores()
        self.layer_decoding_query[self.layer_idx] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # remake causal mask
        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=past_key_value[0].shape[-2] if past_key_value is not None else 0,
            dtype=query_states.dtype,
            device=query_states.device,
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        position_length = kv_seq_len
        if not position_ids.nelement() > 1:
            if position_length < position_ids.item()+1:
                position_length = position_ids.item()+1

        cos, sin = self.rotary_emb(value_states, seq_len=position_length)
        ### Shift Pos: query pos is min(cache_size, idx)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)
        
        prefill=True
        if past_key_value is not None:
            prefill=False
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )
        
        # 计算当前认为重要的kv 的mask
        # print(prefill)
        if prefill:
            if self.use_cycle:
                self.layer_decoding_query[self.layer_idx] = hidden_states[:,-1:,:].detach().clone()
                # print(prefill, hidden_states[:,-1:,:].size())
            inf_attn_mask = self.kv_cache(past_key_value, attn_weights.detach().clone())
        else:
            self.layer_decoding_query[self.layer_idx] = hidden_states.detach().clone()
            
            if self.layer_idx==0 and self.use_cycle:
                _query_states = self.q_proj(self.layer_decoding_query[max(self.layer_decoding_query.keys())])
                _query_states = _query_states.view(
                                bsz, q_len, self.num_heads, self.head_dim
                            ).transpose(1, 2)
                _query_states = apply_rotary_pos_emb_single(_query_states, cos, sin, position_ids)
                attn_weights_for_prefetch = torch.matmul(_query_states, key_states.transpose(2, 3)) / math.sqrt(
                                                self.head_dim
                                            )
                inf_attn_mask = self.kv_cache(past_key_value, attn_weights_for_prefetch)
                attn_weights = attn_weights + inf_attn_mask
                # print(prefill, hidden_states.size(),_query_states.size(), attn_weights_for_prefetch.size())
            elif self.layer_idx==0 and not self.use_cycle:
                pass
            else:
#                 if self.first_3_hidden:
                    
#                 else:
                _query_states = self.q_proj(self.layer_decoding_query[self.layer_idx - 1])
                _query_states = _query_states.view(
                                bsz, q_len, self.num_heads, self.head_dim
                            ).transpose(1, 2)
                _query_states = apply_rotary_pos_emb_single(_query_states, cos, sin, position_ids)

                #######################
                # 对attn 进行近似计算
                #######################
                # 获取query的dim维度的绝对值
                _query_abs = torch.abs(_query_states)

                # 对query的dim维度的绝对值进行topk筛选
                _, topk_indices = torch.topk(_query_abs, self.less_dim, dim=-1)  # 1, 32, 1, 16

                # 根据topk索引选择query的对应维度
                _query_states_topk = torch.gather(_query_states, -1, topk_indices)

                # 根据topk索引选择key_states的对应维度
                _key_states_topk = torch.gather(key_states, -1, topk_indices.repeat(1, 1, key_states.size(2), 1))
                # print(_query_states_topk.size(), _key_states_topk.size(),topk_indices)
                attn_weights_for_prefetch = torch.matmul(_query_states_topk, _key_states_topk.transpose(2, 3)) / math.sqrt(
                                                self.less_dim
                                            )
                inf_attn_mask = self.kv_cache(past_key_value, attn_weights_for_prefetch)
                ####################################
                
                attn_weights = attn_weights + inf_attn_mask
                # print(prefill, hidden_states.size(),_query_states.size(), attn_weights_for_prefetch.size())
            # attn_weights = attn_weights + inf_attn_mask

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        # past_key_value = self.kv_cache(past_key_value, attn_weights.detach().clone())

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class OrPLMsPoEH2OLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        self.layer_decoding_query={}
        for layer_idx in range(num_layers):
            self.layer_decoding_query[layer_idx] = None
        for layer_idx in range(num_layers):
            if layer_idx in config.apply_layers:
                self.model.layers[layer_idx].self_attn = OrPLMsPoEH2OLlamaAttention(config, layer_idx, self.layer_decoding_query)
            else:
                self.model.layers[layer_idx].self_attn = OrPLH2OLlamaAttention(config, layer_idx, self.layer_decoding_query)

    def _reset(self):
        for layer_idx in self.config.apply_layers:
            self.model.layers[layer_idx].self_attn.enable_head_metrics = True
            self.model.layers[layer_idx].self_attn.head_order = None

