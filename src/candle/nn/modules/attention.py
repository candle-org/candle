import math

from ..module import Module
from ..parameter import Parameter
from ..._creation import empty
from .. import functional as F
from .. import init as init


class MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
                 add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
                 batch_first=False, device=None, dtype=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        if self._qkv_same_embed_dim:
            self.in_proj_weight = Parameter(empty(3 * embed_dim, embed_dim, device=device, dtype=dtype))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)
        else:
            self.register_parameter('in_proj_weight', None)
            self.q_proj_weight = Parameter(empty(embed_dim, embed_dim, device=device, dtype=dtype))
            self.k_proj_weight = Parameter(empty(embed_dim, self.kdim, device=device, dtype=dtype))
            self.v_proj_weight = Parameter(empty(embed_dim, self.vdim, device=device, dtype=dtype))

        if bias:
            self.in_proj_bias = Parameter(empty(3 * embed_dim, device=device, dtype=dtype))
        else:
            self.register_parameter('in_proj_bias', None)

        from .linear import Linear
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            init.xavier_uniform_(self.in_proj_weight)
        else:
            init.xavier_uniform_(self.q_proj_weight)
            init.xavier_uniform_(self.k_proj_weight)
            init.xavier_uniform_(self.v_proj_weight)
        if self.in_proj_bias is not None:
            init.constant_(self.in_proj_bias, 0.)
            init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        is_self_attention = query is key and key is value

        if self.batch_first:
            bsz, tgt_len, embed_dim = query.shape
            src_len = key.shape[1]
        else:
            tgt_len, bsz, embed_dim = query.shape
            src_len = key.shape[0]

        # Project Q, K, V.  Self-attention can use PyTorch's packed projection
        # pattern: one linear with the packed in_proj_weight/bias, then split
        # the last dimension into q/k/v views.  Cross-attention keeps separate
        # projections because query and key/value have distinct sequence data.
        if self._qkv_same_embed_dim and is_self_attention:
            packed_qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            q, k, v = F._split_packed_qkv_projection(
                packed_qkv, embed_dim, self.num_heads, batch_first=self.batch_first
            )
        else:
            if self._qkv_same_embed_dim:
                w_q = self.in_proj_weight[:embed_dim]
                w_k = self.in_proj_weight[embed_dim:2*embed_dim]
                w_v = self.in_proj_weight[2*embed_dim:]
            else:
                w_q = self.q_proj_weight
                w_k = self.k_proj_weight
                w_v = self.v_proj_weight

            if self.in_proj_bias is not None:
                b_q = self.in_proj_bias[:embed_dim]
                b_k = self.in_proj_bias[embed_dim:2*embed_dim]
                b_v = self.in_proj_bias[2*embed_dim:]
            else:
                b_q = b_k = b_v = None

            q = F.linear(query, w_q, b_q)
            k = F.linear(key, w_k, b_k)
            v = F.linear(value, w_v, b_v)

        # Reshape: (L, N, E) -> (L, N, H, D) -> (N, H, L, D)
        head_dim = self.head_dim
        num_heads = self.num_heads

        if not (self._qkv_same_embed_dim and is_self_attention):
            if self.batch_first:
                # (N, L, H, D) -t(1,2)-> (N, H, L, D)
                q = q.reshape(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
                k = k.reshape(bsz, src_len, num_heads, head_dim).transpose(1, 2)
                v = v.reshape(bsz, src_len, num_heads, head_dim).transpose(1, 2)
            else:
                # (L, N, H, D) -t(0,1)-> (N, L, H, D) -t(1,2)-> (N, H, L, D)
                q = q.reshape(tgt_len, bsz, num_heads, head_dim).transpose(0, 1).transpose(1, 2)
                k = k.reshape(src_len, bsz, num_heads, head_dim).transpose(0, 1).transpose(1, 2)
                v = v.reshape(src_len, bsz, num_heads, head_dim).transpose(0, 1).transpose(1, 2)

        # Handle key_padding_mask: (N, S) bool where True = ignore
        if key_padding_mask is not None:
            from ..._creation import tensor as _tensor
            from ..._functional import where as _where
            # Expand to (N, 1, 1, S) for broadcasting with (N, H, L, S)
            kpm = key_padding_mask.reshape(bsz, 1, 1, src_len)
            neg_inf = _tensor(float('-inf'), device=query.device)
            zero = _tensor(0.0, device=query.device)
            kpm_mask = _where(kpm, neg_inf, zero)
            if attn_mask is not None:
                from ..._functional import add as _add
                attn_mask = _add(attn_mask, kpm_mask)
            else:
                attn_mask = kpm_mask

        dropout_p = self.dropout if self.training else 0.0

        if need_weights:
            # Manual attention computation to return weights
            from ..._functional import matmul as _matmul
            scale = 1.0 / math.sqrt(head_dim)
            attn_weights = _matmul(q, k.transpose(-2, -1)) * scale
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = _matmul(attn_weights, v)
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

        # (N, H, L, D) -> (N, L, H, D) -> (N, L, E).  For batch_first
        # callers, keep this contiguous B,L,E layout through the output
        # projection; linear acts on the last dimension, so this is equivalent
        # to projecting the transposed L,N,E view while avoiding a non-contig
        # projection input on accelerator backends.
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        if self.batch_first:
            attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        else:
            attn_output = attn_output.transpose(0, 1)
            attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights
        return attn_output, None

    def extra_repr(self):
        return f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, dropout={self.dropout}'
