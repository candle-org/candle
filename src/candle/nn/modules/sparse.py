from ..module import Module
from ..parameter import Parameter
from ..._creation import empty
from .. import functional as F
from .. import init


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                 norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None,
                 _freeze=False, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        if _weight is not None:
            self.weight = Parameter(_weight) if not isinstance(_weight, Parameter) else _weight
        else:
            self.weight = Parameter(empty(num_embeddings, embedding_dim))
            self.reset_parameters()
        if _freeze:
            self.weight.requires_grad = False

    def reset_parameters(self):
        init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self):
        if self.padding_idx is not None:
            from ..._creation import zeros
            self.weight.data[self.padding_idx] = zeros(self.embedding_dim)

    def forward(self, input):
        return F.embedding(input, self.weight, self.padding_idx, self.max_norm,
                           self.norm_type, self.scale_grad_by_freq, self.sparse)

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None, max_norm=None,
                        norm_type=2.0, scale_grad_by_freq=False, sparse=False):
        rows, cols = embeddings.shape
        return cls(rows, cols, padding_idx=padding_idx, max_norm=max_norm,
                   norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                   sparse=sparse, _weight=embeddings, _freeze=freeze)

    def extra_repr(self):
        s = f'{self.num_embeddings}, {self.embedding_dim}'
        if self.padding_idx is not None:
            s += f', padding_idx={self.padding_idx}'
        return s


class EmbeddingBag(Module):
    def __init__(self, num_embeddings, embedding_dim, max_norm=None, norm_type=2.0,
                 scale_grad_by_freq=False, mode='mean', sparse=False, _weight=None,
                 include_last_offset=False, padding_idx=None, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.padding_idx = padding_idx
        if _weight is not None:
            self.weight = Parameter(_weight)
        else:
            self.weight = Parameter(empty(num_embeddings, embedding_dim))
            init.normal_(self.weight)

    def forward(self, input, offsets=None, per_sample_weights=None):
        return F.embedding_bag(input, self.weight, offsets=offsets,
                               mode=self.mode, per_sample_weights=per_sample_weights,
                               padding_idx=self.padding_idx)

    def extra_repr(self):
        return f'{self.num_embeddings}, {self.embedding_dim}, mode={repr(self.mode)}'
