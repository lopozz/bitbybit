import torch
from torch import nn, Tensor
from torch.nn import functional as F


class HeadAttention(nn.Module):
    """
    Single-head scaled dot-product causal attention.

    This module implements the core of "scaled dot-product attention" used in
    Transformer models (Vaswani et al., 2017). It projects the input to key,
    query and value vectors (per-head dimensionality) and computes causal
    (autoregressive) attention across the sequence.

    References
    ----------
    - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
      Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need.
      https://arxiv.org/abs/1706.03762

    """

    def __init__(self, embed_dim: int, num_heads: int, attn_pdrop: float = 0.2):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {embed_dim} and `num_heads`:"
                f" {num_heads})."
            )

        self.head_dim = embed_dim // num_heads

        # Linear projections for key, query, and value
        self.Wk = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.Wq = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, self.head_dim, bias=False)

        self.attn_dropout = nn.Dropout(attn_pdrop)

    def forward(self, x: Tensor) -> Tensor:
        _, seq_len, _ = x.shape

        k = self.Wk(x)  # (B,T,C)
        q = self.Wq(x)  # (B,T,C)
        v = self.Wv(x)  # (B,T,C)

        # Scaled dot-product attention scores
        scores = q @ k.transpose(-2, -1) * self.head_dim**-0.5  # (B, T, T)

        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)) == 0
        scores = scores.masked_fill(mask, float("-inf"))  # (B, T, T)

        # Attention weights and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum of values
        out = attn @ v  # (B, T, C)

        return out


batch_size = 3
max_tokens = 2048
# next_pos = torch.tensor([3, 3, 2])


class KVCacheHeadAttention(nn.Module):
    """
    Single-head causal attention with key–value (KV) caching for efficient autoregressive decoding.

    This module extends scaled dot-product causal attention (Vaswani et al., 2017) by caching the
    projected key and value tensors from previous tokens. During incremental generation, new queries
    attend to both past cached keys/values and the current step, avoiding recomputation and
    enabling linear-time decoding.

    The cache is stored as a static buffer of shape:
        (2, batch_size, max_tokens, head_dim)

    Reference
    - Vaswani, A., et al. (2017). *Attention Is All You Need*. https://arxiv.org/abs/1706.03762
    - Pope, R., et al. (2023). *Efficiently Scaling Transformer Inference*. https://arxiv.org/abs/2211.05102
    """
    def __init__(self, embed_dim: int, num_heads: int, attn_pdrop: float = 0.2):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {embed_dim} and `num_heads`:"
                f" {num_heads})."
            )

        self.head_dim = embed_dim // num_heads

        # Linear projections for key, query, and value
        self.Wk = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.Wq = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, self.head_dim, bias=False)

        self.attn_dropout = nn.Dropout(attn_pdrop)

        self.kv_cache = torch.zeros((2, batch_size, max_tokens, self.head_dim))

    def forward(self, x: Tensor):
        batch_size, seq_len, _ = x.shape

        k = self.Wk(x)  # (B,T,C)
        q = self.Wq(x)  # (B,T,C)
        v = self.Wv(x)  # (B,T,C)

        if (
            seq_len == 1
        ):  # this is decondg phase so start incrementing the last position
            for i in range(batch_size):
                s = int(next_pos[i].item())
                e = s + seq_len
                self.kv_cache[0, i, s:e, :] = k[i]
                self.kv_cache[1, i, s:e, :] = v[i]
                next_pos[i] = e

        max_k = int(next_pos.max().item())
        k_cache = self.kv_cache[0, :batch_size, :max_k, :]  # (B, max_k, C)
        v_cache = self.kv_cache[1, :batch_size, :max_k, :]  # (B, max_k, C)

        # Scaled dot-product attention scores
        scores = q @ k_cache.transpose(-2, -1) * self.head_dim**-0.5  # (B, T, max_k)

        # Causal mask
        mask = torch.tril(torch.ones(max_k, max_k, device=x.device)) == 0
        scores = scores.masked_fill(mask, float("-inf"))  # (B, T, max_k)

        # Attention weights and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum of values
        out = attn @ v_cache  # (B, T, C)

        return out


class PagedHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        block_size: int,
        attn_pdrop: float = 0.2,
    ):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {embed_dim} and `num_heads`:"
                f" {num_heads})."
            )

        self.head_dim = embed_dim // num_heads

        # Linear projections for key, query, and value
        self.Wk = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.Wq = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, self.head_dim, bias=False)

        self.attn_dropout = nn.Dropout(attn_pdrop)

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.block_table = {i: [] for i in range(batch_size)}
        self.free_blocks = set(range(num_blocks))

        self.kv_cache = torch.zeros(
            (2, num_blocks, block_size, self.head_dim)
        )  # (2, P, S, C)

    def forward(self, x: Tensor):
        _, seq_len, _ = x.shape

        q = self.Wq(x)  # (B,T,C)
        k = self.Wk(x)  # (B,T,C)
        v = self.Wv(x)  # (B,T,C)

        self.slot_mapping(x)
        self.write_kv_cache(k, v)

        k_cache, v_cache = self.fetch_kv_cache()
        k_cache, v_cache = k_cache[:, :seq_len, :], v_cache[:, :seq_len, :]

        # Scaled dot-product attention scores
        scores = q @ k_cache.transpose(-2, -1) * self.head_dim**-0.5  # (B, T, max_k)

        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)) == 0
        scores = scores.masked_fill(mask, float("-inf"))  # (B, T, T)

        # Attention weights and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum of values
        out = attn @ v_cache  # (B, T, C)

        return out

    def slot_mapping(self, x):
        req_lens = (x != 0).any(dim=-1).sum(dim=1)  # (B,)
        for i, req_len in enumerate(req_lens.tolist()):
            rem = int(req_len)
            # try to fill existing last block first
            while rem > 0:
                if len(self.block_table[i]) > 0:
                    last_block_id, last_filled = self.block_table[i][-1]
                    avail = self.block_size - last_filled
                    if avail > 0:
                        take = min(avail, rem)
                        # update the last tuple in-place (replace it)
                        self.block_table[i][-1] = (last_block_id, last_filled + take)
                        rem -= take
                        continue  # maybe still remaining -> try to fill again
                # need a new block
                if not self.free_blocks:
                    raise Exception(
                        "No more free blocks. Implement scheduling and preemption."
                    )
                block_id = self.free_blocks.pop()
                take = min(self.block_size, rem)
                self.block_table[i].append((block_id, take))
                rem -= take

    def write_kv_cache(self, k, v):
        for req in self.block_table:
            for block in self.block_table[req]:
                block_id, num_filled_positions = block
                self.kv_cache[0, block_id, :num_filled_positions, :] = k[
                    req, :num_filled_positions, :
                ]
                self.kv_cache[1, block_id, :num_filled_positions, :] = v[
                    req, :num_filled_positions, :
                ]

    def fetch_kv_cache(self):
        self.max_used_blocks = max([len(x) for x in self.block_table.values()])

        for req in self.block_table:
            kv_block_cache = [block_id for block_id, _ in self.block_table[req]]  # N

            paged_cached_K = self.kv_cache[0, kv_block_cache].view(
                len(kv_block_cache) * self.block_size, self.head_dim
            )
            paged_cached_V = self.kv_cache[1, kv_block_cache].view(
                len(kv_block_cache) * self.block_size, self.head_dim
            )

            pad = torch.empty(
                (
                    (self.max_used_blocks - len(kv_block_cache)) * self.block_size,
                    self.head_dim,
                )
            )
            paged_cached_K = torch.cat((paged_cached_K, pad), dim=0)
            paged_cached_V = torch.cat((paged_cached_V, pad), dim=0)

            if req == 0:
                cached_K = paged_cached_K.unsqueeze(0)
                cached_V = paged_cached_V.unsqueeze(0)
            else:
                cached_K = torch.cat((cached_K, paged_cached_K.unsqueeze(0)), dim=0)
                cached_V = torch.cat((cached_V, paged_cached_V.unsqueeze(0)), dim=0)

        return cached_K, cached_V
