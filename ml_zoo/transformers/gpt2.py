"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) Karpathy's nanoGPT implementation:
https://github.com/karpathy/nanoGPT/blob/master/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import torch
from torch import nn
from torch.nn import functional as F

from ml_zoo.transformers.attn import HeadAttention


class GPT2(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        vocab_size,
        max_position_embeddings,
        num_layers,
    ):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings

        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(max_position_embeddings, embed_dim)
        self.blocks = nn.Sequential(
            *[GPT2Block(embed_dim, num_heads) for _ in range(num_layers)]
        )

        self.layernorm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        batch_size, sequence_length = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(sequence_length, device=idx.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.layernorm(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            _, _, vocab_size = logits.shape
            logits = logits.view(batch_size * sequence_length, vocab_size)
            targets = targets.view(batch_size * sequence_length)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last allowed token
            idx_cond = idx[:, -self.max_position_embeddings :]
            logits, _ = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class GPT2Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.attn_layer = GPT2MultiHeadAttention(embed_dim, num_heads)
        self.mlp_layer = GPT2MLP(embed_dim)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn_layer(self.layernorm1(x))
        x = x + self.mlp_layer(self.layernorm2(x))

        return x


class GPT2MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, embed_dim, num_heads, attn_pdrop=0.2):
        super().__init__()

        if embed_dim % num_heads:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {embed_dim} and `num_heads`:"
                f" {num_heads})."
            )

        self.heads = nn.ModuleList(
            [HeadAttention(embed_dim, num_heads, attn_pdrop) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(attn_pdrop)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.attn_dropout(self.proj(out))
        return out
    

class GPT2MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, embed_dim, num_heads, attn_pdrop=0.2):
        super().__init__()

        if embed_dim % num_heads:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {embed_dim} and `num_heads`:"
                f" {num_heads})."
            )

        self.heads = nn.ModuleList(
            [KVCacheHeadAttention(embed_dim, num_heads, attn_pdrop) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(attn_pdrop)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.attn_dropout(self.proj(out))
        return out


class GPT2MLP(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )  # see https://arxiv.org/pdf/1706.03762#page=5

    def forward(self, x):
        return self.net(x)
