import torch
from torch import nn, Tensor
from torch.nn import functional as F


class KVCache:
    """
    Key/Value (KV) cache for transformer attention layers.

    In a naïve implementation of autoregressive text generation every forward
    pass recomputes the attention keys and values for the entire prefix, which
    causes the compute cost to grow quadratically with the sequence length.

    With KV cache instead of recomputing keys and values at every step, the model
    stores (caches) the key and value projections from previous tokens.
    Subsequent steps only compute keys/values for the *new* tokens and append
    them to the cache. During attention, the model can immediately reuse the
    cached tensors, reducing complexity from O(n²) per step to O(n) and enabling
    fast generation.

    Following produciton implementtaion principles this class implements
    pre-allocates a tensor of shape (2, batch_size, num_heads, max_tokens, head_dim)
    and returns the *sliced* view of cached keys/values up to thecurrent length,
    ready to be fed into attention computations.

    """

    def __init__(self, batch_size, max_tokens, num_heads, head_dim):
        self.kv_cache = torch.empty(
            2,
            batch_size,
            max_tokens,
            num_heads,
            head_dim,
        )
        self.max_tokens = max_tokens
        self.cumulative_length = 0

    def update(self, k, v):
        start = int(self.cumulative_length)
        end = start + k.size(-3)
        if end > self.max_tokens:
            raise ValueError("KVCache overflow: increase max_tokens")

        self.kv_cache[0, :, start:end, :, :] = k
        self.kv_cache[1, :, start:end, :, :] = v

        self.cumulative_length += k.size(-3)

        return self.kv_cache[0, :, : self.cumulative_length, :, :], self.kv_cache[
            1, :, : self.cumulative_length, :, :
        ]

    def reset(self):
        """Clear the cache (start a new sequence)."""
        self.cumulative_length = 0


"""
Efficient Memory Management for Large Language
Model Serving with PagedAttention

High throughput serving of large language models (LLMs)
requires batching sufficiently many requests at a time. 
However, existing systems struggle because the key-value cache
(KV cache) memory for each request is huge and grows
and shrinks dynamically. When managed inefficiently, this
memory can be significantly wasted by fragmentation and
redundant duplication, limiting the batch size. To address
this problem, we propose PagedAttention, an attention algorithm 
inspired by the classical virtual memory and paging 
techniques in operating systems. On top of it, we build
vLLM, an LLM serving system that achieves (1) near-zero
waste in KV cache memory and (2) flexible sharing of KV
cache within and across requests to further reduce memory 
usage. Our evaluations show that vLLM improves the
throughput of popular LLMs by 2-4× with the same level
of latency compared to the state-of-the-art systems, such
as FasterTransformer and Orca. The improvement is more
pronounced with longer sequences, larger models, and more
complex decoding algorithms. vLLM’s source code is publicly
available at https://github.com/vllm-project/vllm.

https://arxiv.org/pdf/2309.06180
"""


class PagedKVCache:
    """
    Block-based ("paged") Key/Value (KV) cache for transformer attention layers.

    A standard KV cache usually stores all keys and values for a sequence in a
    single contiguous buffer per batch item. This implies an significant waste of
    memory.

    Paged KV cache  divides memory into fixed-size blocks (pages). Each sequence
    is represented as a list of blocks, and tokens are appended into these blocks
    as needed. This frees unused memory for incoming requdsts increasing server
    throughput.

    Serving as proof of concept, this class demonstrates the core mechanics behind
    paged KV caching: a global pool of fixed-size blocks, per-sequence block tables that
    describe how those blocks are stitched together logically, and a gather
    step that materializes dense K/V tensors for attention when needed.
    """

    def __init__(self, num_blocks, block_size, num_heads, head_dim):
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_table = None  # {i: [] for i in range(B)}
        self.free_blocks = set(range(num_blocks))
        self.kv_cache = torch.empty(
            (2, num_blocks, block_size, num_heads, head_dim)
        )  # (2, nB, B, nH, H)

    def update(self, k, v):
        # allocate in blocks
        batch_size, seq_len, _, _ = k.size()
        if self.block_table is None:
            self.block_table = {i: [] for i in range(batch_size)}

        for b in range(batch_size):
            t = 0
            while t < k.size(1):
                # If there's a last block with free space, fill that first
                if self.block_table[b] and self.block_table[b][-1][1] < self.block_size:
                    block_id, filled = self.block_table[b][-1]
                else:
                    # Need a new block
                    if not self.free_blocks:
                        raise RuntimeError(
                            "No more free blocks. Implement eviction/preemption here."
                        )
                    block_id = self.free_blocks.pop()
                    filled = 0
                    self.block_table[b].append([block_id, 0])

                take = min(self.block_size - filled, k.size(1) - t)
                self.kv_cache[0, block_id, filled:filled+take, :, :] = k.view(batch_size, seq_len, self.num_heads, self.head_dim)[b, t:t+take, :, :]
                self.kv_cache[1, block_id, filled:filled+take, :, :] = v.view(batch_size, seq_len, self.num_heads, self.head_dim)[b, t:t+take, :, :]

                # Update filled count in block_table
                self.block_table[b][-1][1] = filled + take

                t += take

        # Fetch dense K/V for each sequence
        max_len = max([sum(f for _, f in x) for x in self.block_table.values()])

        k_full = k.new_zeros(batch_size, max_len, self.num_heads, self.head_dim)
        v_full = v.new_zeros(batch_size, max_len, self.num_heads, self.head_dim)

        for b in range(batch_size):
            cur = 0
            for block_id, filled in self.block_table[b]:
                assert filled

                # kv_cache: (2, num_blocks, block_size, nH, H)
                # slice: (filled, nH, H) -> permute to (nH, filled, H)
                k_block = self.kv_cache[0, block_id, :filled]  # (filled, nH, H)
                v_block = self.kv_cache[1, block_id, :filled]  # (filled, nH, H)

                k_full[b, cur:cur+filled,:, :] = k_block
                v_full[b,  cur:cur+filled, :, :] = v_block
                cur += filled

        return k_full, v_full
