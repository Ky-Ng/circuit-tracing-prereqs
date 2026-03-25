# Generic Transformer Class written by hand for practice
from torch import nn
from torch import Tensor
import torch
from einops import einsum
"""
0. Embed/Unembed
1. MLP
2. Attention

To implement
3. Residuals
4. Layer Norm
"""


def relu(x: Tensor) -> Tensor:
    # return torch.clamp(x, min=0)
    return torch.max(torch.zeros_like(x), x)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, use_bias: bool = False):
        # Use bias default to False for easier SAE reconstruction
        super().__init__()

        self.up = nn.Linear(d_model, d_mlp, bias=use_bias)
        self.down = nn.Linear(d_mlp, d_model, bias=use_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(relu(self.up(x)))


class CausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_context_len: int):
        assert d_model % n_heads == 0, f"d_model of {d_model} must be divisible by n_heads {n_heads}"
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Note, we have no biases in this simple Attention Module
        self.W_Q = nn.Parameter(torch.empty([n_heads, d_model, self.d_head]))
        self.W_K = nn.Parameter(torch.empty([n_heads, d_model, self.d_head]))
        self.W_V = nn.Parameter(torch.empty([n_heads, d_model, self.d_head]))
        self.W_O = nn.Parameter(torch.empty(
            [n_heads, self.d_head, self.d_model]))

        # Initialize, 0.02 is god-given's number from OpenAI
        nn.init.normal_(self.W_Q, mean=0, std=0.02)
        nn.init.normal_(self.W_K, mean=0, std=0.02)
        nn.init.normal_(self.W_V, mean=0, std=0.02)
        nn.init.normal_(self.W_O, mean=0, std=0.02)

        # Attention Mask
        self.max_context_len = max_context_len
        self.register_buffer(
            "MASK",
            torch.triu(torch.ones(
                [self.max_context_len, self.max_context_len], dtype=torch.bool), diagonal=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Follow notation from Transformer Lens
        """
        batch, seq, d_model = x.shape

        # QKV Projections
        q = einsum(
            x, self.W_Q, "batch seq d_model, n_heads d_model d_head -> batch n_heads seq d_head")
        k = einsum(
            x, self.W_K, "batch seq d_model, n_heads d_model d_head -> batch n_heads seq d_head")
        v = einsum(
            x, self.W_V, "batch seq d_model, n_heads d_model d_head -> batch n_heads seq d_head")

        # Attention Pattern
        attn_scores = einsum(
            q, k, "batch n_heads seqQ d_head, batch n_heads seqK d_head -> batch n_heads seqQ seqK")

        # Mask for causal attention
        attn_scores = attn_scores / (self.d_head ** 0.5)
        attn_scores = attn_scores.masked_fill(
            self.MASK[:seq, :seq], -torch.inf)

        # Attention Scores
        pattern = attn_scores.softmax(dim=-1)

        # V Projections
        z = einsum(
            pattern, v, "batch n_heads seqQ seqK, batch n_heads seqK d_head -> batch n_heads seqQ d_head")

        # O Projections and sum over heads
        attn_out = einsum(
            z, self.W_O, "batch n_heads seq d_head, n_heads d_head d_model -> batch seq d_model")
        return attn_out


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)
        mean_centered_variance_one = (x - mean) / ((var + self.eps) ** 0.5)
        return mean_centered_variance_one * self.gamma + self.beta


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int,  n_heads: int, max_context_len: int):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attention = CausalAttention(d_model, n_heads, max_context_len)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp=d_model*4)

    def forward(self, x: Tensor) -> Tensor:
        """
        Note: use GPT-2 style pre-layer norm: x + Sublayer(LayerNorm(x))
        """
        resid_pre = x
        resid_mid = resid_pre + self.attention(self.ln1(resid_pre))
        resid_post = resid_mid + self.mlp(self.ln2(resid_mid))
        return resid_post


class Transformer(nn.Module):
    def __init__(self, n_layers, d_model: int, n_heads: int, max_context_len: int, vocab_size: int):
        super().__init__()

        # Configs
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_context_len = max_context_len
        self.vocab_size = vocab_size

        # Architecture
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_context_len, d_model)

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, max_context_len)
             for _ in range(n_layers)]
        )
        self.final_ln = LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        batch, seq = tokens.shape

        positions = torch.arange(seq, device=tokens.device)
        embedded = self.embed(tokens) + self.pos_embed(positions)

        out = embedded 
        for layer in self.layers:
            out = layer(out)
        
        out = self.final_ln(out)
        unembedded = self.unembed(out)
        return unembedded