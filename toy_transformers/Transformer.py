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
    return torch.max(torch.zeros(x), x)

def softmax(x: Tensor) -> Tensor:
    """
    Takes Softmax across the last dimension
    """
    # Softmax(x)_i = exp(x_i) / sum(exp(x))
    return x.exp() / torch.sum(x.exp(), dim=-1)

class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, use_bias: bool = False):
        # Use bias default to False for easier SAE reconstruction
        super().__init__()

        self.up = nn.Linear(d_model, d_mlp, bias=use_bias)
        self.down = nn.Linear(d_mlp, d_model, bias=use_bias)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.down(relu(self.up(x)))
        

class CausalAttention(nn.Module):
    def __init__(self, d_model:int, num_heads: int):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.O = nn.Linear(d_model, d_model)

        self.d_head = d_model // num_heads
        self.num_heads = num_heads
    
    def forward(self, x: Tensor) -> Tensor:
        batch, seq, _ = x.shape

        # Equivalent to torch.einsum("batch seq d_model, d_model d_head -> batch seq d_head", x, W) 
        Q = self.Q(x) # [Batch, Seq, d_model = d_head * num_head]
        K = self.K(x) # [Batch, Seq, d_model = d_head * num_head]
        V = self.V(x) # [Batch, Seq, d_model = d_head * num_head]

        # Reshape the last dimension to be d_head * num_head
        Q = Q.reshape(batch, seq, self.num_heads, self.d_head)
        K = K.reshape(batch, seq, self.num_heads, self.d_head)
        V = V.reshape(batch, seq, self.num_heads, self.d_head)
        O = O.reshape(batch, seq, self.num_heads, self.d_head)

        attn_scores = einsum("batch seqQ n_heads d_head, batch seqK n_heads d_head -> batch num_heads seqQ seqK", Q, K) 

        # Softmax (QK.T / sqrt(d)) V
        attn_scores = attn_scores / (self.d_head**0.5)
        pattern = softmax(attn_scores)

        z = einsum("batch seqK n_heads d_head, batch num_heads seqQ seqK -> batch num_heads seqQ", V, pattern)
        return self.O(z)

class LayerNorm(nn.Module):
    pass

class TransformerBlock(nn.Module):
    pass

class Transformer(nn.Module):
    pass