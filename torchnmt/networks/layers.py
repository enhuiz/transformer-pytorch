import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: query (*, query_len, dim)
            k: key (*, key_len, dim)
            v: value (*, key_len, dim)
            mask: (*, query_len, key_len), 0 will be masked out
        Returns:
            context: (*, query_len, dim)
            weights: (*, query_len, key_len)
        """
        dim = q.shape[-1]

        # scores: (batch, query_len, key_len)
        scores = q @ k.transpose(-2, -1) / (dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -np.inf)
        weights = F.softmax(scores, dim=-1)
        context = weights @ v

        return context, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, dim, attention=ScaledDotProductAttention()):
        """
        Args:
            dim: model's dim
            attention: attention layer takes the same input as self.forward
        """
        super().__init__()

        if dim % heads != 0:
            raise ValueError("MHA dim should be a multiple of heads, \
                but got {} and {}".format(dim, heads))

        self.heads = heads
        self.dim = dim
        self.attention = attention

        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.attention = attention
        self.fc = nn.Linear(dim, dim)

    def forward(self, q, k, v, mask):
        """
        Args:
            q: query (batch, query_len, dim)
            k: key (batch, key_len, dim)
            v: value (batch, query_len, dim)
            mask: (batch, query_len, key_len)
        Returns:
            context: (batch, query_len, dim)
            weights: (batch, heads, query_len, key_len)
        """
        bs, ql = q.shape[:2]

        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        q = q.view(bs, -1, self.heads, self.dim // self.heads)
        k = k.view(bs, -1, self.heads, self.dim // self.heads)
        v = v.view(bs, -1, self.heads, self.dim // self.heads)

        # swap len and head
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # add head dim for mask
        if mask is not None:
            mask = mask.unsqueeze(1)

        context, self.weights = self.attention(q, k, v, mask)

        # swap len and head back
        context = context.transpose(1, 2).contiguous()
        context = context.view(bs, ql, self.dim)
        context = self.fc(context)

        return context


class FeedForwardLayer(nn.Module):
    def __init__(self, model_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, model_dim)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()

        pos = np.arange(0, max_len)[:, None]
        i = np.arange(0, dim // 2)
        denom = 10000 ** (2 * i / dim)

        pe = np.zeros([max_len, dim])
        pe[:, 0::2] = np.sin(pos / denom)
        pe[:, 1::2] = np.cos(pos / denom)
        pe = torch.from_numpy(pe).float()

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1]]


class SublayerWrapper(nn.Module):
    def __init__(self, dim, sublayer, dropout):
        super().__init__()
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args):
        return self.norm(x + self.dropout(self.sublayer(x, *args)))
