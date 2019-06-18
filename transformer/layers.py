import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout, inplace=True)

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
        weights = self.dropout(weights)
        context = weights @ v

        return context, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, dim, attention):
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
        mask = mask.unsqueeze(1)

        context, weights = self.attention(q, k, v, mask)

        # swap len and head back
        context = context.transpose(1, 2).contiguous()
        context = context.view(bs, ql, self.dim)
        context = self.fc(context)

        return context, weights
