import copy
import torch
import torch.nn as nn

from .layers import SublayerWrapper, PositionalEncoding, MultiHeadAttention, FeedForwardLayer
from .utils import clones


class TransformerEncoderLayer(nn.Module):
    def __init__(self, self_attn, ffn, dropout):
        super().__init__()
        self.self_attn = SublayerWrapper(self_attn, dropout)
        self.ffn = ffn

    def forward(self, x, m, src_mask):
        """
        Args:
            x: target input (bs, tgt_len, model_dim)
            m: source memory bank (bs, src_len, model_dim)
            src_mask: (bs, src_len, src_len)
            src_mask: (bs, tgt_len, tgt_len)
        """
        x = self.self_attn(x, x, x, src_mask)
        x = self.ffn(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, layers, heads, input_dim, model_dim, ffn_dim, dropout):
        super().__init__()
        self.embed = nn.Embedding(input_dim, model_dim)
        self.pe = PositionalEncoding(model_dim)
        mha = MultiHeadAttention(heads, model_dim)
        ffn = FeedForwardLayer(model_dim, ffn_dim)
        module = TransformerEncoderLayer(mha, ffn, dropout)
        self.modules = clones(module, layers)

    def forward(self, x, m, src_mask, tgt_mask):
        """
        Args:
            x: target input (bs, tgt_len, tgt_dim)
            m: source memory bank (bs, src_len, model_dim)
            src_mask: (bs, src_len, src_len)
            tgt_mask: (bs, tgt_len, tgt_len)
        """
        x = self.embed(x)
        x = self.pe(x)
        for module in self.modules:
            x = module(x, m, src_mask, tgt_mask)
        return x
