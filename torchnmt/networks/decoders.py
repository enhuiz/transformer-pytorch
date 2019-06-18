import copy
import torch
import torch.nn as nn

from .layers import SublayerWrapper, PositionalEncoding, MultiHeadAttention, FeedForwardLayer
from .utils import clones


class TransformerDecoderLayer(nn.Module):
    def __init__(self, self_attn, src_attn, ffn, dropout):
        super().__init__()
        self.self_attn = SublayerWrapper(self_attn, dropout)
        self.src_attn = SublayerWrapper(src_attn, dropout)
        self.ffn = ffn

    def forward(self, x, m, mem_mask, tgt_mask):
        """
        Args:
            x: target input (bs, tgt_len, model_dim)
            m: source memory bank (bs, src_len, model_dim)
            mem_mask: (bs, tgt_len, src_len)
            tgt_mask: (bs, tgt_len, tgt_len)
        """
        x = self.self_attn(x, x, x, tgt_mask)
        if m is not None:
            x = self.src_attn(x, m, m, mem_mask)
        x = self.ffn(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, layers, heads, output_dim, model_dim, ffn_dim, dropout):
        super().__init__()
        self.embed = nn.Embedding(output_dim, model_dim)
        self.pe = PositionalEncoding(model_dim)
        c = copy.deepcopy
        mha = MultiHeadAttention(heads, model_dim)
        ffn = FeedForwardLayer(model_dim, ffn_dim)
        module = TransformerDecoderLayer(mha, c(mha), ffn, dropout)
        self.modules = clones(module, layers)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, x, m, mem_mask, tgt_mask):
        """
        Args:
            x: target input (bs, tgt_len, tgt_dim)
            m: source memory bank (bs, src_len, model_dim)
            mem_mask: (bs, tgt_len, src_len)
            tgt_mask: (bs, tgt_len, tgt_len)
        """
        x = self.embed(x)
        x = self.pe(x)
        for module in self.modules:
            x = module(x, m, mem_mask, tgt_mask)
        x = self.fc(x)
        return x
