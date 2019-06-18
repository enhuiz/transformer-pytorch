import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .encoders import TransformerEncoder
from .decoders import TransformerDecoder

from torchnmt import networks


class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        assert encoder.model_dim == decoder.model_dim

        self.model_dim = encoder.model_dim

        self.encoder = networks.get(encoder)
        self.decoder = networks.get(decoder)

        self.register_buffer('_device', torch.zeros(0))

    @property
    def device(self):
        return self._device.device

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Args:
            src: (bs, src_len, input_dim)
            tgt: (bs, tgt_len, output_dim)
        """
        assert len(src) == len(tgt)  # same batch size

        mem = self.encoder(src, src_mask)
        return self.decoder(tgt, mem, src_mask, tgt_mask)

    def padding_mask(self, lens):
        """Mask out the blank (padding) values
        Args:
            lens: (bs,)
        Return:
            mask: (bs, 1, maxlen)
        """
        bs, maxlen = len(lens), max(lens)
        mask = torch.zeros(bs, 1, maxlen).byte()
        for i, l in enumerate(lens):
            mask[i, :, :l] = 1
        mask = mask.to(self.device)
        return mask

    def subsequent_mask(self, lens):
        """Mask out future word
        Args:
            lens: (bs,)
        Return:
            mask: (bs, maxlen, maxlen)
        """
        bs, maxlen = len(lens), max(lens)
        mask = torch.ones([bs, maxlen, maxlen]).tril_(0)
        mask = mask.to(self.device)
        return mask

    def greedy_inference(self, src, src_mask, vocab, max_len):
        """
        Args:
            src: (bs, src_len, input_dim)
        Outputs:
            tgt_output: [(tgt_len, )] 
        """
        mem = self.encoder(src, src_mask)

        running = torch.ones(len(src), 1) * vocab.bosi
        running = running.long().to(self.device)
        done = []

        for _ in range(max_len):
            outputs = self.decoder(running, mem, src_mask, None)
            outputs = outputs[:, -1].argmax(dim=-1)  # (bs,)
            running = torch.cat(running, outputs[:, None], dim=-1)
            running_idx = (outputs != vocab.eosi).nonzero()
            finished_idx = (outputs == vocab.eosi).nonzero()
            done += running[finished_idx].tolist()
            running = running[running_idx]

        return done
