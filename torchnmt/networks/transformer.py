import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

from .encoders import TransformerEncoder
from .decoders import TransformerDecoder

from torchnmt import networks
from torchnmt.datasets.utils import Vocab


class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        assert encoder.model_dim == decoder.model_dim
        self.encoder = networks.get(encoder)
        self.decoder = networks.get(decoder)
        self.register_buffer('_device', torch.zeros(0))

    @property
    def device(self):
        return self._device.device

    def forward(self, src, tgt=None, max_len=None, beam_width=None, **_):
        """
        Args:
            src: packed sequence (*, input_dim)
            tgt: packed sequence (*, output_dim)
        """
        pad = Vocab.extra2idx('<pad>')
        src, src_len = pad_packed_sequence(src, True, pad)
        src = src.to(self.device)
        src_mask = self.padding_mask(src_len)
        mem = self.encoder(src, src_mask)

        if self.training:
            tgt, tgt_len = pad_packed_sequence(tgt, True, pad)
            tgt = tgt.to(self.device)
            tgt_mask = self.padding_mask(tgt_len) * \
                self.subsequent_mask(tgt_len)

            outputs = self.decoder(tgt, mem, src_mask, tgt_mask)
            logp = F.log_softmax(outputs, dim=-1)

            chopped_outputs = outputs[:, :-1].reshape(-1, outputs.shape[-1])
            shifted_targets = tgt[:, 1:].reshape(-1)
            loss = F.cross_entropy(chopped_outputs, shifted_targets)

            return {
                'logp': logp,
                'loss': loss
            }
        else:
            # time step
            # inputs: hyps (bs, len), mem, mem_mask
            if beam_width == 1:
                hyps = self.greedy_inference(mem, src_mask, max_len)
            else:
                hyps = self.beam_search_inference(
                    mem, src_mask, max_len, beam_width)

            return {
                'hyps': hyps,
            }

    def padding_mask(self, lens):
        """Mask out the blank (padding) values
        Args:
            lens: (bs,)
        Return:
            mask: (bs, 1, max_len)
        """
        bs, max_len = len(lens), max(lens)
        mask = torch.zeros(bs, 1, max_len)
        for i, l in enumerate(lens):
            mask[i, :, :l] = 1
        mask = mask.to(self.device)
        return mask

    def subsequent_mask(self, lens):
        """Mask out future word
        Args:
            lens: (bs,)
        Return:
            mask: (bs, max_len, max_len)
        """
        bs, max_len = len(lens), max(lens)
        mask = torch.ones([bs, max_len, max_len]).tril_(0)
        mask = mask.to(self.device)
        return mask

    def greedy_inference(self, mem, mem_mask, max_len):
        """
        Args:
            mem: (bs, src_len, model_dim)
        Outputs:
            tgt_output: [(tgt_len,)]
        """
        bos = Vocab.extra2idx('<s>')
        eos = Vocab.extra2idx('</s>')

        batch_idx = torch.arange(len(mem))
        running = torch.full((len(mem), 1), bos).long().to(self.device)
        finished = []

        for _ in range(max_len):
            outputs = self.decoder(running, mem, mem_mask, None)
            outputs = outputs[:, -1].argmax(dim=-1)  # (bs,)
            running = torch.cat([running, outputs[:, None]], dim=-1)

            running_idx = (outputs != eos).nonzero().squeeze(1)
            finished_idx = (outputs == eos).nonzero().squeeze(1)

            finished += list(zip(batch_idx[finished_idx],
                                 running[finished_idx].tolist()))

            running = running[running_idx]
            batch_idx = batch_idx[running_idx]
            mem = mem[running_idx]
            mem_mask = mem_mask[running_idx]

            if len(running) == 0:
                break

        finished += list(zip(batch_idx, running.tolist()))
        finished = [x[1] for x in sorted(finished, key=lambda x: x[0])]

        return finished

    def beam_search_helper(self, memory, mem_mask, max_len, beam_width):
        """
        Args:
            mem: (bs, src_len, model_dim)
        Outputs:
            tgt_output: (tgt_len,)
        """
        bos = Vocab.extra2idx('<s>')
        eos = Vocab.extra2idx('</s>')

        logps = torch.tensor([0.0]).to(self.device)
        hyps = torch.tensor([[bos]]).long().to(self.device)

        finished = []

        memory = memory.expand(beam_width, *memory.shape[1:])
        mem_mask = mem_mask.expand(beam_width, *mem_mask.shape[1:])

        for _ in range(1, max_len):
            if len(logps) <= 0:
                break

            outputs = self.decoder(hyps,
                                   memory[:len(logps)],
                                   mem_mask[:len(logps)])

            outputs = torch.log_softmax(outputs[:, -1], dim=-1)

            # for each beam, calculate top k
            tmp_logps, tmp_idxs = torch.topk(outputs, beam_width)

            # calculate accumulated logps
            tmp_logps += logps[:, None]

            # calculate new top k
            tmp_logps = tmp_logps.view(-1)
            tmp_idxs = tmp_idxs.view(-1)

            logps, idxs = torch.topk(tmp_logps, beam_width)

            words = tmp_idxs[idxs]
            hyps_idxs = idxs // len(logps)

            hyps = torch.cat([hyps[hyps_idxs], words[:, None]], dim=1)

            finished_idx = (words == eos).nonzero().squeeze(1)
            running_idx = (words != eos).nonzero().squeeze(1)

            finished += list(zip(logps[finished_idx], hyps[finished_idx]))

            logps = logps[running_idx]
            hyps = hyps[running_idx]

        finished = finished + list(zip(logps, hyps))

        hyp = max(finished, key=lambda t: t[0])[1]

        return hyp

    def beam_search_inference(self, mem, mem_mask, max_len, beam_width):
        return [self.beam_search_helper(mem[i:i + 1],
                                        mem_mask[i:i + 1],
                                        max_len,
                                        beam_width)
                for i in range(len(mem))]
