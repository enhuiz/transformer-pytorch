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
            mem: (bs, src_len, input_dim)
        Outputs:
            tgt_output: [(tgt_len, )] 
        """
        bos = Vocab.extra2idx('<s>')
        eos = Vocab.extra2idx('</s>')

        running = torch.full((len(mem), 1), bos).long().to(self.device)
        done = []

        for _ in range(max_len):
            outputs = self.decoder(running, mem, mem_mask, None)
            outputs = outputs[:, -1].argmax(dim=-1)  # (bs,)
            running = torch.cat([running, outputs[:, None]], dim=-1)
            running_idx = (outputs != eos).nonzero().squeeze(1)
            finished_idx = (outputs == eos).nonzero().squeeze(1)
            done += running[finished_idx].tolist()
            running = running[running_idx]
            mem = mem[running_idx]
            mem_mask = mem_mask[running_idx]

            if len(running) == 0:
                break

        done += running.tolist()
        return done

    def beam_search_helper(self, memory, mem_mask, max_len, beam_width):
        """
        Args:
            videos: unpadded videos, [(l, c, h, w)]
        """
        bos = Vocab.extra2idx('<s>')
        eos = Vocab.extra2idx('</s>')

        running = [(0, torch.LongTensor([bos]))]
        done = []

        memory = memory.expand(beam_width, *memory.shape[1:])
        mem_mask = mem_mask.expand(beam_width, *mem_mask.shape[1:])

        for _ in range(max_len):
            if beam_width <= 0:
                break

            logps, hyps = zip(*running)
            hyps = torch.stack(hyps).to(self.device)
            outputs = self.decoder(hyps,
                                   memory[:len(running)],
                                   mem_mask[:len(running)],
                                   None)

            outputs = outputs[:, -1]
            outputs = torch.log_softmax(outputs, dim=-1)

            # for each beam, calculate top k
            sub_logps, sub_indexes = torch.topk(outputs, beam_width)

            # calculate accumulated logps
            for i in range(len(logps)):
                sub_logps[i] += logps[i]

            # calculate new top k
            sub_logps = sub_logps.view(-1)
            sub_indexes = sub_indexes.view(-1)

            logps, indexes = torch.topk(sub_logps, beam_width)

            word_indexes = sub_indexes[indexes]
            captions_indexes = indexes // len(logps)

            hyps = [hyps[captions_indexes[i]].tolist() + word_indexes[i:i + 1].tolist()
                    for i in range(beam_width)]

            hyps = [torch.LongTensor(hyp) for hyp in hyps]

            running = []
            for logp, hyp in zip(logps, hyps):
                if hyp[-1] == eos:
                    done += [(logp, hyp)]
                    beam_width -= 1
                    memory = memory[:beam_width]
                    mem_mask = mem_mask[:beam_width]
                else:
                    running += [(logp, hyp)]

        done = done + running
        hyp = max(done, key=lambda t: t[0])[1]

        return hyp

    def beam_search_inference(self, mem, mem_mask, max_len, beam_width):
        return [self.beam_search_helper(mem[i:i+1],
                                        mem_mask[i:i+1],
                                        max_len,
                                        beam_width)
                for i in range(len(mem))]
