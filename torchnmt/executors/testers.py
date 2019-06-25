import os
import tqdm
import numpy as np
import pandas as pd
import glob

import torch

from torchnmt.executors.base import Executor
from torchnmt.utils import unpack_packed_sequence
from torchnmt.scores import compute_scores


class Tester(Executor):
    def __init__(self, opts):
        super().__init__(opts)

    def extract_val_loss(self, path):
        epoch = int(path.split(os.path.sep)[-3])
        with open(path, 'r') as f:
            loss = eval(f.read())['loss']
        return epoch, loss

    def prepare_ckpts(self):
        ckpts = self.saver.get_all_ckpts('epoch')
        if self.opts.mode == 'all':
            ckpts = sorted(ckpts.items())
        elif 'every-' in self.opts.mode:
            n = int(self.opts.mode.split('-')[1])
            ckpts = [(e, ckpts[e])
                     for e in range(n, len(ckpts) + 1, n)]
        elif 'top-' in self.opts.mode:
            n = int(self.opts.mode.split('-')[1])
            paths = glob.glob(os.path.join(
                'results', self.opts.name, '**/val/val.txt'))
            if not paths:
                msg = 'results/{}/**/{}/val.txt not found'.format(
                    self.opts.name)
                raise Exception(msg)
            tops = sorted(map(self.extract_val_loss, paths),
                          key=lambda kv: kv[1])[:n]
            ckpts = [(e, ckpts[e]) for e, _ in tops]
        else:
            raise Exception('Unknown test mode: {}'.format(self.opts.mode))
        return ckpts

    def epoch_iter(self):
        ckpts = self.prepare_ckpts()

        splits = [(split,
                   self.create_data_loader(split))
                  for split in self.opts.splits]

        for epoch, ckpt in ckpts:
            for split, dl in splits:
                print('Testing epoch {}, split: {} ...'.format(epoch, split))
                self.epoch = epoch
                self.split = split
                self.dl = dl
                self.ckpt = ckpt
                yield

    def iteration_iter(self):
        self.pbar = tqdm.tqdm(self.dl, total=len(self.dl))
        for batch in self.pbar:
            self.batch = batch
            yield


class NMTTester(Tester):
    def __init__(self, opts):
        super().__init__(opts)

    def on_epoch_start(self):
        self.out_dir = os.path.join('results',
                                    self.opts.name,
                                    str(self.epoch),
                                    self.split)

        if os.path.exists(os.path.join(self.out_dir, 'refs.txt')):
            print('{} exists, skip.'.format(self.out_dir))
            return self.skip_epoch()

        self.refs = []
        self.hyps = []

        if not hasattr(self, 'model') or self.model.ckpt != self.ckpt:
            self.model = self.create_model(self.ckpt).eval()
            self.model.ckpt = self.ckpt

    def update(self):
        refs = unpack_packed_sequence(self.batch['tgt'])
        with torch.no_grad():
            hyps = self.model(**self.batch, **vars(self.opts))['hyps']
        self.refs += refs
        self.hyps += hyps

    def on_epoch_end(self):
        vocab = self.dl.dataset.tgt_vocab

        refs = [vocab.strip_beos_w(vocab.idxs2words(ref))
                for ref in self.refs]
        hyps = [vocab.strip_beos_w(vocab.idxs2words(hyp))
                for hyp in self.hyps]

        scores = compute_scores(refs, hyps)

        print(scores)

        os.makedirs(self.out_dir, exist_ok=True)
        base = os.path.join(self.out_dir, '{}')

        with open(base.format('scores.txt'), 'w') as f:
            f.write(str(scores))

        refs = list(map(' '.join, refs))
        hyps = list(map(' '.join, hyps))

        with open(base.format('refs.txt'), 'w') as f:
            f.write('\n'.join(refs))

        with open(base.format('hyps.txt'), 'w') as f:
            f.write('\n'.join(hyps))
