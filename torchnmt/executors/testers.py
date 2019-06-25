import os
import tqdm
import numpy as np
import pandas as pd

import torch

from torchnmt.executors.base import Executor
from torchnmt.utils import unpack_packed_sequence
from torchnmt.scores import compute_scores


class Tester(Executor):
    def __init__(self, opts):
        super().__init__(opts)

    def start(self):
        self.epoch_iter = self.create_epoch_iter()
        super().start()

    def create_epoch_iter(self):
        if self.opts.all:
            ckpts = self.saver.get_all_ckpts('epoch')
        else:
            ckpts = [self.saver.get_latest_ckpt('epoch')]

        ckpts = [(self.saver.parse_step(ckpt), ckpt)
                 for ckpt in ckpts]

        splits = [(split,
                   self.create_data_loader(split),
                   self.create_writer(split + '*'))
                  for split in self.opts.splits]

        for epoch, ckpt in ckpts:
            model = None
            for split, dl, writer in splits:
                outdir = os.path.join('results',
                                      self.opts.name,
                                      str(epoch),
                                      split)
                if not os.path.exists(outdir):
                    print('Evaluating {} ...'.format(outdir))
                    model = model or self.create_model(ckpt)
                    self.epoch = epoch
                    self.outdir = outdir
                    self.dl = dl
                    self.model = model.eval()
                    self.writer = writer
                    yield

    def done(self):
        try:
            next(self.epoch_iter)
            return False
        except:
            return True


class NMTTester(Tester):
    def __init__(self, opts):
        super().__init__(opts)

    def on_epoch_start(self):
        self.refs = []
        self.hyps = []
        self.losses = []

    def update(self, batch):
        refs = unpack_packed_sequence(batch['tgt'])
        with torch.no_grad():
            out = self.model(**batch, **vars(self.opts))
        hyps = out['hyps']
        self.refs += refs
        self.hyps += hyps

        if 'loss' in out:
            loss = out['loss']
            self.losses.append(loss.item())

    def on_epoch_end(self):
        vocab = self.dl.dataset.tgt_vocab

        refs = [vocab.strip_beos_w(vocab.idxs2words(ref))
                for ref in self.refs]
        hyps = [vocab.strip_beos_w(vocab.idxs2words(hyp))
                for hyp in self.hyps]

        scores = compute_scores(refs, hyps)

        if self.losses:
            loss = np.mean(self.losses)
            ppl = np.exp(loss)
            scores['loss'] = loss
            scores['ppl'] = ppl
            self.writer.add_scalar('loss', loss, self.epoch)
            self.writer.add_scalar('ppl', ppl, self.epoch)

        print(scores)

        os.makedirs(self.outdir, exist_ok=True)
        base = os.path.join(self.outdir, '{}')

        with open(base.format('scores.txt'), 'w') as f:
            f.write(str(scores))

        refs = list(map(' '.join, refs))
        hyps = list(map(' '.join, hyps))

        df = pd.DataFrame({'refs': refs, 'hyps': hyps})

        df['refs'].to_csv(base.format('refs.txt'),
                          header=False, index=False)

        df['hyps'].to_csv(base.format('hyps.txt'),
                          header=False, index=False)
