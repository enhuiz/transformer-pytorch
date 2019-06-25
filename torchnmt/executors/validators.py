import os
import tqdm
import numpy as np
import pandas as pd

import torch

from torchnmt.executors.base import Executor
from torchnmt.utils import unpack_packed_sequence
from torchnmt.scores import compute_scores


class Validator(Executor):
    def __init__(self, opts):
        super().__init__(opts)

    def epoch_iter(self):
        ckpts = sorted(self.saver.get_all_ckpts('epoch').items())

        splits = [(split,
                   self.create_data_loader(split),
                   self.create_writer(split + '(testing)'))
                  for split in self.opts.splits]

        for epoch, ckpt in ckpts:
            for split, dl, writer in splits:
                print('Validating epoch {}, split: {} ...'.format(epoch, split))
                self.epoch = epoch
                self.split = split
                self.dl = dl
                self.writer = writer
                self.ckpt = ckpt
                yield

    def iteration_iter(self):
        self.pbar = tqdm.tqdm(self.dl, total=len(self.dl))
        for batch in self.pbar:
            self.batch = batch
            yield


class NMTValidator(Validator):
    def __init__(self, opts):
        super().__init__(opts)

    def on_epoch_start(self):
        self.out_path = os.path.join('results',
                                     self.opts.name,
                                     str(self.epoch),
                                     self.split,
                                     'val.txt')

        if os.path.exists(self.out_path):
            return self.skip_epoch()

        if not hasattr(self, 'model') or self.model.ckpt != self.ckpt:
            self.model = self.create_model(self.ckpt).train()
            self.model.ckpt = self.ckpt

        self.losses = []

    def update(self):
        with torch.no_grad():
            loss = self.model(**self.batch, **vars(self.opts))['loss']
        self.losses.append(loss.item())

    def on_epoch_end(self):
        loss = np.mean(self.losses)
        ppl = np.exp(loss)

        print('Epoch {}'.format(self.epoch))
        print('{}:\tloss: {:.4g}, ppl: {:.4g}'.format(self.split, loss, ppl))

        self.writer.add_scalar('loss', loss, self.epoch)
        self.writer.add_scalar('ppl', ppl, self.epoch)
        self.writer.flush()  # requires: https://github.com/lanpa/tensorboardX/pull/451

        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

        with open(self.out_path, 'w') as f:
            f.write(str({
                'loss': loss,
                'ppl': ppl,
            }))
