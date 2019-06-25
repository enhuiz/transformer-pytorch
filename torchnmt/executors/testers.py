import os
import tqdm
import numpy as np
import pandas as pd

import torch

from torchnmt.executors.base import Executor
from torchnmt.utils import unpack_packed_sequence
from torchnmt.scores import compute_scores


class Tester(Executor):
    def __init__(self, name, model, dataset, opts):
        super().__init__(name, model, dataset, opts)

    def start(self):
        if self.opts.all:
            ckpts = self.saver.get_all_ckpts('epoch')
        else:
            ckpts = [self.saver.get_latest_ckpt('epoch')]

        dls = {sp: self.create_data_loader(sp) for sp in self.opts.splits}

        for ckpt in ckpts:
            epoch = self.saver.parse_step(ckpt)
            model = None  # lazy loading
            for split in self.opts.splits:
                folder = os.path.join('results', self.name, str(epoch), split)
                if not os.path.exists(folder):
                    model = model or self.create_model(ckpt)
                    os.makedirs(folder, exist_ok=True)
                    self.evaluate(model, dls[split], folder)

    def evaluate(self, model, dl, folder):
        raise NotImplementedError()


class NMTTester(Tester):
    def __init__(self, name, model, dataset, opts):
        super().__init__(name, model, dataset, opts)

    def evaluate(self, model, dl, folder):
        print('Evaluating {} ...'.format(folder))
        model = model.eval()
        vocab = dl.dataset.tgt_vocab

        refs, hyps = [], []
        losses = []
        for batch in tqdm.tqdm(dl, total=len(dl)):
            batch_refs = unpack_packed_sequence(batch['tgt'])
            with torch.no_grad():
                out = model(**batch, **vars(self.opts))
                batch_hyps = out['hyps']
                loss = out['loss']
                losses.append(loss.item())

            refs += [vocab.strip_beos_w(vocab.idxs2words(ref))
                     for ref in batch_refs]
            hyps += [vocab.strip_beos_w(vocab.idxs2words(hyp))
                     for hyp in batch_hyps]

        refs = list(map(' '.join, refs))
        hyps = list(map(' '.join, hyps))

        df = pd.DataFrame({'refs': refs, 'hyps': hyps})

        pathbase = os.path.join(folder, '{}')

        scores = compute_scores(refs, hyps)
        scores['loss'] = np.mean(losses)
        scores['ppl'] = np.exp(scores['loss'])

        with open(pathbase.format('scores.txt'), 'w') as f:
            f.write(str(scores))

        print(scores)

        df['refs'].to_csv(pathbase.format('refs.txt'),
                          header=False, index=False)

        df['hyps'].to_csv(pathbase.format('hyps.txt'),
                          header=False, index=False)
