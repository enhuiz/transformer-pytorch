import os
import glob

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence

from torchnmt.datasets.utils import Vocab


class NMTDataset(Dataset):
    def __init__(self, root, split, src, tgt, vocab_share=False, download=True):
        if download:
            if os.path.exists(root):
                print('{} exists.'.format(root))
            else:
                self.download(root)

        samples = self.make_samples(root, 'train', src, tgt)

        if vocab_share:
            self.src_vocab = Vocab([*map(lambda x: x[0], samples),
                                    *map(lambda x: x[1], samples)])
            self.tgt_vocab = self.src_vocab
        else:
            self.src_vocab = Vocab(map(lambda x: x[0], samples))
            self.tgt_vocab = Vocab(map(lambda x: x[1], samples))

        print('src_vocab', self.src_vocab)
        print('tgt_vocab', self.tgt_vocab)

        if split == 'train':
            self.samples = samples
        else:
            self.samples = self.make_samples(root, split, src, tgt)

    def download(self, root):
        raise NotImplementedError()

    def load_file(self, path):
        raise NotImplementedError()

    def make_samples(self, root, split, src, tgt):
        src = self.load_file(os.path.join(root, split + '.' + src))
        tgt = self.load_file(os.path.join(root, split + '.' + tgt))
        return list(zip(src, tgt))

    def __getitem__(self, index):
        src, tgt = self.samples[index]
        src = ['<s>'] + src + ['</s>']
        tgt = ['<s>'] + tgt + ['</s>']
        src = torch.tensor(self.src_vocab.words2idxs(src)).long()
        tgt = torch.tensor(self.tgt_vocab.words2idxs(tgt)).long()
        return {
            'src': src,
            'tgt': tgt,
        }

    def __len__(self):
        return len(self.samples)

    def get_collate_fn(self):
        def collate_fn(batch):
            collated = {}
            # pack first will make the training faster since it is done by multi workers
            collated['src'] = pack_sequence([s['src'] for s in batch],
                                            enforce_sorted=False)
            collated['tgt'] = pack_sequence([s['tgt'] for s in batch],
                                            enforce_sorted=False)
            return collated
        return collate_fn
