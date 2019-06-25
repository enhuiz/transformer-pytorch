import os
import glob
import urllib

import pandas as pd
import torch
from torch.nn.utils.rnn import pack_sequence

from torchnmt.utils import working_directory
from torchnmt.datasets.base import NMTDataset


class XiaoshiDataset(NMTDataset):
    def __init__(self, root, split, src, tgt, vocab_share=False, download=True):
        super().__init__(root, split, src, tgt, vocab_share, download)

    def download(self, root):
        os.makedirs(root, exist_ok=True)
        print('Downloading dataset.')

        url = 'https://raw.githubusercontent.com/enhuiz/XiaoShi/master/input/raw/raw.csv'

        with working_directory(root):
            urllib.request.urlretrieve(url, 'raw.csv')

            def split(df, frac):
                n = int(len(df) * frac)
                h = df.head(n)
                t = df.tail(len(df) - n)
                assert len(h) + len(t) == len(df)
                return h, t

            def save(df, split):
                df['trans'].to_csv(split + '.zh', header=False, index=False)
                df['origin'].to_csv(split + '.po', header=False, index=False)

            df = pd.read_csv('raw.csv')
            train_df, val_df = split(df, 0.9)
            save(train_df, 'train')
            save(val_df, 'val')
            save(val_df.head(0), 'test')

        print('done.')

    def load_file(self, path):
        with open(path, 'r') as f:
            content = f.read().strip()
        return [list(s.strip()) for s in content.split('\n')]
