import os
import time
import glob

import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tensorboardX import SummaryWriter

from torchnmt import networks, datasets
from torchnmt.scores import compute_scores


class CheckpointSaver(object):
    def __init__(self, root):
        self.root = root

    def save(self, tag, model, global_step):
        path = os.path.join(self.root, '{}/{}.pth'.format(tag, global_step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model, path)

    def get_all_ckpts(self, tag):
        return sorted(glob.glob(os.path.join(self.root, '{}/*.pth'.format(tag))), key=self.parse_step)

    def get_latest_ckpt(self, tag):
        ckpts = self.get_all_ckpts(tag)
        if len(ckpts) > 0:
            return max(ckpts, key=self.parse_step)
        else:
            return None

    def get_latest_step(self, tag) -> int:
        return self.parse_step(self.get_latest_ckpt(tag))

    def parse_step(self, path: str) -> int:
        step = None
        try:
            step = int(path.split(os.path.sep)[-1].replace('.pth', ''))
        except:
            pass
        return step


class Executor(object):
    def __init__(self, name, model, dataset, opts):
        """
        Args:
            model: model opts
            dataset: dataset opts
        """
        self.name = name
        self.model = model
        self.dataset = dataset
        self.opts = opts

        self.writer = SummaryWriter('runs/{}'.format(self.name))
        self.saver = CheckpointSaver('ckpt/{}'.format(self.name))

    def create_model(self, state_dict=None):
        if state_dict is not None:
            self.model.state_dict = state_dict
        return networks.get(self.model).to(self.opts.device)

    def create_dataset(self, split):
        return datasets.get(self.dataset, split)

    def create_data_loader(self, split, shuffle=False):
        dataset = self.create_dataset(split)

        if hasattr(dataset, 'get_collate_fn'):
            collate_fn = dataset.get_collate_fn()
        else:
            collate_fn = default_collate

        return DataLoader(dataset, self.opts.batch_size,
                          shuffle=shuffle,
                          num_workers=4,
                          collate_fn=collate_fn)
