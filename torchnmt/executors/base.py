import numpy as np
import torch
import random
import tqdm

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tensorboardX import SummaryWriter

from torchnmt import networks, datasets
from .utils import CheckpointSaver


class EpochSkipper(Exception):
    pass


class IterationSkipper(Exception):
    pass


class Executor(object):
    def __init__(self, opts, random_seed=7):
        self.opts = opts
        self.saver = CheckpointSaver('ckpt/{}'.format(self.opts.name))
        self.set_seed(random_seed)

    def create_writer(self, split):
        return SummaryWriter('runs/{}/{}'.format(self.opts.name, split))

    def create_model(self, state_dict=None):
        if state_dict is not None:
            self.opts.model.state_dict = state_dict
        return networks.get(self.opts.model).to(self.opts.device)

    def create_dataset(self, split):
        return datasets.get(self.opts.dataset, split)

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

    def set_seed(self, seed):
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed(seed)  # gpu
        np.random.seed(seed)  # numpy
        random.seed(seed)  # random and transforms
        torch.backends.cudnn.deterministic = True  # cudnn

    def start(self):
        for _ in self:
            pass

    def __iter__(self):
        for _ in self.epoch_iter():
            try:
                self.on_epoch_start()
                for _ in self.iteration_iter():
                    try:
                        self.on_iteration_start()
                        self.update()
                        self.on_iteration_end()
                        yield
                    except IterationSkipper:
                        continue
                self.on_epoch_end()
            except EpochSkipper:
                continue

    def skip_epoch(self):
        """
        Skip the current epoch.
        """
        raise EpochSkipper()

    def skip_iteration(self):
        """
        Skip the current iteration.
        """
        raise IterationSkipper()

    def epoch_iter(self):
        raise NotImplementedError()

    def iteration_iter(self):
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()

    def on_epoch_start(self):
        return

    def on_epoch_end(self):
        return

    def on_iteration_start(self):
        return

    def on_iteration_end(self):
        return
