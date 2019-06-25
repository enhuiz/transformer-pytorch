import numpy as np
import torch
import random
import tqdm

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tensorboardX import SummaryWriter

from torchnmt import networks, datasets
from .utils import CheckpointSaver


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

    def done(self):
        return self.epoch >= self.opts.epochs

    def start(self):
        while not self.done():
            self.on_epoch_start()
            self.pbar = tqdm.tqdm(self.dl, total=len(self.dl))
            for batch in self.pbar:
                self.on_iteration_start()
                self.update(batch)
                self.on_iteration_end()
            self.on_epoch_end()

    def on_epoch_start(self):
        pass

    def on_epoch_end(self):
        pass

    def on_iteration_start(self):
        pass

    def on_iteration_end(self):
        self.log()

    def update(self, batch):
        pass

    def log(self):
        pass
