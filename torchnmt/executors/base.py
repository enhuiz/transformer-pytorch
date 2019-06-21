from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tensorboardX import SummaryWriter

from torchnmt import networks, datasets
from .utils import CheckpointSaver


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
