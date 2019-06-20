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
from torchnmt.utils import unpack_packed_sequence


class CheckpointSaver():
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


class Executor():
    def __init__(self, opts):
        self.opts = opts
        self.name = opts.name
        self.writer = SummaryWriter('runs/{}'.format(self.name))
        self.saver = CheckpointSaver('ckpt/{}'.format(self.name))
        self.model = networks.get(opts.model)

    def get_dataset(self, split):
        return datasets.get(self.opts.dataset, split)

    def create_data_loader(self, dataset, batch_size):
        if hasattr(dataset, 'get_collate_fn'):
            collate_fn = dataset.get_collate_fn()
        else:
            collate_fn = default_collate

        return DataLoader(dataset, batch_size,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn)


class Trainer(Executor):
    def __init__(self, opts):
        super().__init__(opts)
        self.init(opts.train)

    def get_optimizer(self, optimizer, params, lr):
        if optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(params, lr=lr)
        elif optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(params, lr=lr)
        else:
            raise Exception("Unknown optimizer {}.".format(optimizer))
        return optimizer

    def init(self, opts):
        self.dataset = self.get_dataset('train')
        self.dl = self.create_data_loader(self.dataset, opts.batch_size)
        self.lr0 = opts.lr
        self.lr = opts.lr
        self.epochs = opts.epochs
        self.epoch = 0

        if opts.continued:
            ckpt = self.saver.get_latest_ckpt('epoch')
            if ckpt is not None:
                self.model.load_state_dict(torch.load(ckpt))
                self.epoch = self.saver.get_latest_step('epoch') + 1
                print('{} loaded.'.format(ckpt))

        self.model = self.model.to(opts.device)

        if not hasattr(opts, 'optimizer'):
            opts.optimizer = 'sgd'

        self.optimizer = self.get_optimizer(opts.optimizer,
                                            self.model.parameters(),
                                            opts.lr)

    def update_lr(self):
        lr = self.lr0 * 0.95 ** (self.epoch // 2)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.lr = lr

    def start(self):
        self.model.train()

        while self.epoch < self.epochs:
            self.update_lr()

            iteration = self.epoch * len(self.dl)
            pbar = tqdm.tqdm(self.dl, total=len(self.dl))
            for batch in pbar:
                loss = self.model(**batch, **vars(self.opts.train))['loss']
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                pbar.set_description('Epoch: [{}/{}], '
                                     .format(self.epoch, self.epochs) +
                                     'loss: {:.4g}, '
                                     .format(loss.item()) +
                                     'ppl: {:.4g}, '.format(torch.exp(loss).item()) +
                                     'lr: {:.4g}'
                                     .format(self.lr))

                self.writer.add_scalar('loss', loss.item(),
                                       iteration)
                self.writer.add_scalar('ppl', torch.exp(loss).item(),
                                       iteration)
                iteration += 1

            self.saver.save('epoch', self.model.state_dict(), self.epoch)
            self.epoch += 1


class Tester(Executor):
    def __init__(self, opts):
        super().__init__(opts)
        self.init(opts.test)

    def init(self, opts):
        self.splits = opts.splits
        self.dss = {sp: self.get_dataset(sp)
                    for sp in opts.splits}
        self.dls = {sp: self.create_data_loader(self.dss[sp], opts.batch_size)
                    for sp in self.dss}
        self.vocab = self.dss[self.splits[0]].tgt_vocab
        self.model = self.model.to(opts.device)

    def start(self):
        self.model.eval()

        k = self.opts.test.eval_every
        ckpts = self.saver.get_all_ckpts('epoch')
        ckpts = ckpts[k - 1::k]

        for ckpt in ckpts:
            epoch = self.saver.parse_step(ckpt)
            for split in self.splits:
                out_dir = os.path.join('results', self.name, str(epoch), split)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                    print('Evaluating ...', epoch, split)
                    self.model.load_state_dict(torch.load(ckpt))
                    self.evaluate(out_dir, self.dls[split])

    def evaluate(self, out_dir, dl):
        model = self.model.eval()

        refs, hyps = [], []
        for batch in tqdm.tqdm(dl, total=len(dl)):
            batch_refs = unpack_packed_sequence(batch['tgt'])
            with torch.no_grad():
                batch_hyps = model(src=batch['src'],
                                   **vars(self.opts.test))['hyps']

            refs += [self.vocab.strip_beos_w(self.vocab.idxs2words(ref))
                     for ref in batch_refs]
            hyps += [self.vocab.strip_beos_w(self.vocab.idxs2words(hyp))
                     for hyp in batch_hyps]

        refs = list(map(' '.join, refs))
        hyps = list(map(' '.join, hyps))

        df = pd.DataFrame({'refs': refs, 'hyps': hyps})

        pathbase = os.path.join(out_dir, '{}')

        with open(pathbase.format('scores.txt'), 'w') as f:
            f.write(str(compute_scores(refs, hyps)))

        df['refs'].to_csv(pathbase.format('refs.txt'),
                          header=False, index=False)

        df['hyps'].to_csv(pathbase.format('hyps.txt'),
                          header=False, index=False)
