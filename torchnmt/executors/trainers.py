import torch
import tqdm

from torchnmt.executors.base import Executor


class Trainer(Executor):
    def __init__(self, name, model, dataset, opts):
        super().__init__(name, model, dataset, opts)
        self.loss = None
        self.lr = self.opts.lr
        self.dl = self.create_data_loader('train', shuffle=True)
        self.model, self.epoch = self.create_model()
        self.optimizer = self.create_optimizer(self.model.parameters())
        self.iteration = self.epoch * len(self.dl) + 1

    def create_optimizer(self, params):
        if hasattr(self.opts, 'optimizer'):
            optimizer = self.opts.optimizer
        else:
            optimizer = 'sgd'
        if optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.opts.lr)
        elif optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.lr)
        else:
            raise Exception("Unknown optimizer {}.".format(optimizer))
        return optimizer

    def create_model(self):
        epoch = 1
        state_dict = None
        if self.opts.continued:
            ckpt = self.saver.get_latest_ckpt('epoch')
            if ckpt is not None:
                state_dict = ckpt
                epoch = self.saver.get_latest_step('epoch') + 1
        model = super().create_model(state_dict)
        return model, epoch

    def start(self):
        self.model.train()

        while self.epoch <= self.opts.epochs:
            self.pbar = tqdm.tqdm(self.dl, total=len(self.dl))

            for batch in self.pbar:
                self.backward(batch)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.iteration_hook()
                self.iteration += 1

            self.epoch_hook()
            self.epoch += 1

    def backward(self, batch):
        raise NotImplementedError()

    def iteration_hook(self):
        self.write_summary()
        self.log_pbar()

    def epoch_hook(self):
        if self.epoch % self.opts.save_every == 0:
            self.saver.save('epoch', self.model.state_dict(), self.epoch)
        self.update_lr()

    def update_lr(self):
        lr = self.opts.lr * 0.95 ** (self.epoch // 2)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.lr = lr

    def write_summary(self):
        self.writer.add_scalar('loss', self.loss.item(), self.iteration)

    def log_pbar(self, extra_keys=[], extra_values=[]):
        keys = ['epoch: [{}/{}]',
                'loss: {:.4g}',
                'lr: {:.4g}'] + extra_keys

        values = [(self.epoch, self.opts.epochs),
                  self.loss.item(),
                  self.lr] + extra_values

        def combine(k, v):
            return k.format(*v) if isinstance(v, tuple) else k.format(v)

        desc = ' '.join(map(combine, keys, values)).capitalize()

        self.pbar.set_description(desc)


class NMTTrainer(Trainer):
    def __init__(self, name, model, dataset, opts):
        super().__init__(name, model, dataset, opts)

    def backward(self, batch):
        loss = self.model(batch['src'], batch['tgt'],
                          **vars(self.opts))['loss']
        loss.backward()
        self.loss = loss
