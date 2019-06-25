import numpy as np
import torch
import tqdm

from torchnmt.executors.base import Executor


class Trainer(Executor):
    def __init__(self, opts):
        super().__init__(opts)
        self.writer = self.create_writer('train')

    def create_optimizer(self):
        if hasattr(self.opts, 'optimizer'):
            optimizer = self.opts.optimizer
        else:
            optimizer = 'adam'

        params = self.model.parameters()

        if optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.opts.lr)
        elif optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.lr)
        else:
            raise Exception("Unknown optimizer {}.".format(optimizer))

        return optimizer

    def create_model(self):
        epoch0, state_dict = (1, None)
        if self.opts.continued:
            try:
                epoch0, state_dict = self.saver.get_latest_ckpt('epoch')
                epoch0 += 1  # start from next epoch
            except:
                pass
        model = super().create_model(state_dict)
        return epoch0, model

    def epoch_iter(self):
        for self.epoch in range(self.opts.epochs):
            yield

    def iteration_iter(self):
        self.pbar = tqdm.tqdm(self.dl, total=len(self.dl))
        for batch in self.pbar:
            self.batch = batch
            yield

    def start(self):
        self.lr = self.opts.lr
        self.dl = self.create_data_loader('train', shuffle=True)
        self.epoch, self.model = self.create_model()
        self.optimizer = self.create_optimizer()
        self.iteration = (self.epoch - 1) * len(self.dl) + 1
        self.model.train()
        super().start()

    def on_iteration_end(self):
        self.writer.add_scalar('batch-loss', self.loss, self.iteration)
        self.log()
        self.iteration += 1

    def on_epoch_end(self):
        if self.epoch % self.opts.save_every == 0:
            self.saver.save('epoch', self.model.state_dict(), self.epoch)
        self.update_lr()
        self.epoch += 1

    def update_lr(self):
        lr = self.opts.lr * 0.95 ** (self.epoch // 2)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.lr = lr

    def log(self, extra_items=[]):
        items = [
            'Epoch: [{}/{}] '.format(self.epoch, self.opts.epochs),
            'lr: {:.4g}'.format(self.lr),
            'loss: {:.4g}'.format(self.loss),
        ] + extra_items
        self.pbar.set_description(' '.join(items))


class NMTTrainer(Trainer):
    def __init__(self, opts):
        super().__init__(opts)

    def on_epoch_start(self):
        super().on_epoch_start()

        self.losses = []

    def update(self):
        loss = self.model(**self.batch, **vars(self.opts))['loss']
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.loss = loss.item()
        self.ppl = np.exp(self.loss)
        self.losses.append(self.loss)

    def on_iteration_end(self):
        self.writer.add_scalar('batch-ppl', self.ppl, self.iteration)
        super().on_iteration_end()

    def on_epoch_end(self):
        loss = np.mean(self.losses)
        ppl = np.exp(loss)

        print('Epoch {}'.format(self.epoch))
        print('Train:\tloss: {:.4g}, ppl: {:.4g}'.format(loss,
                                                         ppl))

        self.writer.add_scalar('loss', loss, self.epoch)
        self.writer.add_scalar('ppl', ppl, self.epoch)
        self.writer.flush()  # requires: https://github.com/lanpa/tensorboardX/pull/451

        super().on_epoch_end()

    def log(self):
        super().log([
            'ppl: {:.4g}'.format(self.ppl),
        ])
