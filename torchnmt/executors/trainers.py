import numpy as np
import torch
import tqdm

from torchnmt.executors.base import Executor


class Trainer(Executor):
    def __init__(self, opts):
        super().__init__(opts)

    def create_optimizer(self):
        if hasattr(self.opts, 'optimizer'):
            optimizer = self.opts.optimizer
        else:
            optimizer = 'sgd'

        params = self.model.parameters()

        if optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.opts.lr)
        elif optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.lr)
        else:
            raise Exception("Unknown optimizer {}.".format(optimizer))

        return optimizer

    def create_model(self):
        epoch0 = 1
        state_dict = None
        if self.opts.continued:
            ckpt = self.saver.get_latest_ckpt('epoch')
            if ckpt is not None:
                state_dict = ckpt
                epoch0 = self.saver.get_latest_step('epoch') + 1
        model = super().create_model(state_dict)
        return model, epoch0

    def start(self):
        self.lr = self.opts.lr
        self.dl = self.create_data_loader('train', shuffle=True)
        self.model, self.epoch = self.create_model()
        self.optimizer = self.create_optimizer()
        self.iteration = self.epoch * len(self.dl) + 1
        self.model.train()
        super().start()

    def on_iteration_end(self):
        self.writer.add_scalar('loss', self.loss, self.iteration)
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

    def update(self, batch):
        loss = self.model(**batch,
                          **vars(self.opts))['loss']
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.loss = loss.item()
        self.ppl = np.exp(self.loss)
        self.losses.append(self.loss)

    def on_iteration_end(self):
        self.writer.add_scalar('ppl', self.ppl, self.epoch)

        super().on_iteration_end()

    def on_epoch_end(self):
        loss = np.mean(self.losses)
        ppl = np.exp(loss)

        print('Epoch {}'.format(self.epoch))
        print('Train:\tloss: {:.4g}, ppl: {:.4g}'.format(loss,
                                                         ppl))

        self.writer.add_scalar('epoch_train_loss', loss, self.epoch)
        self.writer.add_scalar('epoch_train_ppl', ppl, self.epoch)

        super().on_epoch_end()

    def log(self):
        super().log([
            'ppl: {:.4g}'.format(self.ppl),
        ])
