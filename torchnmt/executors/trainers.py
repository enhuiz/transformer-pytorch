import numpy as np
import torch
import tqdm

from torchnmt.executors.base import Executor


class Trainer(Executor):
    def __init__(self, name, model, dataset, opts):
        super().__init__(name, model, dataset, opts)
        self.loss = None
        self.lr = self.opts.lr
        self.dl = self.create_data_loader(opts.splits.train, shuffle=True)
        self.model, self.epoch0 = self.create_model()
        self.optimizer = self.create_optimizer(self.model.parameters())
        self.iteration0 = self.epoch0 * len(self.dl) + 1

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
        self.model.train()
        epoch = self.epoch0
        iteration = self.iteration0

        while epoch <= self.opts.epochs:
            pbar = tqdm.tqdm(self.dl, total=len(self.dl))

            for batch in pbar:
                self.backward(batch)
                self.optimizer.step()
                self.optimizer.zero_grad()

                base = 'Epoch: [{}/{}] '.format(epoch, self.opts.epochs)
                self.log(lambda desc: pbar.set_description(base + desc))

                self.iteration_hook(iteration)
                iteration += 1

            self.epoch_hook(epoch)
            epoch += 1

    def backward(self, batch):
        raise NotImplementedError()

    def iteration_hook(self, iteration):
        self.writer.add_scalar('loss', self.loss, iteration)

    def epoch_hook(self, epoch):
        if epoch % self.opts.save_every == 0:
            self.saver.save('epoch', self.model.state_dict(), epoch)
        self.update_lr(epoch)

    def update_lr(self, epoch):
        lr = self.opts.lr * 0.95 ** (epoch // 2)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.lr = lr

    def log(self, logger, extra_keys=[], extra_values=[]):
        keys = ['lr: {:.4g}',
                'loss: {:.4g}'] + extra_keys

        values = [self.lr,
                  self.loss] + extra_values

        def combine(k, v):
            return k.format(*v) if isinstance(v, tuple) else k.format(v)

        desc = ' '.join(map(combine, keys, values))

        logger(desc)


class NMTTrainer(Trainer):
    def __init__(self, name, model, dataset, opts):
        super().__init__(name, model, dataset, opts)
        self.losses = []  # global loss
        if hasattr(opts.splits, 'val'):
            self.val_dl = self.create_data_loader(opts.splits.val)
        else:
            self.val_dl = None

    def backward(self, batch):
        loss = self.model(**batch,
                          **vars(self.opts))['loss']
        loss.backward()
        self.loss = loss.item()
        self.ppl = np.exp(self.loss)
        self.losses.append(self.loss)

    def iteration_hook(self, iteration):
        super().iteration_hook(iteration)
        self.writer.add_scalar('ppl', self.ppl, iteration)

    def log(self, logger, extra_keys=[], extra_values=[]):
        super().log(logger, ['ppl: {:.4g}'], [self.ppl])

    def epoch_hook(self, epoch):
        super().epoch_hook(epoch)

        train_loss = np.mean(self.losses)
        train_ppl = np.exp(train_loss)
        self.losses = []
        print('Epoch {}'.format(epoch))
        print('Train:\tloss: {:.4g}, ppl: {:.4g}'.format(train_loss,
                                                         train_ppl))

        self.writer.add_scalar('epoch_train_ppl', train_ppl, epoch)
        self.writer.add_scalar('epoch_train_loss', train_loss, epoch)

        if self.val_dl:
            val_loss, val_ppl = self.compute_loss_ppl(self.val_dl)
            print('Valid:\tloss: {:.4g}, ppl: {:.4g}'.format(val_loss,
                                                             val_ppl))

            self.writer.add_scalar('epoch_val_ppl', val_ppl, epoch)
            self.writer.add_scalar('epoch_val_loss', val_loss, epoch)

    def compute_loss_ppl(self, dl):
        losses = []
        with torch.no_grad():
            for batch in dl:
                loss = self.model(**batch, **vars(self.opts))['loss']
                losses.append(loss.item())
        loss = np.mean(losses)
        ppl = np.exp(loss)
        return loss, ppl
