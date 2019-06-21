import torch
import tqdm

from torchnmt.executors.utils import Executor


class Trainer(Executor):
    def __init__(self, name, model, dataset, opts):
        super().__init__(name, model, dataset, opts)

    def get_optimizer(self, params):
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

    def update_lr(self, optimizer):
        lr = self.opts.lr * 0.95 ** (self.epoch // 2)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def create_model(self):
        state_dict = None
        if self.opts.continued:
            ckpt = self.saver.get_latest_ckpt('epoch')
            if ckpt is not None:
                state_dict = ckpt
                self.epoch = self.saver.get_latest_step('epoch') + 1
        model = super().create_model(state_dict)
        return model

    def start(self):
        dl = self.create_data_loader('train', shuffle=True)
        model = self.create_model().train()
        optimizer = self.get_optimizer(model.parameters())
        iteration = self.epoch * len(dl)

        while self.epoch <= self.opts.epochs:
            lr = self.update_lr(optimizer)
            pbar = tqdm.tqdm(dl, total=len(dl))
            for batch in pbar:
                loss = self.criterion(model, batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                pbar.set_description('Epoch: [{}/{}], '
                                     .format(self.epoch, self.opts.epochs) +
                                     'loss: {:.4g}, '
                                     .format(loss.item()) +
                                     'lr: {:.4g}'
                                     .format(lr))

                self.writer.add_scalar('loss', loss.item(), iteration)

                iteration += 1

            if self.epoch % self.save_every == 0:
                self.saver.save('epoch', model.state_dict(), self.epoch)

            self.epoch += 1

    def criterion(self, model, batch):
        raise NotImplementedError()


class NMTTrainer(Trainer):
    def __init__(self, name, model, dataset, opts):
        return super().__init__(name, model, dataset, opts)

    def criterion(self, model, batch):
        return model(batch['src'], batch['tgt'], **vars(self.opts))['loss']
