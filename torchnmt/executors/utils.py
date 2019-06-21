import os
import glob
import torch


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
