import os
import glob
import torch
from typing import Dict, Tuple


class CheckpointSaver(object):
    def __init__(self, root):
        self.root = root

    def save(self, tag, model, global_step):
        path = os.path.join(self.root, '{}/{}.pth'.format(tag, global_step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model, path)
        print('{} saved.'.format(path))

    def get_all_ckpts(self, tag) -> Dict[int, str]:
        ckpts = {self.parse_step(path): path for path in glob.glob(
            os.path.join(self.root, '{}/*.pth'.format(tag)))}
        return ckpts

    def get_latest_ckpt(self, tag) -> Tuple[int, str]:
        ckpts = self.get_all_ckpts(tag)
        if not len(ckpts):
            raise Exception("No ckpts found.")
        return max(ckpts.items())

    def parse_step(self, path: str) -> int:
        step = int(path.split(os.path.sep)[-1].replace('.pth', ''))
        return step
