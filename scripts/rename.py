import os
import sys
import argparse
import torch
import shutil
from functools import partial

desc = 'Rename configuration(s) and update the conresponding ckpt/runs/results.'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('src')
parser.add_argument('dst')
args = parser.parse_args()


def strip_config(path):
    config = 'config' + os.path.sep
    if path[:7] == config:
        path = path.replace(config, '', 1)
    path = os.path.splitext(path)[0]
    return path


args.src = strip_config(args.src)
args.dst = strip_config(args.dst)


assert os.path.exists(os.path.join('config', args.src))


def remove_empty_dir_along(src):
    if not src:
        return
    if not os.listdir(src):
        os.remove(src)
    remove_empty_dir_along(os.path.dirname(src))


def move(base, src, dst):
    src = os.path.join(base, src)
    dst = os.path.join(base, dst)
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        remove_empty_dir_along(os.path.dirname(src))
        print('{} is renamed to {}.'.format(src, dst))
    else:
        print('{} does not exist.'.format(src))


move_base = partial(move, src=args.src, dst=args.dst)

move_base('results')
move_base('ckpt')
move_base('runs')
move_base('config')
