import os
import sys
import argparse
import torch
import shutil
from functools import partial

desc = 'Rename configuration(s) and update conresponding ckpt/runs/results.'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('src')
parser.add_argument('dst')
args = parser.parse_args()


def strip_config(path):
    config = 'config' + os.path.sep
    if path[:len(config)] == config:
        path = path.replace(config, '', 1)
    return path


assert os.path.exists(args.src)

args.src = strip_config(args.src)
args.dst = strip_config(args.dst)


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


def remove_ext(path):
    return os.path.splitext(path)[0]


move('config', args.src, args.dst)

rename = partial(move, src=remove_ext(args.src), dst=remove_ext(args.dst))
rename('results')
rename('ckpt')
rename('runs')
