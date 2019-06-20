import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from torchnmt.executors import Trainer
from torchnmt.utils import parse_config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    opts = parse_config(args.config)

    trainer = Trainer(opts)
    trainer.start()


if __name__ == "__main__":
    main()
