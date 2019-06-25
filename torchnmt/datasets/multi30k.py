import os
import glob
import string
import subprocess

import torch

from torchnmt.datasets.base import NMTDataset
from torchnmt.utils import working_directory


class Multi30kDataset(NMTDataset):
    def __init__(self, root, split, src, tgt, vocab_share=False, download=True):
        super().__init__(root, split, src, tgt, vocab_share, download)

    def download(self, root):
        os.makedirs(root, exist_ok=True)
        print('Downloading dataset.')
        with working_directory(root):
            # credit: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/README.md
            subprocess.call('/bin/bash -c "$scripts"', shell=True, env={'scripts': '''
                wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl
                wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.de
                wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en
                sed -i "s/$RealBin\/..\/share\/nonbreaking_prefixes//" tokenizer.perl
                wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl

                wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz && rm training.tar.gz
                wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz && rm validation.tar.gz
                wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz && tar -xf mmt16_task1_test.tar.gz && rm mmt16_task1_test.tar.gz

                for l in en de; do for f in *.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
                for l in en de; do for f in *.$l; do perl tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok; done; done
                '''})
        print('done.')

    def load_file(self, path):
        with open(path, 'r') as f:
            content = f.read().strip()
        table = str.maketrans(dict.fromkeys(
            string.punctuation + string.digits + "“”"))
        return [s.strip().translate(table).lower().split()
                for s in content.split('\n')]
