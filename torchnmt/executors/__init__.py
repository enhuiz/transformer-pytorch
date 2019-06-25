import copy

from torchnmt.utils import get_class, get_kwargs
from .trainers import NMTTrainer
from .testers import NMTTester


def get(opts, train=False):
    opts = copy.deepcopy(opts)
    if train:
        exec_ops = opts.train
    else:
        exec_ops = opts.test
    exec_ops.name = opts.name
    exec_ops.dataset = opts.dataset
    exec_ops.model = opts.model
    c = get_class([NMTTrainer, NMTTester], exec_ops.proto)
    return c(**get_kwargs(c, opts), opts=exec_ops)
