import copy

from torchnmt.utils import get_class, get_kwargs
from .trainers import NMTTrainer
from .testers import NMTTester
from .validators import NMTValidator


def get(opts, mode):
    opts = copy.deepcopy(opts)
    exec_opts = getattr(opts, mode)
    exec_opts.name = opts.name
    exec_opts.dataset = opts.dataset
    exec_opts.model = opts.model
    c = get_class([NMTTrainer, NMTTester, NMTValidator], exec_opts.proto)
    return c(**get_kwargs(c, opts), opts=exec_opts)
