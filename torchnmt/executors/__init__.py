import copy
from torchnmt.utils import get_class, get_kwargs
from .trainers import NMTTrainer
from .testers import NMTTester


def get(opts, executor_ops):
    opts = copy.deepcopy(opts)
    c = get_class([NMTTrainer, NMTTester], executor_ops.proto)
    return c(**get_kwargs(c, opts), opts=executor_ops)
