import copy
from torchnmt.utils import get_class, get_kwargs
from .multi30k import Multi30kDataset
from .xiaoshi import XiaoshiDataset
from .base import NMTDataset


def get(opts, split):
    opts = copy.deepcopy(opts)
    c = eval(opts.proto)
    return c(split=split, **get_kwargs(c, opts))
