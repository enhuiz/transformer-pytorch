import copy
from torchnmt.utils import get_class, get_kwargs
from .multi30k import Multi30kDataset


def get(opts, split):
    opts = copy.deepcopy(opts)
    c = get_class([Multi30kDataset], opts.proto)
    return c(split=split, **get_kwargs(c, opts))
