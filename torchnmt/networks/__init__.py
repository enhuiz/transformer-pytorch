import copy

import torch
import torch.nn as nn

from torchnmt.utils import get_class, get_kwargs
from .encoders import TransformerEncoder
from .decoders import TransformerDecoder
from .transformer import Transformer


def load_state_dict(model, state_dict, ignore_mismatch=True):
    if ignore_mismatch:
        for k, v in model.state_dict().items():
            if k not in state_dict:
                print('Warning: {} is missing'.format(k))
            elif v.shape != state_dict[k].shape:
                del state_dict[k]
    model.load_state_dict(state_dict, strict=not ignore_mismatch)
    return model


def load_model(c, opts):
    kwargs = get_kwargs(c.__init__, opts)
    model = c(**kwargs)
    if hasattr(opts, 'state_dict'):
        state_dict = torch.load(opts.state_dict, map_location='cpu')
        if isinstance(state_dict, nn.Module):
            state_dict = state_dict.state_dict()
        model = load_state_dict(model, state_dict, True)
        print(opts.state_dict, 'loaded.')
    else:
        print('Model {} created.'.format(c.__name__))

    if hasattr(opts, 'requires_grad') and not opts.requires_grad:
        for param in model.parameters():
            param.requires_grad = False

    return model


def get(opts):
    opts = copy.deepcopy(opts)
    c = get_class([TransformerEncoder, TransformerDecoder, Transformer],
                  opts.proto)
    model = load_model(c, opts)
    return model
