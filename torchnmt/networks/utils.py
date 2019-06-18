import copy
import inspect

import torch
import torch.nn as nn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def load_state_dict(model, state_dict, ignore_mismatch=True):
    if ignore_mismatch:
        for k, v in model.state_dict().items():
            if k not in state_dict:
                print('Warning: {} is missing'.format(k))
                continue
            v_ = state_dict[k]
            if v.shape != v_.shape:
                del state_dict[k]
    model.load_state_dict(state_dict, strict=not ignore_mismatch)
    return model


def get_class(class_list, name):
    class_dict = {class_.__name__.lower(): class_ for class_ in class_list}
    class_ = None
    name = name.lower()
    if name in class_dict:
        class_ = class_dict[name]
    return class_


def get_kwargs(func, opts):
    args = inspect.getargspec(func).args
    opts = vars(opts)
    args = {k: v for k, v in opts.items() if k in args}
    return args


def fetch_model(class_list, opts):
    c = get_class(class_list, opts.proto)
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
