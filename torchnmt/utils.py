import os
import inspect
import argparse
import yaml


from torch.nn.utils.rnn import pad_packed_sequence


def get_class(class_list, name):
    class_dict = {class_.__name__.lower(): class_ for class_ in class_list}
    name_lower = name.lower()
    if name_lower in class_dict:
        class_ = class_dict[name_lower]
    else:
        raise Exception("{} not found.".format(name))
    return class_


def get_kwargs(func, opts):
    args = inspect.getargspec(func).args
    opts = vars(opts)
    args = {k: v for k, v in opts.items() if k in args}
    return args


def make_namespace(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = make_namespace(d[key])
    return argparse.Namespace(**d)


def parse_config(p):
    with open(p, 'r') as stream:
        d = yaml.load(stream, Loader=yaml.FullLoader)
    ns = make_namespace(d)
    name = p.split('config' + os.path.sep)[1]
    ns.name = os.path.splitext(name)[0]
    return ns


def unpack_packed_sequence(sequence):
    seqs, lens = pad_packed_sequence(sequence, True)
    return [seq[:l] for seq, l in zip(seqs, lens)]
