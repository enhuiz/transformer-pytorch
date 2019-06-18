import copy

from .encoders import TransformerEncoder
from .decoders import TransformerDecoder
from .utils import fetch_model


def get(opts):
    opts = copy.deepcopy(opts)
    model_list = [TransformerEncoder, TransformerDecoder]
    model = fetch_model(model_list, opts)
    return model
