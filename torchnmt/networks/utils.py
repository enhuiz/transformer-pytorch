import torch


def padding_mask(lens):
    """Mask out the blank (padding) values
    Args:
        lens: (bs,)
    Return:
        mask: (bs, 1, max_len)
    """
    bs, max_len = len(lens), max(lens)
    mask = torch.zeros(bs, 1, max_len)
    for i, l in enumerate(lens):
        mask[i, :, :l] = 1
    mask = mask > 0
    return mask


def subsequent_mask(lens):
    """Mask out future word
    Args:
        lens: (bs,)
    Return:
        mask: (bs, max_len, max_len)
    """
    bs, max_len = len(lens), max(lens)
    mask = torch.ones([bs, max_len, max_len]).tril_(0)
    mask = mask > 0
    return mask
