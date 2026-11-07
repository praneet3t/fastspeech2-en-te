"""Only the tensor helpers used by the retained upstream architecture."""
import torch
import torch.nn.functional as F


def get_mask_from_lengths(lengths, max_len=None):
    max_len = int(lengths.max()) if max_len is None else max_len
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)


def pad(inputs, max_len=None):
    max_len = max(x.shape[0] for x in inputs) if max_len is None else max_len
    if any(x.shape[0] > max_len for x in inputs):
        raise ValueError('Padding length would truncate acoustic frames')
    return torch.stack([F.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in inputs])
