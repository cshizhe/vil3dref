import torch
from einops import repeat

def gen_seq_masks(seq_lens, max_len=None):
    """
    Args:
        seq_lens: torch.LongTensor, shape=(N, )
    Returns:
        masks: torch.BoolTensor, shape=(N, L), padded=0
    """
    if max_len is None:
        max_len = max(seq_lens)
    batch_size = len(seq_lens)
    seq_masks = repeat(torch.arange(max_len).long(), 'l -> b l', b=batch_size)
    seq_masks = seq_masks  < seq_lens.unsqueeze(1)
    return seq_masks

def pad_tensors(tensors, lens=None, pad=0, pad_ori_data=False):
    """B x [T, ...] torch tensors"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = list(tensors[0].size()[1:])
    size = [bs, max_len] + hid

    dtype = tensors[0].dtype
    output = torch.zeros(*size, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
        if pad_ori_data:
            rt = (max_len - l) // l + 1
            for j in range(rt):
                s = l + j * l
                e = min(s + l, max_len)
                output.data[i, s: e] = t.data[:e-s]
    return output

