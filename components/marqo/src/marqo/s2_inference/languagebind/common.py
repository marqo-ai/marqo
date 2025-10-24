from typing import Optional

import torch


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.

    The method is copied from here: https://github.com/huggingface/transformers/blob/e42587f596181396e1c4b63660abf0c736
    b10dae/src/transformers/models/clip/modeling_clip.py#L49C1-L60C91

    Our existing transformers library does not have this method, so we copied it here.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)