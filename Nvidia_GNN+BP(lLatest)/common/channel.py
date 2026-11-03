"""BPSK over AWGN with Eb/N0 parametrisation. LLR > 0 means bit 0."""

import torch


def ebno_to_sigma(ebno_db, rate):
    ebno = 10.0 ** (torch.as_tensor(ebno_db, dtype=torch.float32) / 10.0)
    return torch.sqrt(1.0 / (2.0 * rate * ebno))


def awgn_llr(c, ebno_db, rate, tx_mask=None, punct_scale=0.0):
    """c: [B, n] codeword bits. ebno_db: scalar or [B] tensor.

    Positions where ``tx_mask`` is False are punctured (LLR = 0); with
    punct_scale > 0 they keep that fraction of their channel LLR (training hint).
    """
    sigma = ebno_to_sigma(ebno_db, rate).to(c.device)
    if sigma.dim() == 1:
        sigma = sigma[:, None]
    y = (1.0 - 2.0 * c) + sigma * torch.randn_like(c)
    llr = 2.0 * y / sigma ** 2
    if tx_mask is not None:
        m = tx_mask.to(llr.dtype)
        llr = llr * (m + punct_scale * (1.0 - m))
    return llr
