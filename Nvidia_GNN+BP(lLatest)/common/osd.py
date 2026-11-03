"""Batched ordered-statistics decoding (OSD) on the GPU.

As in the paper (GNN + OSD), OSD is used as post-processing on frames for which
the iterative decoder did not reach a valid codeword; the soft output of that
decoder provides the reliability ordering.

order 0: re-encode the hard decisions of the most reliable independent positions.
order w>0: additionally test flipping 1 information bit (all positions) and, for
order 2, pairs among the ``lam`` least reliable information positions (the
"combination sweep" restriction used by BP-OSD-CS decoders); the candidate with
the smallest soft discrepancy sum |llr| over disagreeing positions is chosen.
"""

import torch


@torch.no_grad()
def osd(llr, H, rank, order=0, lam=16):
    """llr: [B, n] soft values, H: [m, n] float/bool tensor, rank: GF(2) rank of H.
    Returns hard decisions [B, n] (float 0/1) that satisfy H x = 0."""
    B, n = llr.shape
    m = H.shape[0]
    work = m * (n - rank if order > 0 else 1)
    chunk = max(16, min(256, int(4e7 // max(work, 1))))
    outs = [_osd_chunk(llr[i:i + chunk], H.bool(), rank, order, lam) for i in range(0, B, chunk)]
    return torch.cat(outs) if outs else llr.new_zeros(0, n)


def osd0(llr, H, rank):
    return osd(llr, H, rank, order=0)


def _eliminate(A, rank):
    """In-place GF(2) Gauss-Jordan on A [B, m, n] (columns already in reliability
    order). Returns pivot column per row ([B, m], -1 for no pivot)."""
    B, m, n = A.shape
    dev = A.device
    row_ids = torch.arange(m, device=dev)
    r = torch.zeros(B, dtype=torch.long, device=dev)
    pivot_col = torch.full((B, m), -1, dtype=torch.long, device=dev)
    b = torch.arange(B, device=dev)
    for j in range(n):
        # host syncs are expensive: check for completion only every 16 columns
        if j % 16 == 0 and bool((r >= rank).all()):
            break
        avail = A[:, :, j] & (row_ids[None, :] >= r[:, None])
        has = avail.any(1) & (r < rank)
        # mask-based row swap / elimination (no data-dependent indexing -> no host sync)
        p = avail.float().argmax(1)
        k = r.clamp(max=m - 1)
        rowp, rowk = A[b, p].clone(), A[b, k].clone()
        h = has[:, None]
        A[b, p] = torch.where(h, rowk, rowp)
        A[b, k] = torch.where(h, rowp, rowk)
        col = A[:, :, j] & h
        col[b, k] = False
        A ^= col[:, :, None] & rowp[:, None, :]
        pivot_col[b, k] = torch.where(has, j, pivot_col[b, k])
        r += has.long()
    return pivot_col


def _osd_chunk(llr, H, rank, w, lam):
    B, n = llr.shape
    m = H.shape[0]
    dev = llr.device
    b = torch.arange(B, device=dev)
    perm = torch.argsort(llr.abs(), dim=1)  # least reliable first
    A = H.T[perm].transpose(1, 2).contiguous()  # [B, m, n] columns permuted per frame
    pivot_col = _eliminate(A, rank)
    valid = pivot_col >= 0
    pc = pivot_col.clamp(min=0)
    hard = torch.gather(llr, 1, perm) < 0
    rel = torch.gather(llr.abs(), 1, perm)
    is_piv = torch.zeros(B, n, dtype=torch.bool, device=dev)
    is_piv[b[:, None].expand(-1, m)[valid], pivot_col[valid]] = True
    free_hard = hard & ~is_piv
    par = ((A & free_hard[:, None, :]).sum(-1) % 2).bool()  # [B, m] pivot bits for order 0
    flip = torch.zeros_like(hard)
    if w > 0:
        h_piv = torch.gather(hard, 1, pc) & valid
        a_piv = torch.gather(rel, 1, pc) * valid
        nf = n - rank
        # free (information) positions in ascending reliability
        fpos = torch.argsort(is_piv.long() * n + torch.arange(n, device=dev)[None], dim=1)[:, :nf]
        Af = torch.gather(A, 2, fpos[:, None, :].expand(-1, m, -1))  # [B, m, nf]
        rf = torch.gather(rel, 1, fpos)
        cands_par = [par[:, :, None]]
        cands_cost_free = [torch.zeros(B, 1, device=dev)]
        cands_flip = [torch.full((B, 1, 2), -1, dtype=torch.long, device=dev)]
        # order 1: every information position
        cands_par.append(par[:, :, None] ^ Af)
        cands_cost_free.append(rf)
        idx1 = torch.arange(nf, device=dev)
        cands_flip.append(torch.stack([idx1, torch.full_like(idx1, -1)], -1)[None].expand(B, -1, -1))
        if w > 1:
            L = min(lam, nf)
            i, j = torch.triu_indices(L, L, 1, device=dev)
            cands_par.append(par[:, :, None] ^ Af[:, :, i] ^ Af[:, :, j])
            cands_cost_free.append(rf[:, i] + rf[:, j])
            cands_flip.append(torch.stack([i, j], -1)[None].expand(B, -1, -1))
        P = torch.cat(cands_par, 2)  # [B, m, C]
        cost = (a_piv[:, :, None] * (P ^ h_piv[:, :, None]).float()).sum(1) + torch.cat(cands_cost_free, 1)
        best = cost.argmin(1)
        par = P[b, :, best]
        fl = torch.cat(cands_flip, 1)[b, best]  # [B, 2] indices into fpos (-1 = none)
        for t in range(2):
            ok = fl[:, t] >= 0
            pos = torch.gather(fpos, 1, fl[:, t].clamp(min=0)[:, None]).squeeze(1)
            flip[b[ok], pos[ok]] = True
    x_sorted = free_hard ^ flip
    x_sorted[b[:, None].expand(-1, m)[valid], pivot_col[valid]] = par[valid]
    x = torch.empty_like(x_sorted)
    x.scatter_(1, perm, x_sorted)
    return x.float()
