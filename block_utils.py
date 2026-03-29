from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch


# =========================
# BlockPlan
# =========================
@dataclass(frozen=True)
class BlockPlan:
    lens: torch.LongTensor   # [Nblocks]
    starts: torch.LongTensor # [Nblocks]


def make_plan_from_lens(
    lens_list: List[int],
    *,
    device: Optional[torch.device] = None,
) -> BlockPlan:
    if len(lens_list) == 0:
        raise ValueError("lens_list must be non-empty")
    if any(int(x) <= 0 for x in lens_list):
        raise ValueError(f"All lens must be > 0, got {lens_list}")

    lens = torch.tensor([int(x) for x in lens_list], dtype=torch.long, device=device)
    starts = torch.zeros((lens.numel(),), dtype=torch.long, device=device)
    if lens.numel() > 1:
        starts[1:] = torch.cumsum(lens[:-1], dim=0)
    return BlockPlan(lens=lens, starts=starts)


# =========================
# Compression (random subsample)
# =========================
def compress_random_subsample(
    tokens_1d: torch.LongTensor,  # [L]
    out_len: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.LongTensor:
    """
    Randomly subsample out_len tokens from tokens_1d WITHOUT replacement,
    then sort indices so the compressed sequence keeps original order.

    - If out_len == L: returns identity.
    - If out_len < L: random subsample.
    - If out_len > L: error.
    """
    if tokens_1d.ndim != 1:
        raise ValueError(f"tokens_1d must be 1D, got shape {tokens_1d.shape}")
    L = int(tokens_1d.numel())
    out_len = int(out_len)
    if out_len <= 0:
        raise ValueError("out_len must be > 0")
    if out_len > L:
        raise ValueError(f"out_len={out_len} cannot exceed original length L={L}")

    if out_len == L:
        return tokens_1d.clone()

    perm = torch.randperm(L, generator=generator, device=tokens_1d.device)
    idx = perm[:out_len]
    idx, _ = torch.sort(idx)  # keep original order
    return tokens_1d[idx]


def make_multiscale_compressed_sequence(
    tokens_1d: torch.LongTensor,   # [L]
    sizes: List[int],
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.LongTensor, BlockPlan, torch.LongTensor]:
    """
    Given original tokens length L, and sizes [s1, s2, ...],
    returns:
      packed_tokens: [sum(sizes)]  (concatenated compressed versions)
      plan: BlockPlan with lens = sizes, starts = cumulative
      levels: LongTensor [Nblocks], levels[k] = k
    """
    if len(sizes) == 0:
        raise ValueError("sizes must be non-empty")
    if tokens_1d.ndim != 1:
        raise ValueError("tokens_1d must be 1D")

    L = int(tokens_1d.numel())
    sizes_i = [int(s) for s in sizes]
    if any(s <= 0 for s in sizes_i):
        raise ValueError(f"All sizes must be > 0, got {sizes}")
    if any(s > L for s in sizes_i):
        raise ValueError(f"All sizes must be <= L={L}, got {sizes}")

    device = tokens_1d.device
    blocks = [compress_random_subsample(tokens_1d, s, generator=generator) for s in sizes_i]
    packed = torch.cat(blocks, dim=0)  # [sum(sizes)]

    plan = make_plan_from_lens(sizes_i, device=device)
    levels = torch.arange(len(sizes_i), dtype=torch.long, device=device)
    return packed, plan, levels


# =========================
# Block <-> token packing
# =========================
def token2block_from_plan(plan: BlockPlan, L: int) -> torch.LongTensor:
    if int(plan.lens.sum().item()) != int(L):
        raise ValueError(f"Plan sum {int(plan.lens.sum().item())} != L={int(L)}")
    t2b = torch.empty((int(L),), dtype=torch.long, device=plan.lens.device)
    for bi in range(int(plan.lens.numel())):
        s = int(plan.starts[bi].item())
        l = int(plan.lens[bi].item())
        t2b[s:s + l] = bi
    return t2b


def tokens_to_blocks(
    tokens: torch.LongTensor,   # [B, Ltot]
    plan: BlockPlan,
    Bmax: int,
    pad_id: int,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]:
    B, Ltot = tokens.shape
    Ntotal = int(plan.lens.numel())

    if int(plan.lens.sum().item()) != Ltot:
        raise ValueError(f"Plan sum {int(plan.lens.sum().item())} != token length Ltot={Ltot}")
    if int(plan.lens.max().item()) > Bmax:
        raise ValueError(f"Plan has block len {int(plan.lens.max().item())} > Bmax={Bmax}")

    blocks = tokens.new_full((B, Ntotal, Bmax), pad_id)
    block_mask = torch.zeros((B, Ntotal, Bmax), dtype=torch.bool, device=tokens.device)

    for i in range(Ntotal):
        s = int(plan.starts[i].item())
        l = int(plan.lens[i].item())
        blocks[:, i, :l] = tokens[:, s:s + l]
        block_mask[:, i, :l] = True

    block_lens = plan.lens.unsqueeze(0).expand(B, -1).contiguous()
    return blocks, block_lens, block_mask


def blocks_to_tokens(
    blocks: torch.LongTensor,  # [B, Ntotal, Bmax]
    plan: BlockPlan,
    Bmax: int,
) -> torch.LongTensor:
    B, Ntotal, Bmax_ = blocks.shape
    if Bmax_ != Bmax:
        raise ValueError("Bmax mismatch")
    if int(plan.lens.numel()) != Ntotal:
        raise ValueError("Plan N mismatch with blocks N")
    if int(plan.lens.max().item()) > Bmax:
        raise ValueError(f"Plan has block len {int(plan.lens.max().item())} > Bmax={Bmax}")

    Ltot = int(plan.lens.sum().item())
    tokens = blocks.new_empty((B, Ltot))

    for i in range(Ntotal):
        s = int(plan.starts[i].item())
        l = int(plan.lens[i].item())
        tokens[:, s:s + l] = blocks[:, i, :l]

    return tokens


# =========================
# Tests
# =========================
def _is_subsequence_of(unique_block: torch.LongTensor, original: torch.LongTensor) -> bool:
    """
    Strict check: unique_block should appear as a subsequence of original in order.
    This assumes original tokens are unique (or at least that equality matching is meaningful).
    """
    # two-pointer subsequence check
    i = 0
    for j in range(original.numel()):
        if i >= unique_block.numel():
            break
        if unique_block[i].item() == original[j].item():
            i += 1
    return i == unique_block.numel()


if __name__ == "__main__":
    torch.set_printoptions(linewidth=140)
    print("=== Running multiscale compression tests (strict) ===")

    # Make a length=128 "unique tokens" sequence so we can strictly verify subsequence order.
    # This mimics real token ids better than repeating chars.
    L = 128
    tokens_1d = torch.arange(L, dtype=torch.long)  # unique
    sizes = [32, 64, 128]

    # reproducible randomness
    g = torch.Generator(device=tokens_1d.device)
    g.manual_seed(123)

    packed, plan, levels = make_multiscale_compressed_sequence(tokens_1d, sizes, generator=g)
    print(len(packed))
    print(packed)
    print(plan)
    print(levels)

    # invariants
    assert int(plan.lens.numel()) == len(sizes) == 3
    assert plan.lens.tolist() == sizes
    assert plan.starts.tolist() == [0, 32, 96]
    assert int(plan.lens.sum().item()) == sum(sizes) == 224
    assert levels.tolist() == [0, 1, 2]
    assert packed.numel() == 224

    # Each block should be a subsequence of original (order preserved)
    for k, s in enumerate(sizes):
        st = int(plan.starts[k].item())
        ln = int(plan.lens[k].item())
        blk = packed[st:st + ln]
        assert _is_subsequence_of(blk, tokens_1d), f"Block {k} is not a subsequence of original"

    # Pack/unpack roundtrip through blocks
    Bmax = max(sizes)
    pad_id = -1
    blocks, block_lens, block_mask = tokens_to_blocks(packed.unsqueeze(0), plan, Bmax=Bmax, pad_id=pad_id)
    rec = blocks_to_tokens(blocks, plan, Bmax=Bmax)
    assert torch.equal(rec.squeeze(0), packed), "Roundtrip packed->blocks->packed failed"

    # mask correctness
    for i, s in enumerate(sizes):
        assert block_mask[0, i, :s].all()
        assert (~block_mask[0, i, s:]).all()
        assert (blocks[0, i, s:] == pad_id).all()

    print("All invariants passed: plan OK, subsequence order OK, roundtrip OK, mask OK")
    print("-------------\nAll tests passed!")
