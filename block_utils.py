from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch


@dataclass(frozen=True)
class BlockPlan:
    lens: torch.LongTensor   # [N]
    starts: torch.LongTensor # [N]


def _distribute_diff_inplace(
    lens: List[int],
    target_sum: int,
    min_len: int,
    max_len: int,
    prefer_tail: bool = True,
) -> None:
    """
    Adjust `lens` in-place so sum(lens) == target_sum, while respecting [min_len, max_len].
    Uses a simple round-robin over indices (tail-first by default).
    """
    cur = sum(lens)
    diff = target_sum - cur
    if diff == 0:
        return

    idxs = list(range(len(lens)))
    if prefer_tail:
        idxs = idxs[::-1]

    # One "step" changes sum by +/-1, so we can always converge if feasible.
    step = 1 if diff > 0 else -1
    remaining = abs(diff)

    # Quick feasibility check
    if step > 0:
        cap = sum(max_len - x for x in lens)
        if remaining > cap:
            raise ValueError(
                f"Cannot increase sum to {target_sum}: need +{remaining}, capacity={cap}"
            )
    else:
        cap = sum(x - min_len for x in lens)
        if remaining > cap:
            raise ValueError(
                f"Cannot decrease sum to {target_sum}: need -{remaining}, capacity={cap}"
            )

    # Round-robin adjust
    p = 0
    while remaining > 0:
        i = idxs[p]
        nxt = lens[i] + step
        if min_len <= nxt <= max_len:
            lens[i] = nxt
            remaining -= 1
        p = (p + 1) % len(idxs)


def make_block_plan_pattern(
    L: int,
    N: int,
    pattern: List[int],
    min_len: int,
    max_len: int,
    device: Optional[torch.device] = None,
) -> BlockPlan:
    """
    Make a plan of N blocks whose lengths repeat `pattern` (clamped to [min_len, max_len]),
    then adjust lengths to ensure total sum is exactly L.
    """
    if N <= 0:
        raise ValueError("N must be > 0")
    if len(pattern) == 0:
        raise ValueError("pattern must be non-empty")
    if min_len > max_len:
        raise ValueError("min_len must be <= max_len")
    if L < N * min_len or L > N * max_len:
        raise ValueError(f"Infeasible L={L} for N={N} with [{min_len},{max_len}]")

    lens: List[int] = []
    for i in range(N):
        l = pattern[i % len(pattern)]
        l = max(min_len, min(max_len, int(l)))
        lens.append(l)

    _distribute_diff_inplace(lens, target_sum=L, min_len=min_len, max_len=max_len, prefer_tail=True)

    lens_t = torch.tensor(lens, dtype=torch.long, device=device)
    starts_t = torch.zeros(N, dtype=torch.long, device=device)
    starts_t[1:] = torch.cumsum(lens_t[:-1], dim=0)
    return BlockPlan(lens=lens_t, starts=starts_t)


def make_block_plan_stages(
    L: int,
    blocks_per_stage: int,
    sizes: List[int],
    min_len: int,
    max_len: int,
    device: Optional[torch.device] = None,
    adjust_only_last_stage: bool = True,
) -> BlockPlan:
    """
    Your requested schedule:
      sizes = [4,8,16,32,64], blocks_per_stage = N  => total blocks = len(sizes)*N
      first N blocks are 4, next N are 8, ..., last N are 64.

    If sum(sizes)*N != L, we adjust to match L while respecting [min_len, max_len].
    By default we only adjust blocks in the LAST stage (so earlier stages stay exact).
    """
    if blocks_per_stage <= 0:
        raise ValueError("blocks_per_stage must be > 0")
    if len(sizes) == 0:
        raise ValueError("sizes must be non-empty")
    if min_len > max_len:
        raise ValueError("min_len must be <= max_len")

    # build scheduled lens
    lens: List[int] = []
    for s in sizes:
        s = max(min_len, min(max_len, int(s)))
        lens.extend([s] * blocks_per_stage)

    Ntotal = len(lens)
    if L < Ntotal * min_len or L > Ntotal * max_len:
        raise ValueError(f"Infeasible L={L} for Ntotal={Ntotal} with [{min_len},{max_len}]")

    if sum(lens) != L:
        if adjust_only_last_stage:
            # adjust only the last `blocks_per_stage` blocks
            head = lens[:-blocks_per_stage]
            tail = lens[-blocks_per_stage:]
            target_tail_sum = L - sum(head)
            if target_tail_sum < blocks_per_stage * min_len or target_tail_sum > blocks_per_stage * max_len:
                # fallback: adjust across all blocks
                _distribute_diff_inplace(lens, target_sum=L, min_len=min_len, max_len=max_len, prefer_tail=True)
            else:
                _distribute_diff_inplace(tail, target_sum=target_tail_sum, min_len=min_len, max_len=max_len, prefer_tail=True)
                lens = head + tail
        else:
            _distribute_diff_inplace(lens, target_sum=L, min_len=min_len, max_len=max_len, prefer_tail=True)

    lens_t = torch.tensor(lens, dtype=torch.long, device=device)
    starts_t = torch.zeros(Ntotal, dtype=torch.long, device=device)
    starts_t[1:] = torch.cumsum(lens_t[:-1], dim=0)
    return BlockPlan(lens=lens_t, starts=starts_t)


def tokens_to_blocks(
    tokens: torch.LongTensor,   # [B, L]
    plan: BlockPlan,
    Bmax: int,
    pad_id: int,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]:
    """
    returns:
      blocks:      [B, N, Bmax]
      block_lens:  [B, N]     (true lengths, must be <= Bmax)
      block_mask:  [B, N, Bmax]  True for real positions
    """
    B, L = tokens.shape
    N = int(plan.lens.numel())

    if int(plan.lens.sum().item()) != L:
        raise ValueError(f"Plan sum {int(plan.lens.sum().item())} != token length L={L}")
    if int(plan.lens.max().item()) > Bmax:
        raise ValueError(f"Plan has block len {int(plan.lens.max().item())} > Bmax={Bmax}. Increase Bmax or clamp lens.")

    blocks = tokens.new_full((B, N, Bmax), pad_id)
    block_mask = torch.zeros((B, N, Bmax), dtype=torch.bool, device=tokens.device)

    for i in range(N):
        s = int(plan.starts[i].item())
        l = int(plan.lens[i].item())
        if s + l > L:
            raise ValueError(f"Block {i} out of range: start={s}, len={l}, L={L}")
        blocks[:, i, :l] = tokens[:, s:s + l]
        block_mask[:, i, :l] = True

    block_lens = plan.lens.unsqueeze(0).expand(B, -1).contiguous()
    return blocks, block_lens, block_mask


def blocks_to_tokens(
    blocks: torch.LongTensor,  # [B, N, Bmax]
    plan: BlockPlan,
    Bmax: int,
) -> torch.LongTensor:
    B, N, Bmax_ = blocks.shape
    if Bmax_ != Bmax:
        raise ValueError("Bmax mismatch")
    if int(plan.lens.numel()) != N:
        raise ValueError("Plan N mismatch with blocks N")
    if int(plan.lens.max().item()) > Bmax:
        raise ValueError(f"Plan has block len {int(plan.lens.max().item())} > Bmax={Bmax}")

    L = int(plan.lens.sum().item())
    tokens = blocks.new_empty((B, L))

    for i in range(N):
        s = int(plan.starts[i].item())
        l = int(plan.lens[i].item())
        tokens[:, s:s + l] = blocks[:, i, :l]

    return tokens


# =========================
# Main sanity tests
# =========================
if __name__ == "__main__":
    print("=== Running block utils sanity tests ===")
    torch.set_printoptions(linewidth=140)

    # --------------------------
    # Test 1: pattern plan
    # --------------------------
    B = 2
    L = 20
    N = 5
    Bmax = 6
    pattern = [2, 3, 5, 4, 6]
    min_len = 1
    max_len = Bmax
    pad_id = -1

    plan = make_block_plan_pattern(L=L, N=N, pattern=pattern, min_len=min_len, max_len=max_len)
    print("\n[Test1] Pattern plan")
    print("  lens  :", plan.lens.tolist())
    print("  starts:", plan.starts.tolist())

    assert plan.lens.numel() == N
    assert int(plan.lens.sum().item()) == L
    assert plan.starts[0].item() == 0
    assert torch.all(plan.starts[1:] == torch.cumsum(plan.lens[:-1], dim=0))

    tokens = torch.arange(B * L).view(B, L)
    blocks, block_lens, block_mask = tokens_to_blocks(tokens=tokens, plan=plan, Bmax=Bmax, pad_id=pad_id)
    tokens_rec = blocks_to_tokens(blocks=blocks, plan=plan, Bmax=Bmax)

    assert torch.equal(tokens, tokens_rec)
    assert torch.equal(block_lens[0], plan.lens)
    assert block_mask.shape == (B, N, Bmax)
    print("tokens <-> blocks roundtrip OK")

    # --------------------------
    # Test 2: stage schedule
    # --------------------------
    sizes = [4, 8, 16, 32, 64]
    blocks_per_stage = 3
    Ntotal = len(sizes) * blocks_per_stage
    Bmax2 = 64
    min_len2, max_len2 = 1, Bmax2
    L2 = sum(sizes) * blocks_per_stage  # exact match (no adjustment needed)

    plan2 = make_block_plan_stages(
        L=L2,
        blocks_per_stage=blocks_per_stage,
        sizes=sizes,
        min_len=min_len2,
        max_len=max_len2,
        adjust_only_last_stage=True,
    )

    print("\n[Test2] Stage schedule plan")
    print("  lens  (first 20):", plan2.lens[:20].tolist(), "...")
    print("  starts(first 10):", plan2.starts[:10].tolist(), "...")

    # invariants
    assert plan2.lens.numel() == Ntotal
    assert int(plan2.lens.sum().item()) == L2
    assert plan2.starts[0].item() == 0
    assert torch.all(plan2.starts[1:] == torch.cumsum(plan2.lens[:-1], dim=0))

    # check the schedule segments exactly
    for k, s in enumerate(sizes):
        seg = plan2.lens[k * blocks_per_stage:(k + 1) * blocks_per_stage]
        assert torch.all(seg == s), f"Stage {k} expected all {s}, got {seg.tolist()}"

    tokens2 = torch.arange(B * L2).view(B, L2)
    blocks2, _, mask2 = tokens_to_blocks(tokens=tokens2, plan=plan2, Bmax=Bmax2, pad_id=pad_id)
    tokens2_rec = blocks_to_tokens(blocks=blocks2, plan=plan2, Bmax=Bmax2)
    assert torch.equal(tokens2, tokens2_rec)

    # IMPORTANT: mask2.all() is NOT True because many blocks have l < Bmax2 -> padding False.
    # Instead, validate per-block correctness:
    for i in range(Ntotal):
        l = int(plan2.lens[i].item())
        assert torch.all(mask2[:, i, :l] == True)
        assert torch.all(mask2[:, i, l:] == False)

    print("scheduled lens segments OK, mask OK, tokens <-> blocks roundtrip OK")

    # --------------------------
    # Test 3: stage schedule with non-matching L (forces adjustment in last stage)
    # --------------------------
    L3 = L2 - 5  # make it slightly smaller
    plan3 = make_block_plan_stages(
        L=L3,
        blocks_per_stage=blocks_per_stage,
        sizes=sizes,
        min_len=min_len2,
        max_len=max_len2,
        adjust_only_last_stage=True,
    )
    print("\n[Test3] Stage schedule with adjustment")
    print("  sum(lens):", int(plan3.lens.sum().item()), "target:", L3)

    # earlier stages should stay exact; last stage may be tweaked
    for k, s in enumerate(sizes[:-1]):
        seg = plan3.lens[k * blocks_per_stage:(k + 1) * blocks_per_stage]
        assert torch.all(seg == s)

    assert int(plan3.lens.sum().item()) == L3
    assert int(plan3.lens.max().item()) <= Bmax2
    assert int(plan3.lens.min().item()) >= min_len2

    print("adjusted only tail stage (or safely fell back) and preserved constraints")

    print("\n-------------\nAll block utils tests passed!")
