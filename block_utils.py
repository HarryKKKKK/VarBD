from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch

@dataclass
class BlockPlan:
    lens: torch.LongTensor   # [N]
    starts: torch.LongTensor # [N], (displacecments)

def make_block_plan(
    L: int,
    N: int,
    pattern: List[int],
    min_len: int,
    max_len: int,
) -> BlockPlan:
    # 1) repeat pattern to get N lens
    lens = []
    for i in range(N):
        l = pattern[i % len(pattern)]
        l = max(min_len, min(max_len, l))
        lens.append(l)

    # 2) fix total sum to exactly L by adjusting the last blocks
    total = sum(lens)
    if total != L:
        diff = L - total
        # push diff into last block (or distribute if you prefer)
        lens[-1] = max(min_len, min(max_len, lens[-1] + diff))
        # if still not equal, do a simple fix by distributing over tail
        # (keeps you from crashing on edge cases)
        total = sum(lens)
        j = N - 1
        while total != L and j >= 0:
            step = 1 if total < L else -1
            if min_len <= lens[j] + step <= max_len:
                lens[j] += step
                total += step
            else:
                j -= 1
        if total != L:
            raise ValueError(f"Cannot make block plan: sum(lens)={total} != L={L}")

    lens_t = torch.tensor(lens, dtype=torch.long)
    starts_t = torch.zeros(N, dtype=torch.long)
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
    blocks: [B, N, Bmax]
    block_lens: [B, N]
    block_mask: [B, N, Bmax] True for real positions
    """
    B, L = tokens.shape
    N = plan.lens.numel()

    blocks = tokens.new_full((B, N, Bmax), pad_id)
    block_mask = torch.zeros((B, N, Bmax), dtype=torch.bool, device=tokens.device)

    for i in range(N):
        s = int(plan.starts[i].item())
        l = int(plan.lens[i].item())
        assert s + l <= L
        take = min(l, Bmax)
        blocks[:, i, :take] = tokens[:, s:s+take]
        block_mask[:, i, :take] = True

    block_lens = plan.lens.unsqueeze(0).expand(B, -1).contiguous()  # [B,N]
    return blocks, block_lens, block_mask

def blocks_to_tokens(
    blocks: torch.LongTensor,      # [B, N, Bmax]
    plan: BlockPlan,
    Bmax: int,
) -> torch.LongTensor:
    B, N, Bmax_ = blocks.shape
    assert(Bmax == Bmax_)
    L = int(plan.lens.sum().item())
    tokens = blocks.new_empty((B, L))
    for i in range(N):
        s = int(plan.starts[i].item())
        l = int(plan.lens[i].item())
        take = min(l, Bmax)
        tokens[:, s:s+take] = blocks[:, i, :take]
        assert(l <= Bmax)
    return tokens

if __name__ == "__main__":
    print("=== Running block utils sanity tests ===")
    torch.set_printoptions(linewidth=120)

    # ===== Test config =====
    B = 2
    L = 20
    N = 5
    Bmax = 6
    pattern = [2, 3, 5, 4, 6]
    min_len = 1
    max_len = Bmax
    pad_id = -1

    # ----- 1. make_block_plan -----
    plan = make_block_plan(
        L=L,
        N=N,
        pattern=pattern,
        min_len=min_len,
        max_len=max_len,
    )

    print("Block lens:", plan.lens.tolist())
    print("Block starts:", plan.starts.tolist())

    # invariants
    assert plan.lens.numel() == N
    assert plan.starts.numel() == N
    assert int(plan.lens.sum().item()) == L
    assert plan.starts[0].item() == 0
    assert torch.all(plan.starts[1:] == torch.cumsum(plan.lens[:-1], dim=0))

    # ----- 2. create dummy tokens -----
    # make tokens easy to inspect
    tokens = torch.arange(B * L).view(B, L)
    print("\nOriginal tokens:")
    print(tokens)

    # ----- 3. tokens -> blocks -----
    blocks, block_lens, block_mask = tokens_to_blocks(
        tokens=tokens,
        plan=plan,
        Bmax=Bmax,
        pad_id=pad_id,
    )

    print("\nBlocks:")
    print(blocks)

    print("\nBlock mask:")
    print(block_mask.int())

    # check block contents
    for i in range(N):
        s = plan.starts[i].item()
        l = plan.lens[i].item()
        # real tokens
        assert torch.equal(blocks[:, i, :l], tokens[:, s:s+l])
        # padding tokens
        if l < Bmax:
            assert torch.all(blocks[:, i, l:] == pad_id)
            assert torch.all(block_mask[:, i, l:] == False)
        assert torch.all(block_mask[:, i, :l] == True)

    # ----- 4. blocks -> tokens -----
    tokens_recovered = blocks_to_tokens(
        blocks=blocks,
        plan=plan,
        Bmax=Bmax,
    )

    print("\nRecovered tokens:")
    print(tokens_recovered)

    # final invariant: perfect reconstruction
    assert torch.equal(tokens, tokens_recovered)

    print("\n-------------\nAll block utils tests passed!")
