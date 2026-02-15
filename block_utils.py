from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch


@dataclass(frozen=True)
class BlockPlan:
    lens: torch.LongTensor
    starts: torch.LongTensor

def make_fixed_plan(
    L: int,
    block_size: int,
    *,
    device: Optional[torch.device] = None,
) -> BlockPlan:
    """
    Split a length-L sequence into blocks of fixed size `block_size`.
    The last block is allowed to be shorter (must end the block at sentence end).
    """
    if L <= 0:
        raise ValueError("L must be > 0")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    n = (L + block_size - 1) // block_size 
    lens = [block_size] * n
    lens[-1] = L - block_size * (n - 1)

    lens_t = torch.tensor(lens, dtype=torch.long, device=device)
    starts_t = torch.zeros(n, dtype=torch.long, device=device)
    starts_t[1:] = torch.cumsum(lens_t[:-1], dim=0)
    return BlockPlan(lens=lens_t, starts=starts_t)


def make_multilevel_block_plan_for_repeated_sequence(
    L_single: int,
    block_sizes: List[int],
    *,
    device: Optional[torch.device] = None,
) -> Tuple[BlockPlan, torch.LongTensor]:
    """
    Returns:
      global_plan: BlockPlan over total length Ltot = K * L_single
      levels: LongTensor [Ntotal], level id per block (0..K-1)
    """
    if L_single <= 0:
        raise ValueError("L_single must be > 0")
    if len(block_sizes) == 0:
        raise ValueError("block_sizes must be non-empty")

    all_lens = []
    all_starts = []
    all_levels = []

    K = len(block_sizes)
    for k, b in enumerate(block_sizes):
        plan_k = make_fixed_plan(L_single, int(b), device=device)
        offset = k * L_single
        all_lens.append(plan_k.lens)
        all_starts.append(plan_k.starts + offset)
        all_levels.append(torch.full((plan_k.lens.numel(),), k, dtype=torch.long, device=device))

    lens = torch.cat(all_lens, dim=0)
    starts = torch.cat(all_starts, dim=0)
    levels = torch.cat(all_levels, dim=0)

    return BlockPlan(lens=lens, starts=starts), levels


# =========================
# Block <-> token packing
# =========================
def token2block_from_plan(plan: BlockPlan, L: int) -> torch.LongTensor:
    if int(plan.lens.sum().item()) != L:
        raise ValueError(f"Plan sum {int(plan.lens.sum().item())} != L={L}")
    t2b = torch.empty((L,), dtype=torch.long, device=plan.lens.device)
    for bi in range(int(plan.lens.numel())):
        s = int(plan.starts[bi].item())
        l = int(plan.lens[bi].item())
        t2b[s:s+l] = bi
    return t2b

def tokens_to_blocks(
    tokens: torch.LongTensor,   # [B, Ltot]
    plan: BlockPlan,
    Bmax: int,
    pad_id: int,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]:
    """
    returns:
      blocks:      [B, Ntotal, Bmax]
      block_lens:  [B, Ntotal]
      block_mask:  [B, Ntotal, Bmax]
    """
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
        if s + l > Ltot:
            raise ValueError(f"Block {i} out of range: start={s}, len={l}, Ltot={Ltot}")
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


# Only for testing
def string_to_char_tokens(text: str) -> Tuple[torch.LongTensor, List[str]]:
    """
    Demo tokenizer: character-level.
    Returns:
      tokens: [L] int64
      vocab: list mapping id->char
    """
    chars = list(text)
    # Keep a stable mapping
    uniq = sorted(set(chars))
    char_to_id = {c: i for i, c in enumerate(uniq)}
    ids = torch.tensor([char_to_id[c] for c in chars], dtype=torch.long)
    return ids, uniq


def char_tokens_to_string(tokens_1d: torch.LongTensor, vocab: List[str]) -> str:
    return "".join(vocab[int(i)] for i in tokens_1d.tolist())


def pretty_print_blocks_as_text(
    tokens_1d: torch.LongTensor,
    plan: BlockPlan,
    levels: torch.LongTensor,
    vocab: List[str],
    L_single: int,
    block_sizes: List[int],
) -> None:
    """
    Print each level's blocks as substrings.
    """
    K = len(block_sizes)
    assert tokens_1d.ndim == 1
    assert int(plan.lens.sum().item()) == tokens_1d.numel()

    print("\n=== Pretty print: per-level blocks ===")
    for k in range(K):
        seg_start = k * L_single
        seg_end = (k + 1) * L_single
        level_text = char_tokens_to_string(tokens_1d[seg_start:seg_end], vocab)
        print(f"\n[Level {k}] block_size={block_sizes[k]} | text='{level_text}'")

        block_ids = torch.nonzero(levels == k, as_tuple=False).squeeze(-1).tolist()
        for bi, gb in enumerate(block_ids):
            s = int(plan.starts[gb].item())
            l = int(plan.lens[gb].item())
            block_text = char_tokens_to_string(tokens_1d[s:s + l], vocab)
            print(f"  block {bi:02d}: start={s:03d} len={l:02d} | '{block_text}'")


if __name__ == "__main__":
    print("=== Running multi-level repeat+block tests ===")
    torch.set_printoptions(linewidth=140)

    text = "xxxxxxxxxxxxxxxxxyyyyyyyyyyyyyyyyyyyyyyyyyyzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
    block_sizes = [4, 8, 16] 

    # 1) char-level tokens
    single_tokens_1d, vocab = string_to_char_tokens(text)
    L_single = int(single_tokens_1d.numel())
    K = len(block_sizes)
    Ltot = K * L_single

    # 2) repeat K times (exact copies)
    tokens_1d = single_tokens_1d.repeat(K)  # [Ltot]
    tokens = tokens_1d.unsqueeze(0)         # [B=1, Ltot]

    # 3) build multi-level plan
    plan, levels = make_multilevel_block_plan_for_repeated_sequence(
        L_single=L_single,
        block_sizes=block_sizes,
        device=tokens.device,
    )

    # invariants
    assert int(plan.lens.sum().item()) == Ltot
    assert plan.starts.numel() == plan.lens.numel() == levels.numel()
    assert plan.starts[0].item() == 0
    assert torch.all(plan.starts[1:] >= plan.starts[:-1])  # non-decreasing global starts

    print(f"\nText length (single) L_single={L_single}, repeats K={K}, total Ltot={Ltot}")
    print("Global plan: Ntotal blocks =", int(plan.lens.numel()))
    print("First 20 lens  :", plan.lens[:20].tolist())
    print("First 20 starts:", plan.starts[:20].tolist())
    print("First 20 levels:", levels[:20].tolist())

    # 4) pack/unpack (roundtrip)
    Bmax = int(max(block_sizes))  # safe here: max block size across levels
    pad_id = -1

    blocks, block_lens, block_mask = tokens_to_blocks(tokens=tokens, plan=plan, Bmax=Bmax, pad_id=pad_id)
    tokens_rec = blocks_to_tokens(blocks=blocks, plan=plan, Bmax=Bmax)

    assert torch.equal(tokens, tokens_rec), "Roundtrip tokens->blocks->tokens failed"

    # 5) check: within each level, blocks end exactly at sentence end (last block may be short)
    # We verify the blocks in that level cover exactly [k*L_single, (k+1)*L_single)
    for k, b in enumerate(block_sizes):
        ids = torch.nonzero(levels == k, as_tuple=False).squeeze(-1)
        starts_k = plan.starts[ids]
        lens_k = plan.lens[ids]
        # coverage check: first block starts at k*L_single
        assert int(starts_k[0].item()) == k * L_single
        # last block ends at (k+1)*L_single
        last_end = int((starts_k[-1] + lens_k[-1]).item())
        assert last_end == (k + 1) * L_single

        # all blocks except last should have len==b, last <= b
        if lens_k.numel() > 1:
            assert torch.all(lens_k[:-1] == b)
        assert int(lens_k[-1].item()) <= b

    # 6) validate mask correctness per block
    Ntotal = int(plan.lens.numel())
    for i in range(Ntotal):
        l = int(plan.lens[i].item())
        assert torch.all(block_mask[:, i, :l] == True)
        assert torch.all(block_mask[:, i, l:] == False)
        # padding area should be pad_id
        if l < Bmax:
            assert torch.all(blocks[:, i, l:] == pad_id)

    print("All invariants passed: plan coverage OK, roundtrip OK, mask OK")

    # 7) pretty print blocks as strings so you can eyeball it
    pretty_print_blocks_as_text(
        tokens_1d=tokens_1d,
        plan=plan,
        levels=levels,
        vocab=vocab,
        L_single=L_single,
        block_sizes=block_sizes,
    )

    print("-------------\nAll tests passed!")
