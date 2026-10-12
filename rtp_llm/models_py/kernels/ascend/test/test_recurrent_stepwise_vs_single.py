#!/usr/bin/env python3
"""D5 裁决：GDN recurrent 多 token —— 本地单 launch vs PR57 逐步 T==1。

背景（见 mtp_plan_doc/04_与上游PR57_MTP实现对比.md D5）：
  - 本地 `_fused_recurrent_tokens_device`：种子首个写页 + 一次
    npu_recurrent_gated_delta_rule launch（actual_seq=[0,T,T,...]，全部 B*T 写页）。
  - PR57 `_fused_recurrent_multi_device`：逐步 T==1 launch（上游 fla-npu 的
    多 token 单 launch 据称会写错中间页的 BF16 snapshot）。

本脚本在真实 NPU 上，对同一组输入（含跨页 read→write 与页间链式写）跑两种
路径，比较每 token 输出与最终状态是否位级一致；另给一个 fp32 顺序参考做量级
sanity check。

用法（容器内，需 NPU）:
    cd /home/b30086708/code/rtp_llm/rtp-llm-nb
    python3 rtp_llm/models_py/kernels/ascend/test/test_recurrent_stepwise_vs_single.py
"""

import math
import sys

import torch

try:
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
except Exception as e:  # noqa: BLE001
    print(f"SKIP: NPU unavailable ({e})")
    sys.exit(0)

from rtp_llm.models_py.kernels.ascend.paged_state import (  # noqa: E402
    seed_state_segment,
    speculative_state_indices,
)
from rtp_llm.models_py.kernels.ascend.recurrent import (  # noqa: E402
    _get_ascendc_ops,
    fused_recurrent_gated_delta_rule,
)

DEVICE = "npu"
PAGE = 512
B, T = 3, 4          # batch, verify tokens (= K+1, K=3)
H, DK, DV = 4, 128, 128
SCALE = DK ** -0.5
PAGES = 6 * B + 1   # block_map 用 1..6B 的页号（0 为 null），池需覆盖到 6B


def make_case(seed: int):
    g = torch.Generator().manual_seed(seed)
    # block_map: 每行 6 个互不相同的物理页号（0 为 null，不用）
    base = torch.arange(1, 6 * B + 1).reshape(B, 6)
    perm = torch.randperm(6 * B, generator=g)
    block_map = base.flatten()[perm].reshape(B, 6).to(torch.int32).to(DEVICE)
    # len ≡ 1 (mod PAGE) 的行会跨页（read_col = write_col0 - 1）
    lens = [1025, 600, 77]
    seq_plus_1 = torch.tensor(lens, dtype=torch.int32, device=DEVICE)
    q = torch.randn(B, T, H, DK, generator=g).to(DEVICE).to(torch.bfloat16)
    k = torch.randn(B, T, H, DK, generator=g).to(DEVICE).to(torch.bfloat16)
    v = torch.randn(B, T, H, DV, generator=g).to(DEVICE).to(torch.bfloat16)
    beta = torch.rand(B, T, H, generator=g).to(DEVICE).to(torch.bfloat16)
    gg = torch.rand(B, T, H, generator=g).to(DEVICE).to(torch.float32)
    state0 = (torch.randn(PAGES, H, DV, DK, generator=g) * 0.5).to(DEVICE).to(torch.bfloat16)
    return block_map, seq_plus_1, q, k, v, beta, gg, state0


def run_single_launch(block_map, seq_plus_1, q, k, v, beta, gg, state0):
    """本地路径：fused_recurrent_gated_delta_rule 的 device 多 token 单 launch。"""
    state = state0.clone()
    out, _ = fused_recurrent_gated_delta_rule(
        q, k, v, gg, beta, scale=SCALE,
        initial_state=state, inplace_final_state=True,
        block_map=block_map, sequence_lengths=seq_plus_1,
        seq_size_per_block=PAGE,
        use_qk_l2norm_in_kernel=False,
    )
    return out, state


def run_stepwise(block_map, seq_plus_1, q, k, v, beta, gg, state0):
    """PR57 路径：逐步 T==1 launch（_fused_recurrent_multi_device 逻辑）。"""
    ascendc = _get_ascendc_ops()
    state = state0.clone()
    read_idx, write_idx = speculative_state_indices(
        block_map, seq_plus_1, PAGE, T
    )  # read [B], write [B, T]
    asl = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=DEVICE),
         torch.ones(B, dtype=torch.int32, device=DEVICE)]
    )
    outs = []
    for t in range(T):
        src = read_idx if t == 0 else write_idx[:, t - 1]
        seed_state_segment(state, src, write_idx[:, t])
        result = ascendc.npu_recurrent_gated_delta_rule(
            q[:, t].reshape(-1, *q.shape[2:]).to(torch.bfloat16),
            k[:, t].reshape(-1, *k.shape[2:]).to(torch.bfloat16),
            v[:, t].reshape(-1, *v.shape[2:]).to(torch.bfloat16),
            state,
            beta=beta[:, t].reshape(-1, beta.shape[-1]).to(torch.bfloat16),
            scale=float(SCALE),
            actual_seq_lengths=asl,
            ssm_state_indices=write_idx[:, t].to(torch.int32),
            g=gg[:, t].reshape(-1, gg.shape[-1]).float(),
        )
        out_t = result[0] if isinstance(result, (tuple, list)) else result
        outs.append(out_t.reshape(B, 1, *out_t.shape[1:]))
    return torch.cat(outs, dim=1), state


def fp32_reference(block_map, seq_plus_1, q, k, v, beta, gg, state0):
    """fp32 顺序参考（delta rule 逐步，state 布局 [pages, H, DV, DK]）。

    每 head：S' = S*exp(-g)；u = S' @ k；S_new = S' + (v - u)⊗(β·k)。
    仅作量级 sanity（被测 kernel 内部 bf16/fp32 混合，允许小误差）。
    """
    read_idx, write_idx = speculative_state_indices(
        block_map, seq_plus_1, PAGE, T
    )
    state = state0.float().clone()  # [PAGES, H, DV, DK]
    outs = []
    for b in range(B):
        cur = state[read_idx[b]].float().clone()  # [H, DV, DK]
        for t in range(T):
            new = torch.empty_like(cur)
            o = torch.empty(H, DV, device=state.device)
            for h in range(H):
                s = cur[h] * math.exp(-float(gg[b, t, h]))
                kdir = k[b, t, h].float()
                u = s @ kdir                            # [DV]
                new[h] = s + torch.outer(v[b, t, h].float() - u, beta[b, t, h].float() * kdir)
                o[h] = v[b, t, h].float() - u           # delta rule 输出 = v - u
            cur = new
            state[write_idx[b, t]] = new
            outs.append(o)
    return torch.stack(outs, dim=0).reshape(B, T, H, DV), state


def main():
    print(f"device={DEVICE}  B={B} T={T} H={H} DK={DK} DV={DV} page={PAGE}")
    all_ok = True
    for seed in (7, 1234):
        case = make_case(seed)
        bm, sp1, q, k, v, beta, gg, st0 = case
        out1, st1 = run_single_launch(*case)
        out2, st2 = run_stepwise(*case)
        out_bitexact = bool((out1 == out2).all())
        st_bitexact = bool((st1 == st2).all())
        out_max = (out1.float() - out2.float()).abs().max().item()
        st_max = (st1.float() - st2.float()).abs().max().item()
        # 中间页：单 launch 与逐步在各 write 页上的最终状态（除最后 token 页）
        ref_out, ref_st = fp32_reference(*case)
        ref_gap = (out1.float() - ref_out).abs().max().item()
        print(f"\n[seed {seed}]")
        print(f"  output bit-exact : {out_bitexact}   max|diff|={out_max:.3e}  (vs fp32 ref gap={ref_gap:.3e})")
        print(f"  state  bit-exact : {st_bitexact}   max|diff|={st_max:.3e}")
        # 检查中间写页（非最后 token 的页）——PR57 声称这些页在单 launch 下 snapshot 错误
        read_idx, write_idx = speculative_state_indices(bm, sp1, PAGE, T)
        mid = write_idx[:, :-1].flatten().unique()
        mid_diff = (st1[mid].float() - st2[mid].float()).abs().max().item()
        print(f"  middle pages ({mid.tolist()}) max|diff|={mid_diff:.3e}")
        all_ok &= out_bitexact and st_bitexact
    print("\nRESULT:", "BIT-EXACT — 公共入口（修复后逐步路径）与逐步参考一致"
          if all_ok else "DIVERGENT — 公共入口偏离逐步参考（fla-npu 单 launch snapshot bug 回归？）")


if __name__ == "__main__":
    main()
