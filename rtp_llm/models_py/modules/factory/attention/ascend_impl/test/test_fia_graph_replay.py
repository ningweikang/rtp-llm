#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Level 2-FIA：target-verify FIA 图捕获 → 回放 vs eager（PR #60 手搓 task update）。

验证 AscendPrefillAttnOp 的 FIA 图内刷新机制，流程与引擎严格同序
（AscendGraphRunner::prepareInputs → prepare_cuda_graph → replayGraph）：

  1. 捕获：_forward_fia_graph（ExternalEvent 等待点 + graph_task_group +
     捕获期占位 kvlen + 共享 workspace）在 torch.npu.graph 下干净捕获；
  2. 回放：先 op.prepare + update_graph_fia(真实 kv_lens)（kv_lens 计算镜像
     AscendPrefillImpl.prepare_cuda_graph）再 replay，输出必须与同输入同元数据
     的 eager 调用位级一致（PR #57 曾记录手搓变体错误回放——本脚本即为
     PR #60 手搓变体在 verify 形状下的直接裁决）；
  3. 灵敏度（负对照）：只 signal_graph_events 不 update（等价 C++ 侧捕获后
     的 sanity replay，跑在占位 kvlen 上）时，输出必须偏离 eager——证明真实
     上下文长度确实由 task update 机制携带，测试本身有分辨力；
  4. 未烤死：不同 case（q/prefix/kv_lens 不同）的 replay 输出不同。

用法（容器内，需 NPU；py310_bsy_env）：
    cd /home/b30086708/code/rtp_llm/rtp-llm-nb
    python3 rtp_llm/models_py/modules/factory/attention/ascend_impl/test/test_fia_graph_replay.py
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

try:
    import torch_npu  # noqa: F401

    _HAS_NPU = torch.npu.is_available()
except Exception as e:  # noqa: BLE001
    print(f"SKIP: NPU unavailable ({e})")
    sys.exit(0)

from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_prefill import (  # noqa: E402
    AscendPrefillAttnOp,
)

DEV = "npu"
B, T = 2, 3                    # 流数、每流 token 数（= K+1，K=2 与引擎配置一致）
NUM_TOKENS = B * T
NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM = 8, 4, 128
KERNEL_PAGE = 512              # 池的 kernel-block 粒度（= FIA block_size）
PAGES, COLS = 64, 4
DTYPE = torch.bfloat16

# name, prefix_lens, input_lens, block_table rows, seed, valid_rows(None=全部)
CASES = [
    ("c1-cross",  [100, 700],  [3, 3], [[5, 6, 7, 0], [10, 11, 12, 13]], 11, None),
    ("c2-cross2", [512, 1300], [3, 3], [[5, 6, 7, 0], [10, 11, 12, 13]], 22, None),
    ("c3-padrow", [900, 0],    [3, 0], [[5, 6, 7, 0], [10, 11, 12, 13]], 33, T),
]


def build_pool(seed: int):
    gen = torch.Generator().manual_seed(seed)
    stale = (
        torch.randn(
            PAGES, 2, KERNEL_PAGE, NUM_KV_HEADS, HEAD_DIM, generator=gen
        )
        * 5.0
    ).to(DTYPE)
    return SimpleNamespace(
        kv_cache_base=stale.to(DEV), seq_size_per_block=KERNEL_PAGE
    )


def make_attn_configs():
    return SimpleNamespace(
        head_num=NUM_Q_HEADS,
        kv_head_num=NUM_KV_HEADS,
        size_per_head=HEAD_DIM,
        q_scaling=1.0,
        tokens_per_block=KERNEL_PAGE * 2,
        kernel_tokens_per_block=KERNEL_PAGE,
    )


def mirror_prepare_cuda_graph_kv_lens(pl: torch.Tensor, il: torch.Tensor, graph_bs: int):
    """镜像 AscendPrefillImpl.prepare_cuda_graph 的 host 计算（prefix+input，
    pad 行 ctx=1，截到捕获 bucket 大小）。"""
    valid = int((il > 0).sum().item()) if il.numel() else 0
    valid = max(valid, 1)
    kv = (pl[:valid] + il[:valid]).to(torch.int32).tolist()
    return kv[:graph_bs] + [1] * max(0, graph_bs - len(kv))


def run_eager(cfg, kv_cache, q, pl, il, rows, valid_rows=None):
    """eager 参考：同输入同元数据走 AscendPrefillAttnOp 的 eager 分支。"""
    if valid_rows is not None:
        pl, il, rows, q = pl[:1], il[:1], [rows[0]], q[:valid_rows]
    eager_inputs = SimpleNamespace(
        kv_cache=SimpleNamespace(seq_size_per_block=KERNEL_PAGE),
        kv_cache_kernel_block_id_host=torch.tensor(rows, dtype=torch.int32),
        input_lengths=torch.tensor(il, dtype=torch.int32),
        prefix_lengths=torch.tensor(pl, dtype=torch.int32),
    )
    op = AscendPrefillAttnOp(cfg, eager_inputs)
    op.prepare(eager_inputs)
    out = op.forward(q, kv_cache)
    torch.npu.synchronize()
    return out


def main():
    torch.npu.set_device(0)
    print(f"device={DEV}  B={B} T={T} heads={NUM_Q_HEADS}/{NUM_KV_HEADS} "
          f"dim={HEAD_DIM} kernel_page={KERNEL_PAGE}")

    cfg = make_attn_configs()
    kv_cache = build_pool(seed=42)

    q = torch.zeros(NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)
    bt_dev = torch.zeros(B, COLS, dtype=torch.int32, device=DEV)
    pl_host = torch.zeros(B, dtype=torch.int32)
    il_host = torch.zeros(B, dtype=torch.int32)

    def set_case(pl, il, rows, seed):
        torch.manual_seed(seed)
        q.copy_(torch.randn(q.shape).to(DTYPE))
        pl_host.copy_(torch.tensor(pl, dtype=torch.int32))
        il_host.copy_(torch.tensor(il, dtype=torch.int32))
        bt_dev.copy_(torch.tensor(rows, dtype=torch.int32))

    graph_inputs = SimpleNamespace(
        is_cuda_graph=True,
        kv_cache=SimpleNamespace(seq_size_per_block=KERNEL_PAGE),
        kv_cache_kernel_block_id_device=bt_dev,
        input_lengths=il_host,
        prefix_lengths=pl_host,
    )
    op_graph = AscendPrefillAttnOp(cfg, graph_inputs)
    op_graph.prepare(graph_inputs)

    # ---- warmup（触发 aclnn opapi 加载 / 首次编译，捕获外完成）----
    set_case(*CASES[0][1:5])
    for _ in range(2):
        op_graph.forward(q, kv_cache)  # 未捕获 → eager 分支派生 actual_seq
    torch.npu.synchronize()

    # ---- capture（引擎等价：C++ aclgraph 捕获 verify 前向中的 FIA 任务）----
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        out = op_graph.forward(q, kv_cache)
    torch.npu.synchronize()
    print("[fia-graph] capture OK（ExternalEvent 等待点 + graph_task_group 均在图内）")

    passed, total, outs = 0, len(CASES), {}
    for name, pl, il, rows, seed, valid_rows in CASES:
        set_case(pl, il, rows, seed)

        # 引擎同序：prepareInputs（in-place 刷缓冲）→ prepare_cuda_graph → replay
        op_graph.prepare(graph_inputs)
        kv_lens = mirror_prepare_cuda_graph_kv_lens(pl_host, il_host, B)
        op_graph.update_graph_fia(kv_lens, B)
        g.replay()
        torch.npu.synchronize()
        out_r = out.clone()

        out_e = run_eager(cfg, kv_cache, q, pl, il, rows, valid_rows)

        n = valid_rows if valid_rows is not None else NUM_TOKENS
        diff = (out_r[:n].float() - out_e.float()).abs().max().item()
        ok = diff == 0.0
        print(f"[fia {name:<11}] kv_lens={kv_lens} out_diff={diff:.3e} "
              f"({'位级一致' if ok else 'DIVERGENT'}) -> {'PASS' if ok else 'FAIL'}")
        passed += ok
        outs[name] = out_r

    # ---- 未烤死：不同 case 的 replay 输出必须不同 ----
    if "c1-cross" in outs and "c2-cross2" in outs:
        d = (outs["c1-cross"].float() - outs["c2-cross2"].float()).abs().max().item()
        print(f"[fia not-baked ] c1 vs c2 max diff = {d:.3e}（应 > 0）"
              f" -> {'OK' if d > 0 else 'FAIL: 输出被烤死!'}")
        if d <= 0:
            passed -= 1

    # ---- 灵敏度负对照：只 signal 不 update（占位 kvlen 上回放）必须偏离 eager ----
    name, pl, il, rows, seed, _ = CASES[0]
    set_case(pl, il, rows, seed)
    op_graph.signal_graph_events(B)  # 仅释放等待点，不刷新元数据（sanity replay 语义）
    g.replay()
    torch.npu.synchronize()
    out_stale = out.clone()
    out_e = run_eager(cfg, kv_cache, q, pl, il, rows)
    d_stale = (out_stale.float() - out_e.float()).abs().max().item()
    print(f"[fia neg-ctrl  ] stale(占位 kvlen=[{KERNEL_PAGE}]*{B}) vs eager "
          f"max diff = {d_stale:.3e}（应显著 > 0，证明 kvlen 由 update 携带）"
          f" -> {'OK' if d_stale > 1e-2 else 'FAIL: 测试无分辨力!'}")

    verdict = "PASS — FIA 图回放与 eager 位级一致，task update 机制验证通过" if passed == total else "FAIL"
    print(f"\nRESULT: {verdict}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
