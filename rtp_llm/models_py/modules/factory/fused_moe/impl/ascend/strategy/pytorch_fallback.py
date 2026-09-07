"""Ascend MoE placeholder strategy: MoE is rejected on Ascend for now.

No NPU-capable MoE executor exists yet. The previously wired
BatchedTritonExperts imports Triton and runs Triton grouped GEMM, but Triton
is excluded from the Ascend dependency set (arch_config/arch_select.bzl
_ascend_excluded), so it cannot execute on NPU tensors (upstream PR #1349
review r3913106822).

Ascend TP (tp_size > 1) is additionally not implemented: the batched data
router would slice experts by tp_size and drop the tail experts on pure-TP
topologies (review r3913106830).

check_conditions therefore raises for every configuration so that MoE models
fail fast at strategy selection with a clear error instead of crashing later
inside a Triton-dependent executor or silently dropping experts. Dense
(non-MoE) models never reach this code path.
"""

from typing import Any

from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy


class AscendBf16FallbackStrategy(MoeStrategy):
    """Placeholder that rejects all MoE configurations on Ascend."""

    _REJECT_MSG = (
        "Ascend MoE is not supported yet: no NPU-capable MoE executor is "
        "available (Triton-based executors are excluded from Ascend deps), "
        "and Ascend TP (tp_size > 1) is not implemented either. MoE models "
        "are rejected on Ascend; use tp_size=1 dense models or wait for a "
        "native Ascend MoE executor."
    )

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        raise ValueError(cls._REJECT_MSG)

    def get_attributes(self) -> StrategyAttributes:
        # Never reached: can_handle calls check_conditions first, which
        # raises before attributes are consulted.
        raise RuntimeError(self._REJECT_MSG)
