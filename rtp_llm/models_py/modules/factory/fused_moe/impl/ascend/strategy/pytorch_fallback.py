"""Ascend MoE placeholder: rejects all MoE configurations.

No NPU-capable MoE executor exists for this placeholder path (Triton is
excluded from Ascend deps), and Ascend TP (tp_size > 1) is not implemented.
MoE models fail fast at strategy selection instead of crashing or silently
dropping experts. Dense (non-MoE) models never reach this path.

The BF16 pure-TP CANN pipeline (AscendCannStrategy) supersedes this
placeholder and is the strategy registered in fused_moe/__init__.py; the
class is kept as the fail-fast contract for every other MoE configuration.
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
