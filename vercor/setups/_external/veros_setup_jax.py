"""Minimal setup adapters for the differentiable Veros execution lane."""

from __future__ import annotations

from veros import veros_routine

from vercor.setups._external.veros_setup import CustomGlobalFourDegree


class DifferentiableGlobalFourDegree(CustomGlobalFourDegree):
    """Externally forced global setup with a fixed runtime PyTree structure."""

    @veros_routine
    def set_diagnostics(self, state):  # type: ignore[no-untyped-def]
        """Disable diagnostics whose lazy accumulators change the state PyTree."""

        state.diagnostics.clear()


__all__ = ["DifferentiableGlobalFourDegree"]
