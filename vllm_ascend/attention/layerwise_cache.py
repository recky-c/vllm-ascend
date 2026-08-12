# SPDX-License-Identifier: Apache-2.0
"""Attention-side boundary for layerwise KV-cache services."""

from typing import Protocol


class LayerwiseKVCacheHook(Protocol):
    """Minimal hook consumed by attention implementations.

    The attention data path deliberately knows nothing about KVPP, MemFabric,
    scratch buffers, or the transport schedule. Other layerwise cache services
    can implement the same boundary without changing attention code again.
    """

    def enter_layer(self, layer_name: str) -> None:
        """Begin work that can overlap with this layer's projections."""

    def wait_for_layer(self, layer_name: str) -> None:
        """Make the layer's cache safe to consume on the compute stream."""

    def leave_layer(self, layer_name: str) -> None:
        """Record that all historical-cache consumers were submitted."""
