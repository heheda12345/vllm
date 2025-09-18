# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin wrapper exposing the optimized BlockPool implementation."""

try:  # pragma: no cover - depends on optional cython build
    from vllm.v1.core._block_pool import BlockPool
except ImportError:  # Fallback to the pure Python implementation
    from vllm.v1.core._block_pool_py import BlockPool  # type: ignore

__all__ = ["BlockPool"]
