# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin wrapper exposing the optimized KV cache manager implementation."""

try:  # pragma: no cover - depends on optional cython build
    from vllm.v1.core._kv_cache_manager import KVCacheBlocks, KVCacheManager
except ImportError:  # Fallback to the pure Python implementation
    from vllm.v1.core._kv_cache_manager_py import (  # type: ignore
        KVCacheBlocks,
        KVCacheManager,
    )

__all__ = ["KVCacheBlocks", "KVCacheManager"]
