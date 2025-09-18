# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin wrapper exposing the optimized KV cache coordinator."""

try:  # pragma: no cover - depends on optional cython build
    from vllm.v1.core._kv_cache_coordinator import (  # type: ignore
        CrossAttentionManager,
        FullAttentionManager,
        KVCacheCoordinator,
        KVCacheCoordinatorNoPrefixCache,
        KVCacheCoordinatorWithPrefixCache,
        get_kv_cache_coordinator,
    )
except ImportError:  # Fallback to the pure Python implementation
    from vllm.v1.core._kv_cache_coordinator_py import (  # type: ignore
        CrossAttentionManager,
        FullAttentionManager,
        KVCacheCoordinator,
        KVCacheCoordinatorNoPrefixCache,
        KVCacheCoordinatorWithPrefixCache,
        get_kv_cache_coordinator,
    )

__all__ = [
    "KVCacheCoordinator",
    "KVCacheCoordinatorNoPrefixCache",
    "KVCacheCoordinatorWithPrefixCache",
    "FullAttentionManager",
    "CrossAttentionManager",
    "get_kv_cache_coordinator",
]
