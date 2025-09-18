# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin wrapper exposing the optimized single-type KV cache managers."""

try:  # pragma: no cover - depends on optional cython build
    from vllm.v1.core._single_type_kv_cache_manager import (  # type: ignore
        ChunkedLocalAttentionManager,
        CrossAttentionManager,
        FullAttentionManager,
        MambaManager,
        SingleTypeKVCacheManager,
        SlidingWindowManager,
        get_manager_for_kv_cache_spec,
        spec_manager_map,
    )
except ImportError:  # Fallback to the pure Python implementation
    from vllm.v1.core._single_type_kv_cache_manager_py import (  # type: ignore
        ChunkedLocalAttentionManager,
        CrossAttentionManager,
        FullAttentionManager,
        MambaManager,
        SingleTypeKVCacheManager,
        SlidingWindowManager,
        get_manager_for_kv_cache_spec,
        spec_manager_map,
    )

__all__ = [
    "SingleTypeKVCacheManager",
    "FullAttentionManager",
    "SlidingWindowManager",
    "ChunkedLocalAttentionManager",
    "MambaManager",
    "CrossAttentionManager",
    "get_manager_for_kv_cache_spec",
    "spec_manager_map",
]
