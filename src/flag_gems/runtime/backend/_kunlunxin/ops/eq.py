# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

try:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor
    _HAS_TLE = True
except ImportError:
    tle = None
    TensorDescriptor = None
    _HAS_TLE = False

from flag_gems.ops.eq_ import eq_ as _generic_eq_
from flag_gems.ops.eq_ import eq_scalar_ as _generic_eq_scalar_
from flag_gems.runtime import device

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
device = device.name

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)

# Memory-side candidate (2026-09-15): the generic pointwise_dynamic scalar
# compare path DMAs only 1024-element LM tiles (gm2lm 2KB in / lm2gm 1KB out)
# bracketed by 3 mfences per tile (confirmed in the bf16 [10000,65536] LLVM
# dump). On the memory-bound large shapes this leaves gems at ~43% of torch
# bandwidth. Enlarging the LM buffer + unroll widens each DMA burst and
# amortizes the fence/setup overhead. Scoped to eq_func_scalar only so the
# known-good tensor eq() path (config_) is untouched.
config_scalar_bigtile_ = CodeGenConfig(
    1024,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    buffer_size_limit=16384,
    unroll_num=16,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def eq_func(x, y):
    return x.to(tl.float32) == y.to(tl.float32)


def eq(A, B):
    if A.device != B.device:
        if A.device.type == device:
            B = B.to(A.device)
        else:
            A = A.to(B.device)
    logger.debug("GEMS_KUNLUNXIN EQ")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = eq_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_scalar_bigtile_,
)
@triton.jit
def eq_func_scalar(x, y):
    return x.to(tl.float32) == y



def eq_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN EQ_SCALAR")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    try:
        return eq_func_scalar(A, B)
    finally:
        os.environ.pop("TRITONXPU_COMPARE_FUSION", None)
        os.environ.pop("TRITONXPU_FP16_FAST", None)


@triton.jit
def eq_func_tensor_inplace(x, y):
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = tl.abs(t)
    t = tl.minimum(1.0, t)
    t = tl.maximum(0.0, 1.0 - t)
    return t


def eq_(A, B):
    logger.debug("GEMS_KUNLUNXIN EQ_ TENSOR")
    if A.device != B.device:
        B = B.to(A.device)
    numel = A.numel()
    if A.is_contiguous() and A.dtype in (torch.float16, torch.float32, torch.bfloat16):
        if (
            A.dtype in (torch.float16, torch.float32)
            and B.is_contiguous()
            and B.dtype == A.dtype
            and A.shape == B.shape
            and numel >= _EQ_TENSOR_INPLACE_FAST_TILE * _EQ_TENSOR_INPLACE_MIN_GRID
            and numel % _EQ_TENSOR_INPLACE_FAST_TILE == 0
        ):
            # exact-multiple flat tiles: no mask at all; grid fixed.
            return _eq_tensor_inplace_fast(A, B, numel)
        eq_func_tensor_inplace(A, B, out0=A)
        return A
    # Everything else (non-float dtype, non-contiguous, ...) keeps the
    # original generic in-place path, behavior unchanged.
    return _generic_eq_(A, B)


# in-place alias safety: the fast kernel writes into the SAME tensor it
# reads, so it must keep the DEFAULT isCloseMemoryAsync (True = async copy
# closed); passing False with in-place aliasing is the documented "noc idle
# timeout" deadlock, same as gt.py's _gt_scalar_inplace_fast note.
_EQ_TENSOR_INPLACE_FAST_TILE = 131072
_EQ_TENSOR_INPLACE_MIN_GRID = 128


@triton.jit
def eq_tensor_inplace_fast_kernel(x_ptr, y_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    y = tl.load(y_ptr + tid)
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = tl.abs(t)
    t = tl.minimum(1.0, t)
    t = tl.maximum(0.0, 1.0 - t)
    tl.store(x_ptr + tid, t)


def _eq_tensor_inplace_fast(A, B, numel):
    grid = (numel // _EQ_TENSOR_INPLACE_FAST_TILE,)
    eq_tensor_inplace_fast_kernel[grid](
        A,
        B,
        TILE=_EQ_TENSOR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A


# ---------------------------------------------------------------------------
# eq_scalar_ (in-place alias of eq.Scalar, e.g. `x.eq_(0)` on a float tensor).
# torch keeps the input dtype and stores 0.0/1.0 (False/True) back into x.
#
# Before this change eq_scalar_ was NOT overridden by the kunlunxin backend,
# so it fell to the generic ops/eq_.py wrapper (promotion ALWAYS_BOOL;
# `arith.cmpf -> i1 -> bool` per lane, out0=A). On XPU the big shapes hit the
# documented slow path: [10000,65536] fp16 gems ~1.59s vs torch ~1.43s and
# the dtype-equal-weight baseline measured 0.0446x (fp16 0.0349 / fp32
# 0.0606 / bf16 0.0383, 2026-08-16, XPU 2) -- the same catastrophic
# ALWAYS_BOOL path the eq_/gt_/lt_ in-place family fixed with the saturating
# fp recipe.
#
# Fix (the exact in-place scalar recipe of gt_scalar_ / le_scalar_ family,
# mirrored from _eq_tensor_inplace_ above):
#   1. generic in-place kernel `eq_func_scalar_inplace` (is_tensor=[True,
#      False], DEFAULT promotion, no i1 materialized) under the in-place-safe
#      config_inplace_ (async copy closed — in-place aliasing + async streams
#      is the documented "noc idle timeout" deadlock, see note above).
#   2. unmasked flat-tile in-place fast kernel for fp16/fp32 contiguous
#      tensors whose numel is an exact multiple of TILE (grid >= MIN_GRID):
#      no runtime mask -> fast DMA path.
#   3. scalar-representability gate (wrapped-scalar semantics, same as
#      gt_scalar_): `float(B) == float(torch.tensor(float(B), dtype=A.dtype))`
#      plus a finite check on the wrapped value. The fp32 body then compares
#      bit-identically to torch's wrapped-scalar comparison (e.g. fp16 x vs
#      0.5). Non-representable scalars (fp16 0.1, bf16 0.001, fp32 pi), the
#      +/-inf/NaN scalar corners (x == +/-inf must stay exact), non-float
#      dtypes, and non-contiguous layouts keep the previous generic
#      ALWAYS_BOOL in-place path (forwarded to _generic_eq_scalar_), behavior
#      unchanged.
#   4. body: saturation of the *distance* (equality has no gap direction):
#        t   = min(1, |x - s| * 1e32 * 1e32)   -> 0 when x == s, 1 when x != s
#        out = max(0, 1 - t)                   -> 1 when equal, 0 otherwise
#      Every representable nonzero gap (down to the fp32/bf16 subnormals
#      ~1.4e-45) saturates to >= 1. The slow i1/bool path is never
#      materialized on the fast/saturate paths.
config_inplace_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)

@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_inplace_,
)
@triton.jit
def eq_func_scalar_inplace(x, y):
    t = (x.to(tl.float32) - y) * 1.0e32
    t = t * 1.0e32
    t = tl.abs(t)
    t = tl.minimum(1.0, t)
    t = tl.maximum(0.0, 1.0 - t)
    return t


def eq_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN EQ_ SCALAR")
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == float(torch.tensor(float(B), dtype=dtype).item())
    ):
        wrapped = float(torch.tensor(float(B), dtype=dtype).item())
        if math.isfinite(wrapped):
            if (
                dtype in (torch.float16, torch.float32)
                and numel >= _EQ_SCALAR_INPLACE_FAST_TILE * _EQ_SCALAR_INPLACE_MIN_GRID
                and numel % _EQ_SCALAR_INPLACE_FAST_TILE == 0
            ):
                # exact-multiple flat tiles: no mask at all; grid fixed.
                return _eq_scalar_inplace_fast(A, wrapped)
            eq_func_scalar_inplace(A, B, out0=A)
            return A
    # Everything else (non-representable scalar, non-finite scalar, non-float
    # dtype, non-contiguous, ...) keeps the original generic in-place path,
    # behavior unchanged.
    return _generic_eq_scalar_(A, B)


# in-place alias safety: the fast kernel writes into the SAME tensor it reads,
# so it must keep isCloseMemoryAsync=True (async copy closed); passing False
# with in-place aliasing is the documented "noc idle timeout" deadlock, same
# as _eq_tensor_inplace_fast above.
_EQ_SCALAR_INPLACE_FAST_TILE = 131072
_EQ_SCALAR_INPLACE_MIN_GRID = 128


@triton.jit
def eq_scalar_inplace_fast_kernel(x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    t = (x.to(tl.float32) - scalar) * 1.0e32
    t = t * 1.0e32
    t = tl.abs(t)
    t = tl.minimum(1.0, t)
    t = tl.maximum(0.0, 1.0 - t)
    tl.store(x_ptr + tid, t)


def _eq_scalar_inplace_fast(A, scalar):
    grid = (A.numel() // _EQ_SCALAR_INPLACE_FAST_TILE,)
    eq_scalar_inplace_fast_kernel[grid](
        A,
        scalar,
        TILE=_EQ_SCALAR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A
