# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Codegen registry for CUDA HW ops.

User-facing Python wrappers are hand-written in :mod:`tvm_tirx.tirx.op` so that
editors / static analyzers (Cursor, Pyright) can see their signatures. This
module only handles the backend codegen side.
"""

import functools

import tvm_ffi

CODEGEN_REGISTRY = {}
DTYPE_CODEGEN_REGISTRY = {}


def register_dtype_codegen(dtype, scalar_type: str, source: str):
    """Register a custom dtype's CUDA C++ value type and its definitions."""
    from tvm_tirx import ir

    primitive = dtype if isinstance(dtype, ir.PrimType) else ir.PrimType(dtype)
    ir.custom_dtype_storage(primitive)
    if primitive.dtype.type_code < 129 or primitive.dtype.lanes != 1:
        raise ValueError("CUDA custom dtype registration requires a scalar custom dtype")
    if not scalar_type or not source:
        raise ValueError("CUDA custom dtype registration requires a type and its source")
    key = (primitive.dtype.type_code, primitive.dtype.bits)
    entry = (scalar_type, source)
    if key in DTYPE_CODEGEN_REGISTRY and DTYPE_CODEGEN_REGISTRY[key] != entry:
        raise ValueError(f"CUDA custom dtype is already registered: {dtype}")
    DTYPE_CODEGEN_REGISTRY[key] = entry


@tvm_ffi.register_global_func("tirx.intrinsics.cuda.get_dtype_codegen")
def get_dtype_codegen(dtype):
    """Look up a custom dtype's CUDA scalar representation."""
    return list(DTYPE_CODEGEN_REGISTRY[(dtype.type_code, dtype.bits)])


@tvm_ffi.register_global_func("tirx.intrinsics.cuda.get_codegen")
def get_codegen(op):
    """get the codegen function for a given op"""
    return CODEGEN_REGISTRY.get(op, None)


def register_codegen(op, backend="cuda", *, namespace="tirx"):
    """Register a codegen function for a given op.

    The codegen function should return a ``func_call`` statement, and
    optionally a list of tags that the codegen function needs.
    ``namespace`` selects the owning dialect; it defaults to ``tirx``.
    """

    def decorator(func):
        full_op_name = namespace + "." + op

        @functools.wraps(func)
        def wrapper(arg_list, attrs=None):
            # Keep operand-only helpers callable directly while forwarding the
            # typed static options of an attributed Call to its lowering.
            kwargs = {} if attrs is None else {"attrs": attrs}
            res = func(*arg_list, **kwargs)  # pylint: disable=not-callable
            if isinstance(res, tuple):
                return res[0], res[1]
            return res, list()

        CODEGEN_REGISTRY[full_op_name] = wrapper
        return wrapper

    return decorator
