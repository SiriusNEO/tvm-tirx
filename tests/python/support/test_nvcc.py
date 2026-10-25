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
"""Tests for functions in tvm/python/tvm/support/nvcc.py."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import tvm_tirx.testing
from tvm_tirx.support import nvcc


def _make_cuda_root(root, triples):
    """Create a fake CUDA toolkit exposing the given ``targets/<triple>`` dirs."""
    for triple in triples:
        os.makedirs(os.path.join(root, "targets", triple, "include"))
    return str(root)


@pytest.mark.parametrize(
    "machine,available,expected",
    [
        # ARM64 server toolkits ship the headers under "sbsa-linux".
        ("aarch64", ["sbsa-linux", "aarch64-linux"], "sbsa-linux"),
        # Embedded/L4T toolkits only provide "aarch64-linux".
        ("aarch64", ["aarch64-linux"], "aarch64-linux"),
        ("arm64", ["sbsa-linux"], "sbsa-linux"),
        ("x86_64", ["x86_64-linux"], "x86_64-linux"),
    ],
)
def test_find_cuda_target_include(tmp_path, monkeypatch, machine, available, expected):
    """The architecture-specific include dir matches the installed toolkit layout."""
    monkeypatch.setattr(nvcc.platform, "machine", lambda: machine)
    monkeypatch.setattr(nvcc.platform, "system", lambda: "Linux")
    cuda_path = _make_cuda_root(tmp_path, available)
    assert nvcc._find_cuda_target_include(cuda_path) == os.path.join(
        cuda_path, "targets", expected, "include"
    )


def test_find_cuda_target_include_absent(tmp_path, monkeypatch):
    """Toolkits without a ``targets/`` layout report no architecture-specific dir."""
    monkeypatch.setattr(nvcc.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(nvcc.platform, "system", lambda: "Linux")
    assert nvcc._find_cuda_target_include(str(tmp_path)) is None


@pytest.mark.parametrize("arch", ["sm_90a", "sm_100a", "sm_103a", "sm_100f"])
@pytest.mark.parametrize("output_format", ["fatbin", "ptx", "cubin"])
def test_nvcc_preserves_architecture_qualifiers(monkeypatch, arch, output_format):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"test-compiler-output")
        return SimpleNamespace(returncode=0, communicate=lambda: (b"", None))

    monkeypatch.setattr(nvcc.subprocess, "Popen", run)
    artifact, _ = nvcc._compile_cuda_nvcc("", output_format, arch)
    assert artifact == b"test-compiler-output"
    command = commands[0]
    virtual_arch = "compute_" + arch.removeprefix("sm_")
    if output_format == "fatbin":
        assert command[command.index("-gencode") + 1] == (
            f"arch={virtual_arch},code=[{arch},{virtual_arch}]"
        )
        assert "-arch" not in command
    elif output_format == "ptx":
        assert command[command.index("-arch") + 1] == virtual_arch
        assert command[command.index("-code") + 1] == virtual_arch
    else:
        assert command[command.index("-arch") + 1] == arch


if __name__ == "__main__":
    tvm_tirx.testing.main()
