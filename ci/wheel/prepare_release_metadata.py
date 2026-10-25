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

"""Pin the optional LLVM wheel to the main wheel's resolved release version."""

import argparse
from pathlib import Path

import tomlkit


def resolve_version(root, metadata):
    """Resolve the fixed version or the configured setuptools-scm source once."""
    project = metadata["project"]
    if "version" not in project.get("dynamic", []):
        return str(project["version"])
    from setuptools_scm import get_version

    return get_version(
        **{
            "relative_to": root / "pyproject.toml",
            "dist_name": project["name"],
            **metadata["tool"]["setuptools_scm"],
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[2])
    root = parser.parse_args().root.resolve()
    main_path = root / "pyproject.toml"
    extra_path = root / "ci/wheel/tvm-tirx-llvm-targets/pyproject.toml"
    main = tomlkit.parse(main_path.read_text(encoding="utf-8"))
    extra = tomlkit.parse(extra_path.read_text(encoding="utf-8"))
    version = resolve_version(root, main)
    # Freeze SCM metadata before generated files can make the checkout dirty.
    if "version" in main["project"].get("dynamic", []):
        main["project"]["dynamic"].remove("version")
        main["tool"]["scikit-build"].get("metadata", {}).pop("version", None)
    main["project"]["version"] = version
    main["project"]["optional-dependencies"]["all-targets"] = [
        f"apache-tvm-tirx-llvm-targets=={version}"
    ]
    extra["project"]["version"] = version
    extra["project"]["dependencies"] = [f"apache-tvm-tirx=={version}"]
    main_path.write_text(tomlkit.dumps(main), encoding="utf-8")
    extra_path.write_text(tomlkit.dumps(extra), encoding="utf-8")
    print(f"Prepared main and LLVM targets wheels at version {version}")


if __name__ == "__main__":
    main()
