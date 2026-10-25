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

import tvm_ffi

import tvm_tirx.testing
from tvm_tirx import tirx


def test_mixed_iter_type_detection_interrupts_walk():
    spatial = tirx.Var("spatial", "int32")
    reduction = tirx.Var("reduction", "int32")
    unreachable = tirx.Var("unreachable", "int32")
    visited = []

    def detect(var):
        visited.append(var)
        if var.same_as(reduction):
            return tvm_ffi.VisitInterrupt()
        return None

    result = tvm_ffi.structural_walk(
        (spatial + reduction) + unreachable,
        (tirx.Var, detect),
        order="post",
    )

    assert isinstance(result, tvm_ffi.VisitInterrupt)
    assert any(var.same_as(spatial) for var in visited)
    assert any(var.same_as(reduction) for var in visited)
    assert not any(var.same_as(unreachable) for var in visited)


def test_region_body_metadata_tracks_definition_renewal():
    from tvm_tirx import ir

    storage = ir.Var("storage", tirx.TensorType([4], "float32", scope="local"))
    allocation = ir.Bind(storage, ir.Call("tirx.alloc_tensor", [], ty_args=[storage.ty]))
    region = ir.RegionStmt("tirx.scope", [], [], {"versions": {storage: 3}}, [allocation])
    function = tirx.Function([], [region], ret_type=ir.TupleType([]))
    restored = ir.load_json(ir.save_json(function))
    ir.assert_structural_equal(function, restored)
    assert tvm_ffi.structural_hash(function) == tvm_ffi.structural_hash(restored)
    renewed = tirx.renew_def(function)
    new_region = renewed.body.seq[0]
    new_storage = new_region.body.seq[0].var
    assert not new_storage.same_as(storage)
    assert int(new_region.attrs["versions"][new_storage]) == 3
    ir.assert_structural_equal(function, renewed)
    assert tirx.analysis.check_well_formed(renewed)


def test_index_map_metadata_has_local_parameters_and_lexical_captures():
    from tvm_tirx import ir

    parameter = ir.Var("parameter", "int32")
    coordinate = ir.Var("coordinate", "int32")
    mapping = tirx.IndexMap([coordinate], [coordinate * 2 + parameter], None)
    region = ir.RegionStmt("tirx.scope", [], [], {"maps": [mapping, mapping]}, [])
    function = tirx.Function([parameter], [region], ret_type=ir.TupleType([]))
    assert tirx.analysis.check_well_formed(function)
    assert tirx.analysis.check_well_formed(ir.load_json(ir.save_json(function)))
    assert tirx.analysis.check_well_formed(tirx.renew_def(function))
    missing_capture = tirx.Function([], [region], ret_type=ir.TupleType([]))
    assert not tirx.analysis.check_well_formed(missing_capture, assert_mode=False)
    escaped_parameter = tirx.Function(
        [parameter], [region, ir.Evaluate(coordinate)], ret_type=ir.TupleType([])
    )
    assert not tirx.analysis.check_well_formed(escaped_parameter, assert_mode=False)


def test_function_and_loop_metadata_see_only_visible_definitions():
    from tvm_tirx import ir

    parameter = ir.Var("parameter", "int32")
    local = ir.Var("local", "int32")
    index = ir.Var("index", "int32")
    body = ir.Bind(local, index + parameter)
    loop = ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, body, {"locals": {local: index}})
    function = tirx.Function([parameter], [loop], ret_type=ir.TupleType([]))
    assert tirx.analysis.check_well_formed(function.with_attr("parameter", parameter))
    assert not tirx.analysis.check_well_formed(
        function.with_attr("escaped", local), assert_mode=False
    )
    defined = tirx.Function([parameter], ir.Bind(local, parameter + 1), ret_type=ir.TupleType([]))
    assert tirx.analysis.check_well_formed(defined.with_attr("defined", {local: parameter}))


if __name__ == "__main__":
    tvm_tirx.testing.main()
