from concurrent.futures import ThreadPoolExecutor

import pytest

from tvm_tirx import ir, runtime, tirx
from tvm_tirx.target import Target


@pytest.mark.gpu
def test_tensorless_launch_initializes_the_calling_thread_context(capfd):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    thread = ir.Var("thread", "int32")
    body = ir.Evaluate(ir.Call("tirx.cuda.printf", [ir.StringImm("tensorless thread launch\n")]))
    launch = ir.RegionStmt(
        "tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(1)], [thread], None, body
    )
    function = tirx.Function([], launch).with_attr("global_symbol", "tensorless")
    target = Target(
        {"kind": "cuda", "arch": f"sm_{device.compute_version.replace('.', '')}"}, host="c"
    )
    compiled = tirx.compile(function, target=target)

    def invoke():
        compiled()
        compiled()
        device.sync()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(invoke).result()
    assert capfd.readouterr().out.count("tensorless thread launch") == 2
