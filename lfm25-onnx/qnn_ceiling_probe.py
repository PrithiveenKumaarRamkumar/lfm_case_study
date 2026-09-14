"""Is there a static-tensor ceiling in the offline HTP graph preparer?

The 1.2B target refuses to finalise with

    QNN_COMMON_ERROR_MEM_ALLOC: Memory allocation related error., Code: 1002

at 4.86 GiB of static tensors, while the 430M drafter finalises fine at
2.14 GiB — and the host is nowhere near out of memory when it happens (peak
working set 6.08 GB against 15.6 GB of RAM and a 24.6 GB pagefile).  A size
limit inside the backend is the obvious explanation, so this bisects it with a
graph that is nothing *but* weights: a chain of N [D, D] float32 MatMuls.

The weights are written straight into the external-data file, so building a
5 GiB probe costs 5 GiB of disk and D*D*4 bytes of RAM.

**Answer: no ceiling.**  2.00 / 3.75 / 4.00 / 4.50 / 4.88 / 5.50 GiB all
finalise — 4.88 GiB being more static weight than the target carries.  The
target's failure is structural, not dimensional; see S3.10 in onnx_session.md.

    python qnn_ceiling_probe.py 3.75 4.5 5.5        # sizes in GiB
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper

D = 4096                       # 4096 x 4096 float32 = 64 MiB per layer
LAYER_BYTES = D * D * 4


def build(path: str, layers: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = os.path.basename(path) + ".data"
    rng = np.random.default_rng(0)
    inits, nodes = [], []
    offset = 0
    with open(os.path.join(os.path.dirname(path), blob), "wb") as fh:
        for i in range(layers):
            w = (rng.standard_normal((D, D), dtype=np.float32) * 0.02)
            fh.write(w.tobytes())
            t = TensorProto(name=f"w{i}", data_type=TensorProto.FLOAT, dims=[D, D])
            t.data_location = TensorProto.EXTERNAL
            for k, v in (("location", blob), ("offset", str(offset)),
                         ("length", str(LAYER_BYTES))):
                e = t.external_data.add()
                e.key, e.value = k, v
            inits.append(t)
            nodes.append(helper.make_node(
                "MatMul", [f"h{i}", f"w{i}"], [f"h{i + 1}"], name=f"mm{i}"))
            offset += LAYER_BYTES
    g = helper.make_graph(
        nodes, "ceiling",
        [helper.make_tensor_value_info("h0", TensorProto.FLOAT, [1, D])],
        [helper.make_tensor_value_info(f"h{layers}", TensorProto.FLOAT, [1, D])],
        inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10
    onnx.save(m, path)


def main(sizes: list[float]) -> None:
    root = os.path.join(tempfile.gettempdir(), "qnn_ceiling")
    here = os.path.dirname(os.path.abspath(__file__))
    for gib in sizes:
        layers = max(1, round(gib * 2**30 / LAYER_BYTES))
        actual = layers * LAYER_BYTES / 2**30
        shutil.rmtree(root, ignore_errors=True)
        path = os.path.join(root, "probe.onnx")
        build(path, layers)
        out = subprocess.run(
            [sys.executable, os.path.join(here, "qnn_compile.py"), path,
             os.path.join(root, "out"), "--backend", "htp", "--no-ctx",
             "--opt", "soc_model=69", "--opt", "htp_arch=75"],
            capture_output=True, text=True, errors="replace",
            env={**os.environ, "PYTHONUTF8": "1"})
        blob = (out.stdout or "") + (out.stderr or "")
        verdict = "OK  " if "[ok] session created" in blob else "FAIL"
        detail = ""
        for line in blob.splitlines():
            if "session created" in line or "EPFail" in line:
                detail = line.strip()[:110]
        print(f"{layers:>3} layers  {actual:6.2f} GiB static   {verdict}  {detail}",
              flush=True)
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main([float(a) for a in sys.argv[1:]] or [2.0, 3.0, 3.75, 4.0, 4.25])
