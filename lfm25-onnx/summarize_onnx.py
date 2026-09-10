"""Print graph statistics for an exported ONNX model.

    python summarize_onnx.py onnx/LFM2.5-1.2B-Instruct/model.onnx
"""

from __future__ import annotations

import collections
import os
import sys

import onnx
from onnx import numpy_helper


def dims(t) -> str:
    out = []
    for d in t.type.tensor_type.shape.dim:
        out.append(d.dim_param if d.dim_param else str(d.dim_value))
    et = onnx.TensorProto.DataType.Name(t.type.tensor_type.elem_type)
    return f"{et}[{','.join(out)}]"


def main(path: str) -> None:
    m = onnx.load(path, load_external_data=False)
    g = m.graph
    print(f"file      {path}")
    print(f"          {os.path.getsize(path) / 1e6:.2f} MB graph", end="")
    if os.path.exists(path + ".data"):
        print(f" + {os.path.getsize(path + '.data') / 1e9:.2f} GB external data")
    else:
        print()
    ops = ", ".join(f"{o.domain or 'ai.onnx'}={o.version}" for o in m.opset_import)
    print(f"opset     {ops}   ir_version {m.ir_version}   producer {m.producer_name} {m.producer_version}")
    print(f"nodes     {len(g.node)}   initializers {len(g.initializer)}")

    total = 0
    for init in g.initializer:
        n = 1
        for d in init.dims:
            n *= d
        total += n
    print(f"params    {total / 1e6:.2f} M")

    print("\ninputs:")
    for i in g.input:
        print(f"  {i.name:16} {dims(i)}")
    print("outputs:")
    for o in g.output:
        print(f"  {o.name:16} {dims(o)}")

    hist = collections.Counter(n.op_type for n in g.node)
    print(f"\nop histogram ({len(hist)} distinct):")
    line = []
    for op, c in hist.most_common():
        line.append(f"{c} {op}")
    print("  " + ", ".join(line))


if __name__ == "__main__":
    main(sys.argv[1])
