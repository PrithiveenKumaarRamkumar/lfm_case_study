"""Freeze the dynamic dimensions of an exported graph, cheaply.

QNN EP refuses any node whose shape is not static, so both exports have to be
pinned before they can be compiled.  The obvious tool,

    python -m onnxruntime.tools.make_dynamic_shape_fixed ...

round-trips the *whole* model, which for the 1.2B target means materialising
4.68 GB of initializers and writing a second copy of them.  Not needed: the
weights are already in `model.onnx.data` and external-data locations are
resolved relative to the model file, so writing the patched 1.8 MB topology
*into the same directory* reuses the existing blob untouched.

    python fix_shapes.py onnx/LFM2.5-1.2B-Instruct/model.onnx sequence=32
"""

from __future__ import annotations

import os
import sys

import onnx


def fix(src: str, overrides: dict[str, int], dst: str | None = None) -> str:
    m = onnx.load(src, load_external_data=False)
    seen: dict[str, int] = {}
    for vi in list(m.graph.input) + list(m.graph.output):
        tt = vi.type.tensor_type
        for d in tt.shape.dim:
            if d.HasField("dim_param") and d.dim_param in overrides:
                seen[d.dim_param] = seen.get(d.dim_param, 0) + 1
                d.dim_value = overrides[d.dim_param]
                d.ClearField("dim_param")
    # stale value_info would contradict the new static shapes
    del m.graph.value_info[:]
    unknown = set(overrides) - set(seen)
    if unknown:
        raise SystemExit(f"no such dim_param(s) in {src}: {sorted(unknown)}")
    if dst is None:
        dst = os.path.join(os.path.dirname(src), "model_fixed.onnx")
    onnx.save(m, dst)  # topology only; .data reference is left as-is
    print(f"[ok] {os.path.basename(dst)}  {os.path.getsize(dst) / 1e6:.2f} MB")
    for k, v in overrides.items():
        print(f"     {k} -> {v}   ({seen[k]} occurrence(s) in graph I/O)")
    for vi in m.graph.input:
        dims = [d.dim_value if d.HasField("dim_value") else d.dim_param
                for d in vi.type.tensor_type.shape.dim]
        print(f"     in  {vi.name:16} {dims}")
    return dst


def dims_of(path: str) -> None:
    m = onnx.load(path, load_external_data=False)
    for tag, seq in (("in", m.graph.input), ("out", m.graph.output)):
        for vi in seq:
            tt = vi.type.tensor_type
            dims = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
                    for d in tt.shape.dim]
            print(f"  {tag:3} {vi.name:18} {onnx.TensorProto.DataType.Name(tt.elem_type):7} {dims}")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        dims_of(sys.argv[1])
    else:
        ov = {}
        for a in sys.argv[2:]:
            k, _, v = a.partition("=")
            ov[k] = int(v)
        fix(sys.argv[1], ov)
