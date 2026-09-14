"""Cast an exported graph to float16, keeping the graph inputs/outputs float32.

Motivation was practical: the QNN HTP backend refuses to finalise the 1.2B
target at float32 with

    QNN_COMMON_ERROR_MEM_ALLOC: Memory allocation related error., Code: 1002

and the lowered graph carries 4.86 GiB of static tensors.  HTP does its float
math in fp16 anyway (QAIRT >= 2.35), so handing it fp16 weights should cost
nothing and halve the static footprint.

**It does not help, in either direction.**  The size theory turned out to be
wrong (`qnn_ceiling_probe.py` finalises 5.5 GiB), and onnxruntime-qnn 2.6.0
cannot compose *any* fp16 graph containing `Reciprocal` — its op builder
declares the synthesised numerator float16 and then hands over four bytes:

    Data length mismatch for static tensor. node_name: node_rsqrt
    tensor_name: node_rsqrt_divisor. size calculated from shape: 2,
    tensor.clientBuf.dataSize: 4

Both LFM2.5 graphs have a `Reciprocal` in every RMSNorm.  Kept because the
conversion itself is correct and cheap, and because the repro is worth having.
See S3.10 / S3.11 in onnx_session.md.

    python to_fp16.py onnx/LFM2.5-1.2B-Instruct/model_fixed.onnx
"""

from __future__ import annotations

import os
import sys
import time

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16


def main(src: str, dst: str | None = None) -> None:
    dst = dst or os.path.join(os.path.dirname(src), "model_fp16.onnx")
    t0 = time.time()
    m = onnx.load(src)  # pulls in the external .data - this is the expensive bit
    print(f"[ok] loaded in {time.time() - t0:.1f}s  "
          f"{len(m.graph.node)} nodes  {len(m.graph.initializer)} initializers")

    t1 = time.time()
    m = convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=True)
    print(f"[ok] converted in {time.time() - t1:.1f}s  {len(m.graph.node)} nodes "
          f"(+Cast at the boundary)")

    t2 = time.time()
    onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True,
              location=os.path.basename(dst) + ".data", size_threshold=1024)
    print(f"[ok] saved in {time.time() - t2:.1f}s")
    for p in (dst, dst + ".data"):
        if os.path.exists(p):
            print(f"     {os.path.basename(p):24} {os.path.getsize(p) / 1e6:10.2f} MB")

    try:
        import psutil

        print(f"[mem] peak working set "
              f"{psutil.Process().memory_info().peak_wset / 2**30:.2f} GB")
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main(*sys.argv[1:])
