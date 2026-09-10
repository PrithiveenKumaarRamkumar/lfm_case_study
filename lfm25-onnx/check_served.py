"""Verify what each Netron port is *actually* serving.

A reachability check on / is not enough: the Netron viewer HTML is identical for
every model, so a port serving the wrong graph still answers HTTP 200 with
byte-identical content.  Fetch /data/model.onnx and parse it instead.

    python check_served.py 8080 8081
"""

from __future__ import annotations

import collections
import io
import sys
import urllib.request

import onnx


def probe(port: int) -> None:
    url = f"http://localhost:{port}/data/model.onnx"
    try:
        raw = urllib.request.urlopen(url, timeout=20).read()
    except Exception as exc:  # noqa: BLE001
        print(f"port {port}: UNREACHABLE ({exc})")
        return
    m = onnx.load_model_from_string(raw)
    g = m.graph
    hist = collections.Counter(n.op_type for n in g.node)
    params = 0
    for init in g.initializer:
        n = 1
        for d in init.dims:
            n *= d
        params += n
    print(f"port {port}: {len(raw):>9,} bytes  {len(g.node):>4} nodes  "
          f"{params / 1e6:>8.2f}M params  Conv={hist.get('Conv', 0)} "
          f"Softmax={hist.get('Softmax', 0)}  in={[i.name for i in g.input]}")


if __name__ == "__main__":
    for p in sys.argv[1:] or ["8080", "8081"]:
        probe(int(p))
