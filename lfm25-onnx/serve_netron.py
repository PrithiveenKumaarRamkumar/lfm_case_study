"""Serve every graph this session produced in Netron, one child process per file.

Same constraint as `serve_both.py`: netron/server.py keeps the served model on a
*class* attribute of the request handler, so two `netron.start()` calls in one
interpreter make every port serve whichever file loaded last.  One child each.

    python serve_netron.py            # print the table and serve
    python serve_netron.py --open     # ... and open a browser tab per port
    python serve_netron.py a.onnx:8090 b.json:8091

Ports:
    8080/8081  the two source ONNX graphs
    8082       method 1 - the EPContext wrapper (drafter)
    8083..     method 3 - the lowered QNN graphs and the EP's input graphs
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
ONNX = os.path.join(ROOT, "onnx")
QNN = os.path.join(ROOT, "qnn")
T = "LFM2.5-1.2B-Instruct"
D = "LFM2.5-1.2B-Instruct-DSpark"


def _lowered(model: str, sub: str) -> str | None:
    """The lowered QNN graph JSON, i.e. not a tensor_log / op_trace / input graph."""
    d = os.path.join(QNN, model, sub)
    if not os.path.isdir(d):
        return None
    for f in sorted(os.listdir(d)):
        if (f.startswith("QNNExecutionProvider") and f.endswith(".json")
                and "tensor_log" not in f):
            return os.path.join(d, f)
    return None


def targets() -> list[tuple[str, int, str]]:
    out = [
        (os.path.join(ONNX, T, "model.onnx"), 8080, "source ONNX - target"),
        (os.path.join(ONNX, D, "model.onnx"), 8081, "source ONNX - drafter"),
        (os.path.join(QNN, D, "htp", "model_fixed_ctx.onnx"), 8082,
         "METHOD 1 - EPContext wrapper (drafter)"),
    ]
    port = 8083
    for model, sub, label in ((D, "htp", "drafter, htp"), (T, "ir", "target, ir"),
                              (T, "htp", "target, htp partition")):
        p = _lowered(model, sub)
        if p:
            out.append((p, port, f"METHOD 3 - lowered QNN graph ({label})"))
            port += 1
    for model, label in ((D, "drafter"), (T, "target")):
        for sub in ("htp", "ir", "htp_fp16"):
            p = os.path.join(QNN, model, sub, "main_graph.1_qnn_ep_input_graph.json")
            if os.path.exists(p):
                out.append((p, port, f"METHOD 3 - what QNN EP was handed ({label})"))
                port += 1
                break
    return out


def serve_one(path: str, port: int) -> None:
    import netron

    netron.start(path, address=("localhost", port), browse=False)
    print(f"SERVING {path} on {port}", flush=True)
    threading.Event().wait()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        serve_one(sys.argv[2], int(sys.argv[3]))
        return

    args = [a for a in sys.argv[1:] if a != "--open"]
    browse = "--open" in sys.argv
    plan = ([(a.rsplit(":", 1)[0], int(a.rsplit(":", 1)[1]), "") for a in args]
            if args else targets())

    children = []
    for path, port, label in plan:
        if not os.path.exists(path):
            print(f"MISSING  :{port}  {path}")
            continue
        children.append(subprocess.Popen(
            [sys.executable, "-u", os.path.abspath(__file__), "--serve", path, str(port)]
        ))
        print(f"http://localhost:{port}  {os.path.getsize(path):>10,} B  "
              f"{label:<44} {os.path.relpath(path, ROOT)}", flush=True)
        if browse:
            webbrowser.open(f"http://localhost:{port}")
    print("READY", flush=True)
    for c in children:
        c.wait()


if __name__ == "__main__":
    main()
