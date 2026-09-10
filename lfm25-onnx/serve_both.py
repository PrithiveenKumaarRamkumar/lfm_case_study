"""Serve both ONNX models in Netron, one child process per model.

ONE NETRON SERVER PER PROCESS.  netron/server.py stashes the served model on a
*class* attribute:

    class _HTTPRequestHandler(http.server.BaseHTTPRequestHandler):
        content = None                                     # class level
        ...
        self.server.RequestHandlerClass.content = content

`RequestHandlerClass` *is* that shared class, so a second netron.start() in the
same interpreter overwrites the first and every port serves whichever model
loaded last.  It fails quietly -- each port still answers HTTP 200 on / and the
viewer HTML is byte-identical regardless of which model is loaded -- so a
reachability check passes while the pages are wrong.  Hence one child each.

    python serve_both.py                          # both, ports 8080/8081
    python serve_both.py --serve <path> <port>     # internal single-model child
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = [
    ("LFM2.5-1.2B-Instruct",
     os.path.join(ROOT, "onnx", "LFM2.5-1.2B-Instruct", "model.onnx"), 8080),
    ("LFM2.5-1.2B-Instruct-DSpark",
     os.path.join(ROOT, "onnx", "LFM2.5-1.2B-Instruct-DSpark", "model.onnx"), 8081),
]


def serve_one(path: str, port: int) -> None:
    import netron

    netron.start(path, address=("localhost", port), browse=False)
    print(f"SERVING {os.path.basename(os.path.dirname(path))} on {port}", flush=True)
    threading.Event().wait()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        serve_one(sys.argv[2], int(sys.argv[3]))
        return

    children = []
    for name, path, port in MODELS:
        if not os.path.exists(path):
            print(f"MISSING {path}")
            continue
        children.append(subprocess.Popen(
            [sys.executable, "-u", os.path.abspath(__file__), "--serve", path, str(port)]
        ))
        print(f"{name:32} http://localhost:{port}", flush=True)
    print("READY", flush=True)
    for c in children:
        c.wait()


if __name__ == "__main__":
    main()
