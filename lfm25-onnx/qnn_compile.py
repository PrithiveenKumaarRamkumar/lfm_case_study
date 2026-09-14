"""Compile an ONNX graph with the Qualcomm QNN Execution Provider on an x86_64
host, and emit every artifact that lets you *see* what the lowering did.

No Snapdragon is needed.  onnxruntime-qnn ships `libs/amd64/` containing
`QnnHtp.dll` + `HtpPrepare.dll` (the offline graph-preparation library),
`QnnIr.dll` and `QnnSaver.dll`, so the whole compile happens on the host and
the result is a deployable context binary.

Artifacts, named after the two viable methods in
`networkconversion_documentation_updated.md` -> "Visualizing the qnn lowered
artifact":

  method1/  <name>_ctx.onnx            EPContext wrapper - the macro view
            <name>_ctx.onnx_<g>_qnn.bin   the context binary itself (embed_mode=0)
  method3/  <graph>_json_qnn_graph.json   the lowered QNN graph - the micro view
            qnn_op_trace.json             ONNX op -> QNN op mapping
            <graph>_qnn_ep_input_graph.json  what QNN EP was handed, pre-partition
            <graph>.dlc                   QNN IR container (backend_type=ir)
            saver_output.c                the QNN C API call trace (backend_type=saver)

Usage:
    python qnn_compile.py <model.onnx> <outdir> [--backend htp|ir|saver]
                          [--htp-arch 73] [--soc-model 60] [--no-ctx]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import time
import traceback


def _register(ort, qnn_ep):
    name = qnn_ep.get_ep_name()
    try:
        ort.register_execution_provider_library(name, qnn_ep.get_library_path())
    except Exception as exc:                              # already registered
        if "already" not in str(exc).lower():
            raise
    devices = [d for d in ort.get_ep_devices() if d.ep_name == name]
    return name, devices


def describe_devices(devices) -> None:
    for d in devices:
        hw = getattr(d, "device", None)
        print(f"     OrtEpDevice ep_name={d.ep_name!r} "
              f"vendor={getattr(d, 'ep_vendor', '?')!r} "
              f"type={getattr(hw, 'type', '?')} "
              f"metadata={dict(getattr(d, 'ep_metadata', {}) or {})}")


def compile_model(model: str, outdir: str, backend: str, opts: dict,
                  ctx: bool, embed: str, verbose: int) -> dict:
    import onnxruntime as ort
    import onnxruntime_qnn as qnn_ep

    os.makedirs(outdir, exist_ok=True)
    name, devices = _register(ort, qnn_ep)
    print(f"[ep] {name}  onnxruntime {ort.__version__}  "
          f"onnxruntime-qnn {qnn_ep.__version__} (QNN SDK {qnn_ep.build_and_package_info.qnn_version})")
    print(f"[ep] libs: {qnn_ep.LIB_DIR_FULL_PATH}")
    if not devices:
        raise SystemExit("no OrtEpDevice advertised by the QNN plugin EP")
    describe_devices(devices)

    so = ort.SessionOptions()
    # 2 = WARNING (default here), 1 = INFO, 0 = VERBOSE.  The QNN EP prints its
    # partition summary ("Number of partitions supported by QNN EP") at VERBOSE.
    so.log_severity_level = max(0, 2 - verbose)
    so.log_verbosity_level = verbose
    if ctx:
        ctx_path = os.path.join(outdir, os.path.splitext(os.path.basename(model))[0] + "_ctx.onnx")
        so.add_session_config_entry("ep.context_enable", "1")
        so.add_session_config_entry("ep.context_file_path", ctx_path)
        so.add_session_config_entry("ep.context_embed_mode", embed)
    else:
        ctx_path = None

    ep_options = {"backend_type": backend}
    ep_options.update(opts)
    print(f"[ep] options {json.dumps(ep_options, indent=2)}")
    so.add_provider_for_devices(devices, ep_options)

    t0 = time.time()
    sess = ort.InferenceSession(model, sess_options=so)
    dt = time.time() - t0
    used = sess.get_providers()
    print(f"[ok] session created in {dt:.1f}s   providers={used}")
    for i in sess.get_inputs():
        print(f"     in  {i.name:18} {i.type:16} {i.shape}")
    for o in sess.get_outputs():
        print(f"     out {o.name:18} {o.type:16} {o.shape}")
    del sess
    return {"seconds": dt, "providers": used, "ctx": ctx_path}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("outdir")
    ap.add_argument("--backend", default="htp", choices=["htp", "ir", "saver", "cpu", "gpu"])
    ap.add_argument("--htp-arch", default=None)
    ap.add_argument("--soc-model", default=None)
    ap.add_argument("--no-ctx", action="store_true")
    ap.add_argument("--embed", default="0", choices=["0", "1"])
    ap.add_argument("--json-graph", action="store_true", help="method 3: dump the lowered QNN graph as JSON")
    ap.add_argument("--op-trace", action="store_true", help="method 3: ONNX op -> QNN op mapping")
    ap.add_argument("--input-graph", action="store_true", help="method 3: the ONNX graph the EP received")
    ap.add_argument("--dlc", action="store_true", help="method 3: QNN IR .dlc (needs --backend ir)")
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--opt", action="append", default=[],
                    metavar="KEY=VALUE", help="any other QNN EP provider option")
    a = ap.parse_args()

    opts: dict[str, str] = {}
    if a.htp_arch:
        opts["htp_arch"] = a.htp_arch
    if a.soc_model:
        opts["soc_model"] = a.soc_model
    if a.json_graph:
        opts["dump_json_qnn_graph"] = "1"
        opts["json_qnn_graph_dir"] = os.path.abspath(a.outdir)
    if a.op_trace:
        opts["enable_framework_op_trace"] = "1"
        opts["framework_op_trace_dir"] = os.path.abspath(a.outdir)
    if a.input_graph:
        opts["dump_qnn_ep_input_graph"] = "1"
        opts["dump_qnn_ep_input_graph_dir"] = os.path.abspath(a.outdir)
    if a.dlc:
        opts["dump_qnn_ir_dlc"] = "1"
        opts["dump_qnn_ir_dlc_dir"] = os.path.abspath(a.outdir)
    for kv in a.opt:
        k, _, v = kv.partition("=")
        opts[k] = v

    a.model = os.path.abspath(a.model)
    a.outdir = os.path.abspath(a.outdir)
    os.makedirs(a.outdir, exist_ok=True)
    if a.backend == "saver":
        os.chdir(a.outdir)   # QnnSaver.dll writes saver_output/ into the CWD

    print(f"[in] {a.model}  {os.path.getsize(a.model) / 1e6:.2f} MB")
    data = a.model + ".data"
    if os.path.exists(data):
        print(f"     + {os.path.basename(data)}  {os.path.getsize(data) / 1e9:.2f} GB external data")

    try:
        compile_model(a.model, a.outdir, a.backend, opts,
                      not a.no_ctx, a.embed, a.verbose)
    except Exception:
        traceback.print_exc()
        print("\n[FAILED]")

    # this box has 15.6 GB and a 24.6 GB pagefile; the peak is the whole story
    try:
        import psutil

        mi = psutil.Process().memory_info()
        print(f"\n[mem] peak working set {mi.peak_wset / 2**30:.2f} GB   "
              f"peak commit {mi.peak_pagefile / 2**30:.2f} GB")
    except Exception:  # noqa: BLE001
        pass

    print("\n[artifacts]")
    for root, _dirs, files in os.walk(a.outdir):
        for f in sorted(files):
            p = os.path.join(root, f)
            print(f"     {os.path.getsize(p):>14,}  {os.path.relpath(p, a.outdir)}")


if __name__ == "__main__":
    main()
