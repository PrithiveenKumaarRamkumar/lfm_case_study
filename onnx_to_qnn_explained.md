All eight ports still live. Here's what's actually happening, and what each tab shows you.

---

# Part 1 — What "ONNX → QNN" actually is

The first thing to unlearn: **this is not a file conversion.** There is no `onnx2qnn.exe`. The QNN Execution Provider is a translator that walks your ONNX graph and *makes C API calls* to build a second graph inside Qualcomm's runtime, then asks Qualcomm's compiler to turn that into Hexagon machine code. Nothing is written to disk unless you explicitly ask for a dump — which is exactly what Methods 1 and 3 are.

## The nine stages

```
  model.onnx  (612 nodes, dims 'batch'/'block'/'ctx')
      │
  1 ──┤  freeze shapes            fix_shapes.py → model_fixed.onnx
      │                           QNN has no notion of a symbolic dim
      ▼
  2 ──┤  ORT Level-1 optimisation  612 → 574 nodes   (before the EP sees anything)
      ▼
  3 ──┤  GetCapability()           per node: backendValidateOpConfig()
      │                            "can HTP run this?"   1197 calls for 620 nodes
      ▼
  4 ──┤  partitioning              supported nodes → fused subgraphs
      │                            rejected nodes → stay on CPUExecutionProvider
      ▼
  5 ──┤  layout transform          ONNX is NCHW, QNN is NHWC
      │                            target: 60 Transposes → 30      ← ports 8087 vs 8085
      ▼
  6 ──┤  COMPOSE                   graphCreate / tensorCreateGraphTensor / graphAddNode
      │                            ONNX ops become qti.aisw ops      ← METHOD 3 dumped HERE
      ▼
  7 ──┤  graphFinalize()           HtpPrepare.dll: fusion, tiling, VTCM
      │                            allocation, kernel selection, fp16
      │                            ███ completely opaque ███
      ▼
  8 ──┤  contextGetBinary()        1,130,196,264 bytes of Hexagon machine state
      ▼
  9 ──┤  wrap                      one EPContext ONNX node              ← METHOD 1
```

**The single most useful thing to understand from that diagram:** Method 3 is a snapshot taken at stage 6, Method 1 is the product of stage 8. Everything Qualcomm's own compiler does — stage 7 — is invisible in both. That's why the Method 3 dump still shows all 27 RMSNorms as four separate ops each (`Pow`, `ReduceMean`, `Sqrt`, `Divide`): whatever HTP does to fuse them happens after the snapshot and never surfaces.

It also explains something that looks paradoxical in this session's results: **the target model produced a Method 3 artifact but no Method 1 artifact.** The dump happens at stage 6, the crash happened at stage 7. Method 3 survives failures Method 1 cannot.

## What translation actually changes

The ops get renamed into a different vocabulary, and the renaming is informative:

| ONNX | QNN (`qti.aisw`) | why it matters |
|---|---|---|
| `MatMul` ×58 | `FullyConnected` ×48 + `MatMul` ×10 | QNN distinguishes *weight × activation* from *activation × activation*. 48 Linears, 10 real batched matmuls (5 attention layers × QK and PV) |
| `Conv` ×10 | `DepthWiseConv2d` ×10 + `Reshape` ×20 | LFM2's ShortConv **is** a depthwise conv, and Qualcomm's op-support logic says so. Two reshapes each because QNN's depthwise op is 2-D and a sequence is 1-D |
| `Reciprocal` | `ElementWiseDivide` | QNN has no reciprocal — the builder synthesises a `1.0` numerator constant. Harmless at fp32, **fatal at fp16** |
| `Expand` ×10 | `ElementWiseMultiply` | no broadcast-materialising op, so a broadcast is a multiply by ones. These ten are the GQA head fan-out |
| `Mul` + `Softmax` | **one** `Softmax` | `ScaleSoftmaxFusion`, once per attention layer, all five |
| `Unsqueeze` ×46 | `Reshape` | plus one reshape per `FullyConnected` — which is why 55 ONNX reshapes become 150 |
| `Gather` | `Cast` → `Gather` | QNN indexes with int32, so every int64 token-id input gets narrowed |

And there are **three** dialects in play, not two — the same three-node smoke test reads differently depending on where you look:

| | Conv | Relu | Add |
|---|---|---|---|
| ONNX | `Conv` | `Relu` | `Add` |
| QNN JSON / saver (`qti.aisw`) | `Conv2d` | `Relu` | `ElementWiseAdd` |
| inside the `.dlc` FlatBuffer | `Conv2d` | `Neuron` | `Eltwise_Binary` |

## The one structural surprise

The lowering **duplicates weights**. Every `Transpose(initializer) → MatMul` in the exported graph gets constant-folded into a new static tensor in `[out, in]` layout for `FullyConnected` — and where the original is still consumed elsewhere, both copies live in the graph:

| | ONNX params | QNN static tensors | delta |
|---|---:|---:|---:|
| drafter | 429.94 M | **574.65 M** | +144.71 M |
| target | 1170.34 M | **1305.35 M** | +135.01 M |

Both deltas are ≈ 134.2 M = 65536 × 2048 — one extra copy of the tied embedding. In ONNX the tie is one initializer with two consumers (the embedding `Gather` and the `lm_head`); after lowering it's `shared_embedding` **and** `t_transpose`, 537 MB each. **Weight tying does not survive the NPU lowering.**

---

# Part 2 — The eight Netron tabs

Four of these are the drafter's full journey and four are the target's. They're deliberately paired.

## The drafter — the path that worked end to end

| port | what it is | nodes | what to look at |
|---|---|---:|---|
| **8081** | source ONNX, as `torch.onnx.export` wrote it | 612 | the input ONNX. Note the dims are symbolic — `input_ids [batch, block]`. This graph is **rejected** by QNN EP as-is |
| **8086** | what QNN EP was actually handed | 574 | after ORT's Level-1 pass. 38 nodes already folded away. Still pure ONNX op names |
| **8083** | **METHOD 3** — the lowered QNN graph | 620 | the payoff. `qti.aisw::FullyConnected`, `ElementWiseMultiply`, `ElementWiseDivide`. Click a `Softmax` — it ate a `Mul`. Click a weight — `val_10_transpose [2048, 10240]`, the transposed copy |
| **8082** | **METHOD 1** — the EPContext wrapper | **1** | five inputs, three outputs, and one opaque node between them. All 620 ops are gone into a 1.13 GB binary. Click it to see `ep_cache_context`, `source=QNNExecutionProvider`, `ep_sdk_version=v2.50.40.260831140417` |

Read **8081 → 8086 → 8083 → 8082** and you watch 612 ONNX nodes become 574, then 620 QNN ops, then a single black box. That progression *is* the two methods: 8083 is the white box, 8082 is the black box, and they're the same model.

## The target — the path that stopped short

| port | what it is | nodes | what to look at |
|---|---|---:|---|
| **8080** | source ONNX | 869 | the 1.2B hybrid. 10 `Conv` + 6 `Softmax` interleaved — §3's architecture, visible |
| **8087** | what QNN EP was handed (after layout transform) | 783 | compare against 8080: **30 of the 60 Transposes are gone.** The layout optimiser cancelled the Transpose→Conv→Transpose sandwiches around the ShortConv stack before QNN saw them |
| **8084** | **METHOD 3** — full lowered graph, via the `ir` backend | 893 | the whole target in QNN dialect, one partition, zero unsupported. **This is where `Conv → DepthWiseConv2d` is visible** — ten of them |
| **8085** | **METHOD 3** — the HTP partition fragment | **138** | the consequence. Under the real HTP backend, one `GatherNd` is rejected (`backendValidateOpConfig` error 3110) and the graph is cut. This is what survived: embedding + two ShortConv layers + 13 `FullyConnected`, handing **six** tensors back to the CPU, one of them boolean |

8084 next to 8085 is the most instructive pair in the set. Same model, same session, two backends: the `ir` backend validates against a schema and takes all 893 ops; the `htp` backend validates against actual silicon and refuses one node, which fragments an otherwise fully-offloadable 1.2 B model. **There is no Method 1 tab for the target** because stage 7 never completed — and the reason is on port 8085.

## A note on what Netron can and can't tell you here

`netron/qnn.js` is upstream Netron's QNN reader, and its first line is `// Experimental`. It renders every node with correct types, shapes and edges, but `qnn-metadata.json` documents only **7** ops — `Conv2d`, `DepthWiseConv2d`, `FullyConnected`, `PoolAvg2d`, `Pool`, `Transpose`, `Neuron`. Everything else shows generic `input`/`output` port names instead of documented ones. For these two graphs that's most of the nodes, so read the edges rather than the port labels.

And if you point Netron at the context binary itself (`model_fixed_ctx_qnn.bin`), it *recognises* the format — the file starts with one of four signatures `qnn.js` knows — and then deliberately refuses: `File contains undocumented QNN serialized context.` That's the hard boundary. Stage 7's output is Qualcomm's, and nothing in this toolchain reads it.

Six things, roughly in order of how badly they'll bite. Each is something this session either brushed past or deliberately didn't do.

---

## 1. Everything we did was float — and that's the unusual case

Look at the tensor table from the lowered drafter graph. All 739 tensors carry:

```json
"quant_params": { "definition": 2147483647, "encoding": 2147483647 }
```

`0x7fffffff` is QNN's *undefined*. Method 3's advertised headline feature — "quantization encodings (scale/zero-point)" — had an empty column for us, in every single tensor.

That's not normal. The Hexagon NPU will happily execute fp16, but it is engineered around **int8/int16 fixed point**: that's where the MAC throughput and the power advantage live. Nearly every production QNN deployment goes through quantization, which means a novice's second project should be:

```python
from onnxruntime.quantization import quantize_static, get_qdq_config, QuantType, CalibrationDataReader
```

All of that ships with the `onnxruntime` you already have. The workflow is different in kind from what we did:

- You need **calibration data** — real inputs, a few hundred, run through the model to record activation ranges. There is no "just quantize it" button that preserves accuracy.
- You emit **QDQ format** (`QuantizeLinear`/`DequantizeLinear` pairs around each op), not raw int8 tensors. QNN EP reads the QDQ pattern and folds it into native quantized ops.
- You then have an **accuracy problem to manage**, which is a modelling task, not a compilation task. Per-channel vs per-tensor, which ops to keep in fp16, where outliers break your scales.

The honest framing: what we built is *the easy half*. Getting the graph onto the NPU is an afternoon. Getting it onto the NPU **at int8 without losing quality** is the actual project.

---

## 2. The context binary is a lockfile, not a portable artifact

This is the trade you make by choosing AOT, and it's easy to miss because the win is so visible (137 s of compile collapsing into an instant session on device).

Compare the two EPContext models this session produced:

```
smoke test, no SoC specified :  v2:6:2.50.40:5.50.0:0:0:0:0
drafter, soc_model=69 arch=75:  v2:6:2.50.40:5.50.0:75:69:0:0
                                   │      │       │  │
                            ORT EP │      │       │  └── SoC model
                              QAIRT SDK ──┘       └───── HTP arch (V75)
```

That `ep_compatibility_info` string is a compatibility gate, and every field in it can invalidate your binary:

- **Compiled for SoC 69 / V75.** It will not load on a V73 or V79 part. Ship to three phone tiers, compile three binaries.
- **Bound to QAIRT 2.50.40.** Bump `onnxruntime-qnn` and you recompile everything. This is a CI concern, not a one-off.
- Our 1.13 GB drafter binary is **the deployable artifact** — larger than the 0.86 GB fp16 ONNX it came from, and useless anywhere but the chip it was built for.

The mental model that helps: a context binary is closer to a statically-linked executable for one CPU model than it is to a portable model file. The `.onnx` stays your source of truth; the `.bin` is a build output you regenerate per target.

---

## 3. What we built is legible, not deployable — and an LLM is an awkward fit here

Two facts sit badly together:

- QNN requires **static shapes**. Not "prefers" — `EP_FAIL`, no session, as port 8081's graph demonstrates.
- Autoregressive generation is **inherently dynamic**: the sequence grows by one token per step, and a KV cache grows with it.

Our export dodged this by being prefill-only at `sequence=32`, which makes the architecture readable in Netron (that was the point, §6 of the session log) but is not a thing you can generate text with. Real deployments resolve the tension one of three ways: fixed-length **buckets** (pad to 128/256/512, compile one binary each), a separate **prefill graph and decode graph** with the cache as explicit fixed-size I/O, or a purpose-built runtime.

Which brings up something sitting in plain sight that we never touched. In the wheel listing:

```
     9,613,008  Genie.dll        (amd64)
    19,775,696  Genie.dll        (arm64ec)
```

**Genie is Qualcomm's LLM runtime** — the piece that handles KV cache management, bucketing and token-by-token decode on Hexagon. It ships alongside the EP and is the actual answer to "how do I run a 1.2 B model on this chip". A novice should know it exists before concluding from our results that LLMs and QNN don't mix. They do; just not through the plain EP path we used.

---

## 4. Count partitions, not unsupported operators

The instinct is to read `unsupported_nodes: 0` and relax. The number that actually predicts performance is `qnn_subgraphs`.

Every partition boundary is a **round trip across the NPU/CPU memory fence** — outputs copied out of shared RPC memory, CPU kernels run, inputs copied back. The target's failing HTP partition hands **six** tensors back to the host, one of them a boolean mask. Six copies, twice per boundary, per inference.

The pathological outcome is real: a graph chopped into a dozen partitions can run *slower on the NPU than pure CPU*, because you've paid for all the transfers and none of the throughput. And it only takes one op — our target fragmented on a **single** `GatherNd`.

So the checks, in order:

```python
trace["summary"]["qnn_subgraphs"]      # want 1, tolerate 2
trace["summary"]["unsupported_nodes"]  # want 0
trace["unsupported_nodes"]             # the actual list, when it isn't 0
```

And when you find an offender, the fix is usually **upstream in the export**, not downstream in the EP. Our `GatherND` came from `transformers` 5.x mask construction — not from LFM2's architecture at all. Change how you export and it disappears. That's the general shape of the fix: unsupported ops are far more often an artifact of your exporter than a genuine hardware limitation.

---

## 5. Lowering is not semantics-preserving, and nobody checks it for you

`export_lfm2.py` verifies ONNX against PyTorch and prints `max|diff| 5.770e-05, argmax agreement 100.00%`. **There is no equivalent check for the QNN stage**, and we couldn't run one — no Hexagon in this laptop.

Meanwhile the lowering changed, silently:

- **Precision.** HTP does fp16 math from QAIRT 2.35 regardless of what you asked for. Your fp32 model is not running fp32.
- **Integer width.** Every int64 input was narrowed to int32 (`input_ids` → `input_ids_int32`). Fine for a 65 536 vocabulary. Not fine for any int64 arithmetic that can exceed 2³¹ — and nothing warns you.
- **Fused ops.** `ScaleSoftmaxFusion` merged a `Mul` into each `Softmax`. Different rounding.
- **Layout.** NCHW → NHWC, weights relaid OIHW → HWIO.
- Then **stage 7** did more fusion, retiling and kernel selection that no dump shows.

Each is individually defensible; stacked, they are a real numerical delta that you have to measure. So the deployment checklist has a step that our session structurally could not perform: **run the context binary on the actual device against the same inputs, and diff.** Treat any pipeline without that step as unverified, however clean the compile log looked.

---

## 6. QNN error messages are misleading — learn the escalation ladder

Both blocking errors in this session pointed at the wrong thing:

| what it said | what it meant |
|---|---|
| `Dynamic shape is not supported yet, for input: val_16` | `val_16` is an *internal* tensor. The thing you must change is a graph input, which it does not name |
| `QNN_COMMON_ERROR_MEM_ALLOC: Memory allocation related error` | not memory. 6.08 GB peak against 15.6 GB RAM, and a 5.5 GiB synthetic graph finalises fine. Sometimes it means "you didn't set `soc_model`" |

The same string meant two unrelated things in one session. So don't debug from the exception — debug from the artifacts, cheapest first:

1. **`qnn_op_trace.json`** — `unsupported_nodes` and `qnn_subgraphs`. Answers most questions in ten seconds.
2. **`dump_qnn_ep_input_graph`** — is the problem even reaching the EP, or did ORT's optimiser create it? Diffing pass `.0` against `.1` is how we saw 30 Transposes get cancelled.
3. **`dump_json_qnn_graph`** — the lowered graph. Dumped at *compose*, so it survives a finalize crash, which is precisely how we got Method 3 for the target.
4. **`--verbose 2`** — where `backendValidateOpConfig() failed for node 'node_GatherND_34' ... error code 3110` finally appeared. One line, and it named the culprit exactly.
5. **`backend_type=saver`** — the nuclear option. 50,344 lines of C replaying every API call. Reach for it when you suspect the EP, not your model.

There's a structural lesson underneath: **use `backend_type=ir` as a control.** It validates against a schema instead of silicon, so `ir` succeeding while `htp` fails tells you the problem is hardware op support, not your graph. That's how we separated "the target lowers cleanly" (893 ops, one partition) from "the target won't compile" (fragmented on `GatherNd`) — two very different diagnoses that the HTP path alone would have conflated.

---

## The compressed version

If a novice keeps five sentences from all of this: **quantize, because that's what the hardware is for.** **Freeze your shapes before you do anything else.** **One partition or bust — and check the count, not the op list.** **Your context binary is built for exactly one chip and one SDK version.** **Nothing has verified your numerics until you've run the binary on real silicon.**

If you want, I can append this as an `S3.20 — what to do next` section to the session log, or fold it into `onnx_to_qnn_explained.md` as a Part 3.
