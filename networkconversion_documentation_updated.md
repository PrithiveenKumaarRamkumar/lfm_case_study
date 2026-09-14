In the Renesas R-Car V3H software development workflow, managing data **tiling** and configuring **IMPC (Image Measurement Processor Core / IMP-X5+) registers** are **critical components of optimizing hardware-accelerated computer vision pipelines** . [1, 2]

The core conceptual architecture, driver framework, and expectations for configuring these registers within the [R-Car SDK](https://www.renesas.com/en/products/r-car-v3h) operate as follows: [3]

### 1. Tiling in the R-Car V3H Platform

The IMP-X5 (IMPC) subsystem relies on strict memory sub-sampling and local RAM caching to achieve high performance while avoiding external DRAM bandwidth bottlenecks. [2]

- **The Purpose:** Image frames are divided into regional 2D rectangles called **tiles** . The hardware reads an input tile into internal high-speed local memory (CLRAM), processes it using individual IMP vector/signal cores, and writes the output tile back out to memory.
- **Overlap &amp; Padding:** For convolutional filters or localized kernel operators (such as a 3x3 or 5x5 Sobel filter), your tiling engine must configure **overlapping borders (margins)** . The SDK handles this via software frameworks that compute the bounding boxes dynamically based on the kernel size so that boundary pixels are not lost between adjoining tiles.
- **Software Abstraction:** In modern R-Car SDKs (e.g., using **RoX SDK** or **R-Car AI Studio** packages), developers rarely write low-level loop bounds for tiling manually. Algorithms are mapped using Renesas' graph tools or specialized vision libraries (like their optimized version of OpenVX/proprietary vision APIs) which automatically slice the input buffers into hardware-compliant tiles. [2, 4]

### 2. Setting IMPC Registers and Configuration

Low-level IMPC configuration relies on the **IMP Driver API** interacting with the hardware registers. Hardware processing relies on a multi-stage sequence typically structured like this:

- **Control and Command Lists:** Instead of updating hardware registers directly on the fly from the Cortex-A53 cores (which introduces latency), the IMPC uses a **Command List (CL)** architecture. [5, 6]
- **The Routine:**
    - 1.1. The software prepares a sequence of commands (macro-instructions) in system memory.
    - 1.2. These commands explicitly write to IMPC internal control registers, configuration registers, source/destination stride variables, and base memory addresses.
    - 1.3. The CPU kicks off execution by writing to the IMP main execution trigger register. The IMP internal controller then reads the command list and updates its own local configuration registers sequentially.

### Key Types of IMPC Registers Configured via the SDK:

- **Memory Base and Stride Registers:** Define the base addresses of the image tiles in physical memory space, alongside line strides (pitches) to ensure correct memory-to-CLRAM mapping.
- **Core Execution Target Registers:** Determine which hardware acceleration units (e.g., IMP Vector Cores, Specialized Hardware Accelerators like CNN, Stereo Disparity, or Optical Flow blocks) are linked to execute the current tile. [1]
- **Sync / Interrupt Registers:** Configure internal hardware barrier registers to synchronize multiple parallel IMP cores. This ensures Core B doesn't start processing a tile until Core A has finished transferring it via the internal DMA.

### 3. Locating the Right Documentation in your SDK Package

When you log into the **Renesas Document Portal** or open your downloaded offline SDK archive, prioritize the following documents for implementation: [3, 7]

1. **R-Car V3H User's Manual: Hardware (Section on IMP-X5 / IMPC):** Look up this chapter specifically to understand raw register definitions (e.g., control status registers, memory address mapping, interrupt registers).
2. **IMP Driver API Reference Manual:** Part of the SDK bundle. This covers the actual C language structs used to build commands, manage tile allocations, and issue context queues safely without needing to manually execute ioremap pointer math.
3. **R-Car SDK Sample Applications:** Check the /sw/ or /samples/ path inside the SDK. There are concrete pipeline templates demonstrating how raw pixel buffers are converted into tiles and pushed into the execution queue. [3, 8]

[1] [https://www.renesas.com](https://www.renesas.com/en/about/press-room/renesas-electronics-delivers-r-car-v3h-system-chip-automotive-front-cameras-level-3-and-level-4)

[2] [https://www.renesas.cn](https://www.renesas.cn/zh/document/gde/car-v3h-v3m-guidebook)

[3] [https://community.renesas.com](https://community.renesas.com/automotive-soc/adas/f/adas---forum-1845141225/31733/r-car-v3h-starter-kit-how-to-reduce-the-dram-size)

[4] [https://www.renesas.com](https://www.renesas.com/en/products/automotive-products/automotive-system-chips-socs)

[5] [https://www.renesas.com](https://www.renesas.com/en/document/prb/r-car-v3m-v3h-v3h2-overview-block-diagram-adm5-21-0083)

[6] [https://www.renesas.com](https://www.renesas.com/en/document/prb/r-car-v3m-v3h-v3h2-overview-block-diagram-adm5-21-0083)

[7] [https://www.renesas.com](https://www.renesas.com/en/products/r-car-v3h)

[8] [https://community.renesas.com](https://community.renesas.com/the_vault/automotive-soc/adas/eval_board_and_starter_kit/f/apps-kits-demo-boards---forum/30601/running-sample-applications-from-r-car-v3h-sdk)

[9] [https://www.renesas.com](https://www.renesas.com/en/products/automotive-products/automotive-system-chips-socs/r-car-h3-m3-documents-software)

commands applied on the network like tiling, padding, fusion, slicing, setting impc register banks based on a config file

In a typical Renesas R-Car V3H workflow—especially when using deployment frameworks like the **R-Car AI Studio** , **R-Car Consortium toolchains** , or embedded graph compilers—deep learning networks are translated into a sequence of hardware-level macros. Because the hardware engines (like the **IMPC / IMP-X5** and **CNN accelerator** ) operate on tightly managed local memory layout constraints rather than flat system memory, a configuration file (often compiled from an ONNX/Caffe model or custom YAML/JSON pipeline spec) is parsed into a specific **execution command list** .

The core sequence of hardware and driver-level commands applied to the network layers based on these configuration files includes the following grouped operations:

### 1. Data Tiling &amp; Partitioning Commands

Because the IMPC utilizes internal high-speed local memory ( **CLRAM** ), entire feature maps rarely fit at once. The configuration file specifies how to slice tensors into manageable blocks.

- ALLOC\_TILE\_BUFFER / SET\_TILE\_DIM: Divides a large input tensor into a 2D grid of smaller tiles (e.g., N × M pixel segments). It computes the specific byte offsets and dimensions for each partition based on the CLRAM size boundaries.
- SET\_STRIDE\_PITCH: Configures the input and output memory line strides. This tells the internal DMA how many bytes to skip in main memory to advance to the next vertical line of the tile.
- GENERATE\_TILE\_LOOP: Sets up the loop bounds inside the hardware sequencer to iterate through the columns and rows of tiles covering the entire network layer.

### 2. Border Padding &amp; Margin Control Commands

Kernel operations (like a 3x3 or 5x5 convolution or pooling layer) require boundary pixels that sit outside a given tile's core data.

- SET\_MARGIN\_EXT: Calculates and sets the overlapping margin pixels (e.g., top, bottom, left, right padding) that must be loaded alongside the tile data to prevent edge artifacts.
- CONFIG\_BORDER\_MODE: Dictates how the IMPC hardware handles the absolute edges of the entire image where no neighboring tile exists. The configuration file maps this to commands for:
    - **Zero Padding:** Hardware automatically injects zeros into the boundary registers.
    - **Clamp / Replication Padding:** Hardware replicates the outermost boundary pixel values.
    - **Mirror Padding:** Hardware mirrors the internal pixels across the edge.

### 3. IMPC Register Bank Configuration Commands

The driver maps model layer parameters (weights, activation functions, configurations) directly into the IMPC hardware control and configuration register banks.

- SELECT\_REG\_BANK: Targets specific internal register banks. The IMP-X5 architecture often uses multiple bank sets to allow **double-buffering** —where Register Bank A executes a tile while Register Bank B is being loaded with parameters for the next tile.
- WRITE\_WEIGHT\_BASE / WRITE\_BIAS\_BASE: Writes the physical memory addresses pointing to the parsed layer weights and biases into the dedicated accelerator pointer registers.
- SET\_ACTIVATION\_FUNCTION: Configures the activation function register (e.g., bypassing, enabling ReLU, or setting clamped thresholds like Leaky ReLU / ReLU6) directly at the output stage of the hardware execution pipeline.
- SET\_DATA\_FORMAT: Programs the bit-width and quantization format registers (e.g., switching the hardware math blocks between INT8, INT16, or FP16 based on how the network was quantized).

### 4. Memory Transfer &amp; Synchronization Commands

Once the geometry and registers are prepared, data must be moved and executed in exact sequence without causing memory race conditions.

- DMA\_TRANSFER\_INPUT (MEMCPY\_TO\_CLRAM): Instructs the internal DMA controller to pull a specific tile (plus its calculated padding margins) out of external DDR memory and push it into the targeted CLRAM bank.
- KICK\_CORE\_EXECUTE: Writes to the hardware trigger register to start the vector pipeline or CNN engine on the currently configured register bank.
- SET\_HARDWARE\_BARRIER / WAIT\_INTERRUPT: Inserts synchronization commands into the command list. This forces the hardware sequencer to halt until a specific core finishes its math or a DMA engine finishes pulling the output tile back into system memory (DMA\_TRANSFER\_OUTPUT).

how does the edgeai-tidl-tools implement a similar mechanism [https://github.com/TexasInstruments/edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools)

While the Renesas R-Car V3H depends heavily on lower-level **IMPC register banks and specialized Command Lists (CL)** to manually steer data through internal memory (CLRAM), Texas Instruments handles this concept through a highly automated, open-source middleware architecture within the **edgeai-tidl-tools** framework.

Instead of forcing developers to orchestrate register mapping and manual tile loops, TI utilizes a **Subgraph Offloading and Delegation** mechanism powered by TI Deep Learning (TIDL) custom Execution Providers and Delegates.

### 1. The Subgraph Offloading Mechanism (The "Similar Mechanism")

The equivalent to Renesas' hardware "Command List execution" in TI's world is handled at the deep learning runtime layer (DLRT) via **ONNX Runtime Execution Providers (EP)** or **TFLite Delegates** :

- **Graph Parsing &amp; Partitioning:** During the **Model Compilation** phase on an x86 PC, TI's TIDLCompilationProvider parses the network graph (e.g., an ONNX file). It identifies which subgraphs and layer operators are natively supported by the target TI hardware accelerator (the **C7x DSP / MMA** multi-core engine).
- **Heterogeneous Execution:** Layers that the accelerator cannot handle are scheduled to remain on the main **Cortex-A ARM core** . The segments designated for acceleration are packaged into optimized binary artifacts (.bin or network deploy files).

### 2. Tiling &amp; Hardware-Level Memory Management

Renesas explicitly exposes tile loops and strides to the programmer via the IMP driver API. In contrast, edgeai-tidl-tools abstracts this completely inside the **TIDL-RT (Native Runtime)** C API layer.

- **Automated L1/L2 Caching:** TI’s C7x DSP and MMA (Matrix Multiply Accelerator) hardware feature tight internal local memory boundaries, similar to Renesas' CLRAM. The underlying TIDL-RT engine autonomously computes data partitioning (tiling) based on the specific TI SoC architecture selected (e.g., TDA4VM, AM62A, or AM68A).
- **Hidden Stride &amp; Padding:** Border padding (Zero, Clamp, etc.) and line strides are solved during the offline compilation phase. The toolchain bakes the required internal DMA transfer sizes directly into the layer's binary configuration, removing the need for manual register configuration in the application code.

### 3. Setting "Register Banks" via Configuration Files

Where a Renesas config might parse out specific hardware register shifts, TI relies on **Compilation Options (Runtime Parameters)** passed as a C++ struct or Python dictionary during the graph build process.

Instead of low-level register files, the pipeline is configured using key-value properties:

- **Quantization &amp; Bit-Width Execution:** Instead of writing to a data format register, you pass configuration flags like tensor\_bits = 8 or 16 (enabling INT8 or INT16 acceleration on the MMA).
- **Memory Optimization Flags:** Parameters like untrustedNumParamBufs or balance priorities optimize how weights and biases are buffered across L2 SRAM and external DDR—mirroring the double-buffering logic of Renesas register banks.

### Summary Comparison: Renesas vs. TI Architecture

| **Feature**                | **Renesas R-Car V3H (IMPC)**                                         | **TI Edge AI (edgeai-tidl-tools)**                                                    |
|----------------------------|----------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| **Workflow Entry**         | Custom YAML/JSON parsed to driver C code.                            | Standard ONNX/TFLite model parsed by Compilation Providers.                           |
| **Tiling &amp; Padding**   | Manually calculated via driver or explicit OpenVX nodes.             | Automated; calculated implicitly by the compiler and executed by the TIDL-RT library. |
| **Hardware Orchestration** | Building a macro  **Command List (CL)**  to flip physical registers. | Abstracted  **Subgraph Delegation**  to an Execution Provider/Delegate wheel.         |
| **Fallback Strategy**      | Complex manual routing across IMP Vector Cores vs CPU.               | Out-of-the-box routing: Supported nodes go to C7x/MMA; unsupported go to ARM.         |

The core philosophical difference lies in **where the compilation boundary stops and who owns the hardware orchestration** :

### The Renesas R-Car V3H Paradigm: Imperative Compiling

The Renesas toolchain acts as a **low-level macro compiler** .

- It takes your abstract network layers and aggressively flattens them into an **imperative, sequential command list** of hardware instructions.
- The resulting binary artifact essentially dictates exactly when a hardware register bank swaps, exactly how many bytes the DMA must skip for a tile, and precisely where padding margins are injected into local CLRAM.
- The application developer or raw driver layer must explicitly submit this command list to the hardware queue.

### The TI TIDL Paradigm: Declarative Subgraph Offloading

The TI Edge AI toolchain treats the network as a **topological graph optimization problem** handled by open-source runtimes (like ONNX Runtime or TFLite).

- The TIDLCompilationProvider operates as a graph partitioner. It doesn't create low-level register-manipulation loops. Instead, it looks at the target SoC capability (e.g., the number of C7x DSP cores or Matrix Multiply Accelerator sizes available on a TDA4VM vs. an AM68A) and declares: *"This specific cluster of nodes belongs to TI, and this cluster belongs to the ARM CPU."*
- The resulting .bin artifact is less about raw register sequencing and more of an **optimized intermediate representation (IR) network configuration** .
- At runtime, **TIDL-RT (the native C runtime engine)** takes that IR blueprint alongside your C++ struct or Python configuration dictionary. It is TIDL-RT itself that dynamically manages the internal L1/L2 SRAM memory layout, handles tiling, schedules vector/MMA execution streams, and maps parameters based on your high-level tuning choices (like INT8/INT16 precision or param buffering).

On Qualcomm Snapdragon SoCs utilizing the **Qualcomm AI Runtime (QAIRT)** backend via onnxruntime-qnn, the compilation, memory management, and runtime orchestration map to a highly standardized, modern **Plugin Execution Provider (EP)** architecture.

Instead of requiring custom vendor forks of ONNX Runtime (like older solutions), Qualcomm uses a standalone shared library package (onnxruntime-qnn) that dynamically plugs directly into any upstream, stock ONNX Runtime installation.

### 1. The Subgraph Offloading Equivalent

The equivalent to TI's TIDLCompilationProvider is the **QNN Execution Provider (QNNExecutionProvider)** library.

- **The Mechanism:** Instead of a custom build, you dynamically register the Qualcomm provider at runtime using ort.register\_execution\_provider\_library().
- **Graph Partitioning:** The registered QNN EP automatically intercepts the incoming ONNX model graph. It separates nodes supported by the Qualcomm hardware from those that must fall back to the standard ARM CPU.

### 2. Runtime Memory Management &amp; Tiling

Just as TI shifts the memory management burdern onto **TIDL-RT** , Qualcomm completely shifts layout management, tensor tiling, and padding optimization onto the underlying **Qualcomm AI Runtime (QAIRT) HTP backend** .

- **Opaque Execution:** The high-speed internal vector memory (Hexagon Vector Extensions / HMX / HVX) caching and alignment are handled entirely under the hood.
- **AOT vs. JIT Compilation:** Qualcomm allows you to either compile the model on the fly at runtime (JIT) or use an X64 Windows/Linux PC to compile the model ahead-of-time (AOT) into a specialized serialized binary context. This binary context directly captures the optimized hardware layout for a specific Snapdragon chip generation.

### 3. External Configurations, Quantization &amp; Memory Optimization

Instead of a separate static C++ struct file or independent YAML config file like TI or Renesas require, Qualcomm injects these optimizations directly into ONNX Runtime via session-level **Provider Options** (Python dictionaries or C++ API maps) and **Run Options** .

The onnxruntime-qnn syntax mirrors your description by cleanly separating these parameters into key-value pairs at two specific lifecycle moments:

### A. Session Initialization (Memory &amp; Backend Mapping)

When creating the initial session blueprint, you define the physical path to the hardware backend (e.g., the HTP / Hexagon NPU library) and pass initial quantization/compilation choices:

```
# Equivalent to configuring the target SOC core boundaries
ep_options = {
    'backend_path': qnn_ep.get_qnn_htp_path() # Targets Qualcomm HTP NPU
}

session_options = ort.SessionOptions()
session_options.add_provider_for_devices(selected_ep_devices, ep_options)
session = ort.InferenceSession("model.onnx", sess_options=session_options)
```

### B. Per-Run Options (Dynamic Hardware Overrides)

Qualcomm uniquely allows you to map execution tuning properties—like power profiles, latency thresholds, and structural scheduling—dynamically on a **per-inference call** basis using RunOptions configurations:

```
# Passing runtime optimizations dynamically without breaking the compiled graph
run_options = ort.RunOptions()

# Tells the NPU execution engine to maximize clocks (burst mode)
run_options.add_run_config_entry("qnn.perf_mode", "burst") 

# Sets low-latency remote procedure call parameters for memory streaming
run_options.add_run_config_entry("qnn.rpc_control_latency", "100") 

result = session.run(None, {"input": input_data}, run_options)
```

### Conceptual Alignment

| **Concept**         | **Renesas R-Car V3H**                                  | **TI Edge AI (edgeai-tidl-tools)**          | **Qualcomm AI (onnxruntime-qnn)**                              |
|---------------------|--------------------------------------------------------|---------------------------------------------|----------------------------------------------------------------|
| **Integration**     | Imperative Command List compiler API.                  | Custom vendor fork of ONNX Runtime.         | Standard Plugin EP (Standalone .dll/.so package).              |
| **Compilation**     | Direct register bank mapping and tile loop generation. | TIDLCompilationProvider offline generation. | QNNExecutionProvider (supports runtime JIT or standalone AOT). |
| **Config Delivery** | Raw register macros / Driver API inputs.               | External C++ structs / Python parameters.   | Native ORT ep_options dictionary & RunOptions strings.         |

Apache TVM

**Apache TVM** is an open-source machine learning compiler framework designed to optimize, compile, and run deep learning models efficiently across diverse hardware backends, such as CPUs, GPUs, and specialized hardware accelerators. It lowers high-level models from frameworks like PyTorch, ONNX, and TensorFlow into optimized machine code using intermediate representations (like TensorIR and Relax).

#### How TVM is Incorporated in edgeai-tidl-tools

In Texas Instruments' **edgeai-tidl-tools** (part of TI's Edge AI SDK for Jacinto/Sitara processors like TDA4VM, AM68A, and AM69A), TVM acts as one of the primary high-level runtime and compilation frontends alongside ONNX Runtime and TFLite Runtime.

- **Heterogeneous Graph Offloading:** TVM uses TI’s TIDL BYOC (Bring Your Own Code) backend integration. When you feed a model (e.g., ONNX, MXNet, or PyTorch) into the TVM pipeline, TVM partitions the model graph:
- Supported layers/subgraphs are compiled and offloaded to the hardware C7x DSP / MMA deep learning accelerator via the TIDL runtime.
- Unsupported layers automatically fall back to executing on the host ARM CPU via standard TVM-compiled code.
- **Compilation &amp; Calibration:** During the model compilation stage in edgeai-tidl-tools, the TVM frontend intercepts the graph to apply TIDL-specific quantization, calibration (floating-point to INT8/INT16), and layer validation.
- **Deployment Artifacts:** The process generates a unified deployment package containing the compiled TVM host runtime artifacts alongside the TIDL-subgraph binary artifacts, which can be executed directly on the target EVM using the TVM Python or C++ runtime APIs.

**Apache TVM is not used in onnxruntime-qnn.**

They represent two separate compiler and runtime stacks:

- **What onnxruntime-qnn uses:** It is an Execution Provider (EP) for ONNX Runtime designed specifically for Qualcomm Snapdragon SoCs. It relies directly on Qualcomm's proprietary **Qualcomm AI Engine Direct SDK (QNN SDK)** . The QNN EP translates ONNX graph nodes directly into QNN API calls to execute on Qualcomm hardware backends (such as the Hexagon NPU/HTP, Adreno GPU, or Kryo CPU).
- **Where TVM fits in ONNX Runtime:** While ONNX Runtime has a distinct, community-maintained **TVM Execution Provider** (TVM EP) used to compile subgraphs via TVM's optimization pipeline, it is entirely independent of the QNN Execution Provider.
- **Comparison to TI's approach:** In TI's edgeai-tidl-tools, TI actively maintains a TVM frontend plugin (via TVM BYOC) alongside ONNX Runtime and TFLite frontends to target their C7x DSP/MMA. In contrast, the Qualcomm QNN integration bypasses TVM altogether in favor of their native QNN libraries.

Both the **TVM Execution Provider (TVM EP)** and the **Qualcomm AI Engine Direct Execution Provider (QNN EP)** allow ONNX Runtime (ORT) to delegate graph execution to specialized backends, but their architectures, target environments, and compilation mechanics are fundamentally different.

| **Feature**                     | **TVM Execution Provider (TVM EP)**                                                            | **QNN Execution Provider (QNN EP)**                                                                                |
|---------------------------------|------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| **Primary Focus**               | General-purpose graph-to-code compiler optimization across diverse CPU/GPU targets.            | Native edge hardware acceleration tailored for Qualcomm Snapdragon SoCs.                                           |
| **Backend Target**              | Heterogeneous hardware (x86/ARM CPUs, NVIDIA/AMD GPUs via LLVM/CUDA/OpenCL).                   | Qualcomm NPU (Hexagon HTP/DSP), Adreno GPU, and Kryo CPU.                                                          |
| **Underlying Engine**           | Apache TVM compiler runtime (Relay/Relax IR, TensorIR, Auto-scheduler/Ansor).                  | Qualcomm AI Engine Direct SDK (libQnn*.so).                                                                        |
| **Compilation Workflow**        | JIT or AOT compiles ONNX subgraphs into lowered native machine code/kernels via TVM pipelines. | Converts ONNX nodes into QNN API primitives, composing a Qualcomm graph definition.                                |
| **Quantization &amp; Context**  | Relies on TVM quantization tools or standard ONNX quantized nodes (QLinearConv, etc.).         | Heavily optimized for INT8/INT16 NPU pipelines; supports generating and loading pre-compiled QNN Context Binaries. |
| **Maintenance &amp; Ecosystem** | Community-maintained EP; experimental/preview status.                                          | Actively developed and maintained by Qualcomm and Microsoft for production Windows on ARM and Android deployments. |

#### Implementation &amp; Architectural Contrasts

- **Lowering Mechanism (Codegen vs. Runtime API Translation):**
- **TVM EP:** Translates the assigned ONNX subgraph into TVM's high-level Intermediate Representation (IR). It then applies TVM optimization passes (fusion, layout transformation, tensorization) and generates compiled code (e.g., via LLVM or CUDA backends), wrapping the resulting module inside an ORT custom kernel.
- **QNN EP:** Acts as a bridge to Qualcomm’s proprietary C API. Instead of compiling custom kernels from scratch via an intermediate compiler IR, it maps ONNX operators directly to supported QNN backend operations (or custom op packages) configured via the QNN SDK.
- **Ahead-of-Time (AOT) Caching:**
- **TVM EP:** Relies on compiling and serializing TVM runtime artifacts or caching compiled shared libraries (.so/.dll).
- **QNN EP:** Provides native support for **Context Caching** . It allows compiling the model once on a target or host toolchain, serializing the hardware-specific NPU binary blob, and instantiating the execution graph instantly during subsequent inferences without recompilation overhead.
- **Target Deployment Context:**
- **TVM EP:** Best suited for experimenting with automated kernel tuning (AutoTVM/Ansor), research, or deploying models to non-standard hardware targets without vendor-locked runtimes.
- **QNN EP:** Built for production on-device edge AI (mobile phones, Snapdragon X Elite PCs, automotive platforms) where direct low-power interaction with the Hexagon NPU is required.

The Execution Provider (EP) itself does **not** perform low-level hardware orchestration like tiling, register allocation, or generating microcode.

Instead, the EP acts as an **orchestration and translation bridge** . It offloads the graph to an underlying compiler or driver backend, and that backend handles the low-level hardware optimizations.

How this division of labor works differs between **TVM EP** and **QNN EP** :

#### 1. Qualcomm QNN EP

The QNN Execution Provider delegates hardware compilation to the **Qualcomm AI Engine Direct SDK (QNN)** backend libraries (libQnnHtp.so, etc.).

- **What QNN EP does:**
- **Graph Partitioning:** Inspects the ONNX graph to identify which nodes are supported by the selected QNN backend (e.g., HTP/Hexagon NPU).
- **API Translation:** Converts supported ONNX nodes into Qualcomm QNN API graph constructs (Qnn\_OpConfig\_t).
- **Context Cache Trigger:** Signals the QNN backend to finalize and serialize the graph into a .bin Context Binary.
- **What QNN EP delegates to the QNN Backend/Driver:**
- **Tiling &amp; Slicing:** The Hexagon backend breaks tensors into tiles that fit into tightly coupled vector memory (TCM/VTCM).
- **Padding &amp; Layout Transformation:** Adjusts data alignment to match native HTP vector register widths (e.g., transforming to planar or blocked formats).
- **Register Allocation &amp; Scheduling:** Handled entirely by Qualcomm's proprietary HTP compiler/micro-kernel generator embedded inside the QNN backend.

#### 2. TVM EP

The TVM Execution Provider delegates hardware compilation to the [**Apache TVM**](https://github.com/apache/tvm) **Compiler Pipeline** .

- **What TVM EP does:**
- **Graph Partitioning:** Identifies supported ONNX subgraphs.
- **IR Lowering:** Ingests the ONNX subgraph and converts it into TVM's high-level Intermediate Representation (Relay or Relax).
- **Bridge Execution:** Calls the TVM build API and packages the resulting runtime module into an ONNX Runtime custom kernel.
- **What TVM EP delegates to the TVM Engine:**
- **Tiling, Loop Splitting, &amp; Slicing:** Optimized at the TensorIR level via schedules (or search engines like MetaSchedule/Ansor) to optimize cache locality.
- **Padding &amp; Vectorization:** Transforms memory layouts and matches SIMD/vector intrinsics.
- **Register Offloading &amp; Machine Code Generation:** Handled by TVM lowering to backends like **LLVM** (for CPUs) or **NVCC/CUDA** (for GPUs), which handle register spilling, instruction scheduling, and binary emission (.so, .dll, or target assembly).

#### Summary of Responsibilities

| **Responsibility Level**             | **Who Handles It?**                                                              | **Specific Actions**                                                                         |
|--------------------------------------|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| **Graph-Level**                      | **Execution Provider (EP)**                                                      | Subgraph extraction, fallback handling to CPU, operator translation, triggering compilation. |
| **Optimization &amp; Lowering**      | **Underlying Backend**  *(QNN Engine / TVM)*                                     | Memory layout conversion, operator fusion, loop tiling, buffer allocation.                   |
| **Hardware Execution &amp; Codegen** | **Hardware Compiler / Microcode Generator**  *(LLVM, CUDA, Hexagon Core Engine)* | Register allocation, vectorization, instruction scheduling, emitting the binary artifact.    |

In [Apache TVM](https://github.com/apache/tvm) , the workload is split cleanly between **TVM's intermediate representations (IR)** and the **backend compiler (LLVM or NVCC/CUDA)** :

- **Tiling, Slicing, Padding, &amp; Fusion** are handled entirely by **TVM** .
- **Register Allocation &amp; Instruction Scheduling** are handed off to **LLVM** (for CPU) or **NVCC/PTX AS** (for CUDA/GPU).

#### 1. Where TVM Handles High-Level Transformations

TVM operates across two primary intermediate representation layers: **Relax** (high-level computational graph) and **TensorIR (TIR)** (tensor program/loop-level representation).

- **Operator Fusion:**
- Handled at the graph IR level in src/relax/transform/fuse\_ops.cc and src/relax/transform/fuse\_tir.cc (or legacy Relay in src/relay/transforms/fuse\_ops.cc).
- Groups adjacent element-wise, broadcast, and reduction operators into a single loop nest to prevent redundant memory round-trips.
- **Tiling, Slicing, &amp; Loop Splitting:**
- Implemented as loop transformations on TensorIR in src/tir/schedule/primitive/split\_fuse.cc and src/tir/schedule/primitive/cache\_read\_write.cc.
- Primitives like sch.split(), sch.reorder(), and sch.tile() restructure nested loops so data chunks fit into L1/L2 caches or GPU Shared Memory (\_\_shared\_\_).
- Automated search frameworks like **MetaSchedule** (src/meta\_schedule/) find optimal tile sizes automatically.
- **Vectorization &amp; Padding:**
- Vector loop transformations live in src/tir/transforms/vectorize\_loop.cc.
- Buffer flattening and storage planning (allocating intermediate tile buffers and scratchpads) live in src/tir/transforms/flatten\_buffer.cc and src/tir/transforms/storage\_flatten.cc.

#### 2. Where the Lowering Transitions to LLVM / NVCC

Once TVM has rewritten the loops, placed memory buffers, and injected vector/SIMD pragmas, it translates the AST into backend-specific code:

- **For CPUs (via LLVM):**
- **Code Generator:** Located in src/target/llvm/codegen\_llvm.cc and src/target/llvm/codegen\_cpu.cc.
- TVM directly builds an in-memory LLVM IR module (llvm::Module).
- **LLVM takes over:** TVM calls standard LLVM optimization and target passes. **LLVM’s backend handles register allocation** (e.g., Greedy Register Allocator), instruction selection (SelectionDAG/GlobalISel), instruction scheduling, and emitting machine code (.o or .so).
- **For NVIDIA GPUs (via NVCC / CUDA / PTX):**
- **Code Generator:** Located in src/target/source/codegen\_cuda.cc (emitting CUDA C++ source) or src/target/llvm/codegen\_nvptx.cc (emitting direct PTX via LLVM).
- **NVCC / PTX takes over:** TVM feeds the generated kernel string to nvrtc (NVIDIA Runtime Compilation) or nvcc. The NVIDIA toolchain handles **GPU register allocation** (allocating the 32-bit registers per thread, managing spills to local memory), warp scheduling, and SASS binary generation.

#### Division of Labor

| **Optimization Task**    | **TVM Responsibility**                                                       | **Backend Compiler Responsibility (LLVM / NVCC)**                                            |
|--------------------------|------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| **Operator Fusion**      | Identifies fusable subgraphs and generates unified loop nests.               | Inlines small functions / scalar cleanup.                                                    |
| **Tiling &amp; Slicing** | Splits and reorders multi-dimensional loops; creates scratchpad allocations. | Does not deduce structural multi-dimensional tiling.                                         |
| **Vectorization**        | Emits vector data types (e.g., llvm::VectorType, float4).                    | Maps vector types to native SIMD instructions (AVX-512, Neon).                               |
| **Register Allocation**  | None (emits infinite virtual variables / registers).                         | **Full responsibility:**  Maps virtual variables to physical registers and handles spilling. |
| **Target Emission**      | Builds LLVM IR or CUDA C++ strings.                                          | Compiles IR to native ELF binary, object code, or SASS/CUBIN.                                |

These four compiler transformations are used to optimize tensor computations for cache locality, parallelism, and target hardware execution:

- **Fusion (Operator Fusion):**
- **What it does:** Combines multiple sequential operators (such as Conv2D followed by ReLU and Add) into a single computational kernel or unified loop nest.
- **Why it matters:** Prevents writing intermediate tensors back to slow global DRAM only to read them immediately for the next step, keeping intermediate values hot in CPU/GPU registers or local cache.
- **Loop Splitting (Loop Tiling / Blocking):**
- **What it does:** Breaks a single loop iterating over a large dimension into nested outer and inner loops (e.g., dividing an iteration range of $N=1024$ into an outer loop of $16$ steps and an inner loop of $64$ steps).
- **Why it matters:** Structures iteration into smaller sub-blocks ("tiles") sized to fit precisely within fast, constrained memory spaces like L1 cache, GPU shared memory, or NPU tightly coupled memory (TCM).
- **Slicing:**
- **What it does:** Extracts a sub-tensor along specified dimensions and ranges (e.g., slicing a $128 \times 128$ tile from a $1024 \times 1024$ input activation map).
- **Why it matters:** Complements loop splitting by isolating localized chunks of memory for streaming across high-speed scratchpads or allocating memory dynamically during execution.
- **Flattening (Buffer / Layout Flattening):**
- **What it does:** Maps high-dimensional logical index computations (e.g., a 4D tensor [N, C, H, W]) down to a single 1D physical memory address space using a 1D index formula (e.g., $i = n \cdot S\_0 + c \cdot S\_1 + h \cdot S\_2 + w$).
- **Why it matters:** Physical computer memory is flat and linear. Flattening removes multidimensional abstraction, calculates byte offsets directly, and prepares the code for vectorization and pointer arithmetic in backends like LLVM or CUDA.

After the TVM optimizes on high-level computational graph and TensorIR and before handing it off to LLVM/ NVCC does it generate a xml file with pragmas and an AST representation of the model

**No, TVM does not generate an XML file.**

TVM keeps the representation entirely in-memory as native C++ AST (Abstract Syntax Tree) objects across all lowering stages rather than serializing to XML.

#### 1. The Internal AST Representation

- **C++ AST Hierarchy:** Throughout the compiler pipeline—from Relax (computational graph) down to TensorIR (loop/tensor programs)—the representation is maintained as an AST built from C++ classes derived from tvm::relay::Expr, tvm::relax::Expr, and tvm::tir::Stmt / tvm::tir::PrimExpr.
- **Pragmas as AST Attributes:** Optimization pragmas (such as #pragma unroll, vectorization widths, or custom accelerator directives) are not stored in markup files. Instead, they are represented as:
- **tir::AttrStmt nodes:** AST nodes wrapping loops or statements that carry attribute keys (e.g., attr::pragma\_scope, attr::pragma\_unroll\_explicit).
- **Loop annotations:** Key-value metadata dictionaries attached directly to tir::For AST nodes.

#### 2. How TVM Hands Off to LLVM / NVCC

Because TVM integrates directly with backend compiler APIs in memory, there is no disk or text serialization step:

- **To LLVM (CPU, Hexagon, NVPTX):**
- TVM uses its CodeGenLLVM visitor class to traverse the TensorIR AST in memory.
- As it visits each AST node (tir::For, tir::BufferLoad, tir::Add), it issues direct calls to the **LLVM C++ API** (e.g., llvm::IRBuilder&lt;&gt;), building an in-memory llvm::Module.
- Pragmas in the AST are translated into LLVM metadata nodes (e.g., llvm.loop.unroll.enable or llvm.loop.vectorize.enable) attached to the LLVM basic blocks.
- **To CUDA / NVCC:**
- TVM’s CodeGenCUDA traverses the TensorIR AST and prints an in-memory **CUDA C++ source string** (injecting #pragma unroll or \_\_launch\_bounds\_\_ directly into the string).
- This source string is handed in-memory to **NVRTC** (NVIDIA Runtime Compilation) or piped directly to nvcc to produce PTX or a CUBIN binary.

#### 3. Where XML/Text Might Appear in Similar Toolchains

- **Why XML might come to mind:** Vendor-specific deployment runtimes often export intermediate XML descriptors (for example, Intel OpenVINO exports models as .xml network topologies + .bin weights).
- **TVM's serialization format:** When TVM does serialize an IR AST (e.g., for debugging or caching before codegen), it prints to **TVMScript** (a Python-like domain-specific language) or serializes into **JSON** via its unified object reflection system, never XML.

In renesas rcar v3h starter kit toolchain sdk it had a .xml representation of the intermediate representation of the model before compiling

That explains where that XML workflow comes from. In embedded automotive SDKs like the **Renesas R-Car V3H** , intermediate representation (IR) handling uses a fundamentally different philosophy than modern compiler frameworks like TVM.

#### How Renesas R-Car V3H Handled Intermediate Representation

In the Renesas R-Car SDK (specifically within the **R-Car CNN Tool / R-Car V3H compiler toolchain** targeting the IMP-X5 / CNN-IP hardware accelerator), the toolchain typically operates in decoupled stages:

- **Model Conversion to XML:** Framework frontends (such as Caffe or ONNX parsers) export an **intermediate network graph as an .xml topology file** paired with raw binary weight files.
- **Pragmas, Tiling, and Layer Configuration:** The .xml file explicitly describes:
- Layer-by-layer topologies, connections, and input/output shapes.
- Accelerator-specific partition instructions, quantization parameters (INT8/INT16 scaling and zero points), padding, and buffer stride allocations.
- Memory placement rules defining which activations reside in external DDR versus internal tightly coupled SRAM/local memory.
- **Hardware Generator:** A dedicated Renesas backend generator (or e² studio toolchain plugin) consumes that .xml and generates the final microcode/binary artifacts (.bin / firmware blobs) to be loaded by the R-Car runtime onto the hardware engines.

#### Why TVM Avoids the .xml Step

While Renesas, Intel OpenVINO, and older embedded NPU toolchains use human-readable .xml files as decoupled interchange formats between the converter and the hardware code generator, TVM is built as an **in-memory monolithic optimizing compiler** :

- **Live C++ AST Transformations:** Instead of dumping graph structures and pragmas to an XML schema, TVM keeps everything in memory as structured C++ nodes (Relax expressions and TensorIR statements).
- **Direct Lowering:** Tiling, slicing, loop transformations, and pragmas remain attached directly to the AST as in-memory loop attributes (tir::AttrStmt or annotations on tir::For).
- **Direct Codegen Hand-off:** When lowering completes, TVM directly issues calls into backend APIs (like LLVM C++ builders for CPUs/DSPs or NVRTC/NVCC for GPUs), avoiding the disk I/O, schema parsing, and impedance mismatch of an intermediate XML serialization step.

While [Apache TVM](https://github.com/apache/tvm) does not emit XML, it provides several built-in methods to render and inspect the lowered representation into human-readable text and graphical formats at any optimization stage.

#### 1. TVMScript (The Native Human-Readable Format)

**TVMScript** is TVM's primary human-readable representation. It prints the exact AST (including Relax dataflow graphs, TensorIR loop nests, buffer allocations, tiling, and attached pragmas) as structured Python syntax.

You can inspect the IRModule at any point before backend codegen:

Python

```
# Print to console/terminal
print(mod.script())

# Or explicitly show line numbers and syntax highlighting
mod.show()

# Save to a file to inspect loop structures and pragmas
with open("lowered_model.py", "w") as f:
    f.write(mod.script())
```

In the printed script, you can directly see:

- **Tiling &amp; loops:** for i\_outer, i\_inner in T.grid(...)
- **Buffer allocations:** T.alloc\_buffer(...)
- **Pragmas/Attributes:** Explicit annotations such as T.attr(..., "pragma\_unroll\_explicit", 1) or thread-binding directives.

#### 2. Dumping Lowered Assembly / IR Strings

If you want to view the target representation right before or after handing off to the machine compiler:

- **LLVM IR (Human-Readable Assembly):**

When targeting CPU or DSP through LLVM, you can inspect the generated textual LLVM IR:

Python

```
rt_mod = tvm.build(mod, target="llvm")
# Prints standard textual LLVM IR (.ll format)
print(rt_mod.get_source())
```

- **CUDA / PTX Source:**

When targeting NVIDIA GPUs, you can dump the raw CUDA C++ string:

Python

```
rt_mod = tvm.build(mod, target="cuda")
# Prints the generated CUDA C++ kernel string with all #pragma unroll directives
print(rt_mod.imported_modules[0].get_source())
```

#### 3. Step-by-Step Pass Instrument / Visual Tracing

To see the IR transformation step-by-step (e.g., seeing the graph *before* and *after* FuseOps or loop splitting):

Python

```
# Prints the IR diff after every single transformation pass
with tvm.transform.PassContext(opt_level=3, trace=tvm.ir.transform.PrintIR()):
    compiled_mod = tvm.relax.transform.FuseOps()(mod)
```

#### 4. Graphical Visualizations (Netron &amp; Graphviz)

If you want visual node-and-edge diagrams similar to how XML topologies are visualized:

- **Netron (via ONNX export or Netron-TVM):** If you preserve an ONNX checkpoint prior to lowering, [Netron](https://netron.app/) renders interactive layer blocks, input/output tensors, and attributes.
- **Graphviz / DOT AST Dumps:** TVM provides AST-to-DOT graph exporters (tvm.contrib.relay\_viz or AST dump utilities) that emit .dot files, which can be rendered into SVG/PNG images showing the operator hierarchy and fused subgraphs.

If TVM is going to parse the subgragphs and make the optimizations and handing it over to LLVM / NVCC / CUDA why is there TIDL? Just to make the TIDL subgraphs? and what happens to the unsupported layers by the C7X Cortex or MMA do they get offloaded to ARM and what is the parallel scenario for unsupported layers in qnn

#### 1. Why TIDL Exists if TVM Already Optimizes and Targets LLVM

TIDL (TI Deep Learning) is not just a graph partitioner—it is TI’s proprietary hardware acceleration engine and runtime specifically tailored for the **C7x DSP and MMA (Matrix Multiply Accelerator)** .

- **Standard LLVM Does Not Know the MMA:** While LLVM has standard code-generation targets for generic CPUs and standard DSPs, it has no native concept of TI’s proprietary Matrix Multiply Accelerator (MMA) or the hardware-level streaming engines (SE) and data transfer engines (SA) inside the C7x.
- **Hand-Crafted Micro-Kernels:** TIDL provides heavily hand-tuned, hardware-optimized assembly micro-kernels for convolutions, matrix multiplications, and pooling that maximize MAC utilization and zero-overhead DMA streaming directly into local L2/SRAM scratchpads.
- **Hardware-Centric Quantization &amp; Calibration:** TIDL provides the quantization toolchain to profile activations and generate hardware-accurate scaling factors (e.g., INT8/INT16) required specifically by the C7x/MMA fixed-point pipeline.
- **TVM’s Actual Role Here:** TVM acts as the **high-level frontend and graph partitioner (via BYOC - Bring Your Own Code)** . TVM identifies subgraphs that fit TIDL’s supported layer list, bundles them as black-box TIDL delegate nodes, and passes the heavy lifting of executing those layers to the pre-built TIDL runtime binary.

#### 2. What Happens to Unsupported Layers in edgeai-tidl-tools

When a model contains operators not supported by the C7x/MMA TIDL engine (e.g., non-standard activations, dynamic sequence control, complex slicing):

- **Partitioning &amp; Boundary Extraction:** TVM breaks the model into multiple subgraphs:
- Supported subgraphs $\rightarrow$ marked as TIDL offload partitions.
- Unsupported nodes $\rightarrow$ retained by TVM.
- **Execution Offload (Arm Fallback):**
- The unsupported layers are kept in TVM's computational graph and lowered via **TVM's standard compilation pipeline** (emitting C/LLVM IR).
- They are compiled to native machine code targeting the **host Cortex-A (ARM) core** (or compiled as scalar C7x code, depending on your build target options).
- **Memory Transfers:** At runtime, the TVM graph executor handles intermediate buffer copying: ARM DRAM $\rightarrow$ shared memory buffer $\rightarrow$ C7x/MMA processing $\rightarrow$ return buffer back to ARM.

#### 3. The Parallel Scenario for Unsupported Layers in QNN

In the Qualcomm QNN ecosystem (specifically via the **ONNX Runtime QNN Execution Provider** ), a very similar fallback pattern occurs, but within ONNX Runtime's orchestration architecture:

- **Graph Capability Query:**
- When the model is loaded, the QNN EP calls GetCapability() to evaluate which nodes can run on the chosen backend (e.g., Hexagon NPU / HTP).
- **Partitioning:**
- **Supported Nodes:** Clustered into a single or multiple fused subgraphs, converted into QNN API graph structures, and compiled into the QNN Context Binary (running on the Hexagon NPU).
- **Unsupported Nodes:** Rejected by the QNN EP during the initial capability scan.
- **Execution Fallback:**
- ONNX Runtime’s orchestrator assigns the rejected nodes to the next available provider in its priority list—which is almost always the default **CPU Execution Provider (CPUExecutionProvider)** .
- During inference, ORT executes the QNN subgraph on the Hexagon NPU, transfers the output tensors across the shared RPC/ION memory boundary back to the host, runs the unsupported operators on the Qualcomm Kryo/ARM CPU using standard ORT CPU kernels, and passes the tensors back to the NPU if another accelerated subgraph follows.

Visualizing the qnn lowered artifact

Once the model is compiled by the [QNN Execution Provider](https://github.com/onnxruntime/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md) , the compiled target artifact—known as a **QNN Context Binary** —is a proprietary, serialized machine blob rather than an open AST. You cannot open the raw binary directly in Netron to see inner operations, but you can inspect and visualize the lowered artifact using several complementary methods:

#### 1. Visualizing the EPContext ONNX Wrapper in Netron

When QNN EP performs Ahead-of-Time (AOT) compilation, it can serialize the compiled artifact inside an ONNX wrapper model using an **EPContext** node.

- If you configure ep.context\_file / qnn\_save\_context, ORT exports an ONNX model where the accelerated subgraphs are collapsed into an EPContext operator.
- **Opening in Netron:** You can open this generated .onnx file directly in Netron.
- **What you can inspect:** Netron will display the graph topology showing which boundary nodes remained on CPU vs. which subgraphs were offloaded into the EPContext block, alongside metadata attributes like source (e.g., QNNExecutionProvider), ep\_cache\_context, and hardware backend targets (HTP, DSP).

#### 2. Inspecting the Compiled Binary with Qualcomm QAIRT Utilities

If you have access to the [Qualcomm AI Runtime SDK (QAIRT)](https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK) toolchain alongside onnxruntime-qnn, Qualcomm provides dedicated CLI tools to inspect the compiled context binary:

- **qnn-context-binary-utility:**

Inspects the serialized context binary and dumps the internal structural metadata into human-readable text:

Bash

```
qnn-context-binary-utility --context_binary <model_cache.bin> --info
```

This command outputs the internal graph names, tensor quantization encodings (scale, zero point, bit-width), data types, and buffer alignments selected by the HTP compiler.

#### 3. Dumping the Pre-Compiled QNN Graph to JSON / Netron

To view the exact lowered operators, node fusions, and tensor shapes *after* QNN operator translation but *before* binary generation:

- **QNN Model Converter/Dump Flags:** When compiling through QNN SDK utilities or QNN EP logging, you can enable verbose logging:

Python

```
# In Python session options
session_options.add_session_config_entry("session.qnn.profiling_level", "basic")
```

- **Convert via QNN DLC (Deep Learning Container):**

If you convert the ONNX model using Qualcomm's qnn-onnx-converter, it generates a .dlc intermediate container:

- Run qnn-dlc-viewer --input\_dlc model.dlc to launch a browser-based, interactive visual representation of the lowered graph.
- Run qnn-dlc-info -i model.dlc to output a plain-text table detailing each lowered layer, memory usage, MAC count, and quantization parameters.

Method 2 is not viable

The core difference comes down to **granularity and representation level** :

- **Method 1 (EPContext ONNX Model):** Shows the **coarse macro partition** between host CPU and NPU.
- **Method 3 (Pre-Compiled / Intermediate Graph):** Shows the **fine-grained micro operator graph** inside the offloaded partition before it gets baked into machine binary.

| **Feature**              | **Method 1: EPContext in Netron**                                         | **Method 3: Intermediate Lowered Graph**                                                 |
|--------------------------|---------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| **What it Represents**   | The  **post-compilation deployable wrapper**                              | The  **pre-compilation translated layer graph**                                          |
| **Level of Detail**      | Macro-level (Black box)                                                   | Micro-level (White box)                                                                  |
| **Subgraphs Inside NPU** | Collapsed into a single opaque EPContext operator                         | Expanded into individual nodes (fused GEMMs, activations, layouts)                       |
| **Visible Elements**     | Graph partitions, CPU fallbacks, boundary tensor names/shapes             | Individual QNN/HTP ops, quantization encodings (scale/zero-point), channel layouts       |
| **Primary Use Case**     | Verifying CPU vs. NPU fallback boundaries and packaging deployable models | Verifying operator fusion, unsupported op transformations, and layer-by-layer topologies |

#### What You Actually See in Method 1 (EPContext)

When you instruct onnxruntime-qnn to generate an EPContext model (via session options like ep.context\_file), ORT takes all the contiguous operators supported by the HTP backend, compiles them into a binary blob, and wraps that blob inside an ONNX node called **EPContext** .

When you open this .onnx file in Netron:

- **Inside the partition:** You do **not** see the individual convolutions, layer norms, or matrix multiplies. They are swallowed by the EPContext node, which stores the compiled NPU context in its attributes.
- **Outside the partition:** You see which operators were rejected by the QNN EP and remain on the host CPU.

#### What You Actually See in Method 3 (Pre-Compiled Graph)

Method 3 captures the model **after** ONNX Runtime has partitioned and translated the nodes into the Qualcomm execution dialect, but **before** the Qualcomm compiler packs them into an unreadable machine binary.

In this visualization:

- **The partition is transparent:** You can inspect the actual graph of operators that will run on the NPU.
- **Detailed tensor changes:** You see how ONNX operators were transformed (e.g., how standard ONNX ops were converted to quantized fixed-point primitives, explicit NHWC layout transposes, and fused activations).