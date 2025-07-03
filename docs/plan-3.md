### **Hackathon Plan 3.0: 1-Day GPU-Accelerated Semantic Search**

#### **🎯 Core Vision & Pitch**

Today, we will build a production-grade, GPU-native semantic code search engine
from the ground up. By leveraging Mojo and the full power of high-performance
NVIDIA GPUs, we will replace the core computational kernels of a transformer
model with hand-optimized, tiled, and autotuned code. The goal is to achieve
sub-20ms search latency across a massive codebase, demonstrating a level of
performance that is simply out of reach for standard Python frameworks. This
isn't an academic exercise; it's a real-world system built for speed.

#### **🛠️ Technical Stack & Environment**

- **Cloud Provider:** Lambda Cloud ($400 credit)
- **Hardware:** 2 x High-Performance NVIDIA GPU Instances (e.g., A100 or H100)
- **Core Tech:** Mojo, Python 3.10+
- **Libraries:** torch, transformers (for baseline & tokenizer), tree-sitter,
  numpy, flask
- **Target Model:** GTE-small (384-dim embeddings)

#### **🚀 The 1-Day GPU-First Hackathon Timeline (8 AM \- 10 PM)**

This timeline parallelizes data processing and development across your two GPU
machines.

- **Machine 1 (Dev Machine):** Used for interactive development, kernel coding,
  and UI.
- **Machine 2 (Data/Tune Machine):** Used for heavy, long-running batch jobs.

**Phase 1: Setup & GPU Baseline (Hours 0-3 / 8 AM \- 11 AM)**

- **\[Hour 0\] Cloud Deployment:**
  - **Action:** Spin up two Lambda Cloud GPU instances. Choose an image with the
    NVIDIA drivers and CUDA Toolkit pre-installed if possible.
  - **Action:** ssh into both machines and install the Mojo SDK.
- **\[Hour 1\] Environment Sync:**
  - **Action (Both Machines):** Set up a Python virtual environment. pip install
    torch transformers "tree-sitter\~=0.20" numpy flask. Clone your 48-project
    portfolio to both machines.
- **\[Hour 2\] Data Pipeline Kickoff:**
  - **Action (Machine 2):** Start the "Vectorize-All-The-Things" pipeline (from
    Plan 2.0). Use a Python script with Tree-sitter to parse all 48 projects and
    save the code snippets to a JSON file. This will run for a while in the
    background. **Use the standard PyTorch model on the GPU for this initial
    batch processing.** The goal is to have the full dataset ready for when the
    optimized kernel is done.
- **\[Hour 3\] GPU Baseline & Sanity Check:**
  - **Action (Machine 1):** Write a script to prove you can get a single code
    snippet embedded using the _PyTorch_ model running on the GPU (.to('cuda')).
  - **Action (Machine 1):** Profile the PyTorch forward() pass using
    torch.profiler. Identify the matmul operations as the primary bottleneck.
    This confirms the optimization target and provides a baseline performance
    number to beat.

**Phase 2: Core Kernel Optimization (Hours 4-9 / 11 AM \- 4 PM)**

This is the heart of the hackathon. We will build the GPU kernel layer by layer,
based on the research report.

- **\[Hours 4-5\] Naive GPU Kernel:**
  - **Goal:** Port the matmul operation to a basic, functionally correct Mojo
    GPU kernel.
  - **Action (Machine 1):** Implement the **Global Thread Indexing** pattern
    (Pattern 2.2.2). Create a kernel that takes pointers to three matrices (C,
    A, B) and performs the multiplication. Use the block\_idx.x \* block\_dim.x
    \+ thread\_idx.x logic and a boundary check.
  - **Integration:** Write the Python-Mojo bridge to allocate GPU memory
    (DeviceContext), copy data, launch this naive kernel, and copy the result
    back (Pattern 2.3.1).
  - **Result:** A working, end-to-end embedding process that runs entirely on
    the GPU, even if it's not yet faster than PyTorch.
- **\[Hours 6-8\] The Shared Memory Leap:**
  - **Goal:** Dramatically increase memory bandwidth and compute efficiency by
    implementing shared memory tiling.
  - **Action (Machine 1):** Refactor your naive kernel to use the **GPU Shared
    Memory Tiling Pattern** (Pattern 3.3.1 / 4.4).
    1. Define Shared memory tiles for matrix A and B.
    2. Implement the **Load-Sync-Compute-Store** workflow. Threads in a block
       will cooperatively load data from global memory into the shared tiles.
    3. Use barrier() to ensure the tiles are loaded before computation begins.
    4. Perform the matmul on the tiles using the fast shared memory.
    5. Use another barrier() before the next tile iteration.
  - **Hardcode TILE\_DIM to 16 or 32 for now.**
- **\[Hour 9\] Benchmarking & The "Aha\!" Moment:**
  - **Action (Machine 1):** Create a benchmark.py script.
  - **Benchmark 1:** PyTorch matmul on GPU.
  - **Benchmark 2:** Your naive Mojo GPU kernel.
  - **Benchmark 3:** Your **tiled Mojo GPU kernel**.
  - **Result:** You should now see a significant performance jump with the tiled
    version. This graph is a cornerstone of your final demo.

**Phase 3: Autotuning, Integration & Demo (Hours 10-16 / 4 PM \- 10 PM)**

- **\[Hours 10-12\] Autotuning on Autopilot:**
  - **Goal:** Find the absolute optimal tile size for the Lambda Cloud GPU
    architecture without manual guesswork.
  - **Action (Machine 2):** The full dataset should be finished processing by
    now. Use this machine to run the **Autotuned Kernel** pattern (Pattern 4.5).
    Create a script that uses @adaptive and autotune.search to test your tiled
    matmul kernel with a range of TILE\_DIM values (e.g., 8, 16, 32, 64). Let
    this run. It may take an hour or two, but it's working in parallel.
  - **Action (Machine 1):** While the autotuner runs, begin building the search
    interface. Create the simple Flask API and the vanilla JS frontend. Get it
    working with your _currently hardcoded_ tiled kernel.
- **\[Hour 13\] Final Kernel Integration:**
  - **Action (Machine 2):** The autotuner should have finished and identified
    the fastest TILE\_DIM.
  - **Action (Machine 1):** Update your kernel on the dev machine with this
    optimal TILE\_DIM. Your search engine is now running at its peak possible
    performance.
- **\[Hours 14-16\] Full System Test & Demo Prep:**
  - **Action (Machine 1):** Load the full 100k+ vector corpus generated by
    Machine 2 into your search engine. Test the end-to-end performance and user
    experience.
  - **Action (Machine 1):** Prepare your final presentation.
    - Record a "bake-off" video: Standard VSCode search vs. your instant
      semantic search.
    - Create the slide with the benchmark graph: PyTorch GPU \-\> Naive Mojo GPU
      \-\> Tiled Mojo GPU \-\> **Autotuned Tiled Mojo GPU**. This tells a
      powerful story of methodical optimization.
    - Highlight that the entire process, from cloud deployment to an autotuned
      kernel, was completed in a single day.

#### **📊 Success Metrics & Demo Story**

- **Primary KPI:** End-to-end search latency \< 20ms over 100,000+ code
  snippets.
- **Secondary KPI:** A clear, compelling benchmark graph demonstrating a 5-10x
  (or more) speedup for the core matmul kernel over the baseline.
- **Demo Narrative:** We didn't just build a search tool; we engineered a
  high-performance compute kernel. We started with a top-tier GPU and a standard
  PyTorch model, but it wasn't fast enough. By systematically applying advanced
  Mojo optimization techniques—from basic GPU threading to shared memory tiling
  and finally to automated tuning—we unlocked the true potential of the
  hardware, delivering an experience that feels instantaneous.

#### About Autotuning

It's a key detail, and the systems are new. Here’s the simple breakdown:

**Autotuning is a feature of Mojo/MAX, which you will run on the Lambda Cloud
hardware.**

Let's clarify the roles:

1. **Lambda Cloud:** Provides the **hardware**. It gives you a powerful NVIDIA
   A100 or H100 GPU. Think of it as providing a world-class racetrack. It
   doesn't know or care what car you're driving; it just provides the pristine
   asphalt.

2. **Mojo/MAX:** Provides the **software and the tuning tools**. Mojo's
   `autotune` library is the expert driver and pit crew. It takes your code (the
   "car") and runs it on the racetrack (the Lambda Cloud GPU) over and over,
   trying different settings to see what makes it go fastest _on that specific
   track_.

### How It Works in Practice

The autotuning step doesn't happen "via" the cloud provider in the sense that
they offer a service for it. You will run a Mojo script _on your Lambda Cloud
machine_ that performs the tuning.

Here is what the code pattern looks like, which an LLM code-generator can easily
implement. This is a simplified version of what you'll do in **Phase 3 (Hours
10-12)** of your plan.

**1. Your Tunable Kernel (`matmul_kernel.mojo`)**

You'll write your tiled matrix multiplication kernel to accept the `TILE_DIM` as
a compile-time parameter.

```mojo
from autotune import autotune_select
from sys.intrinsics import get_env_var
from memory import alloc, memset_zero
from tensor import Tensor
from device import Device, host, cuda

// This kernel is parameterized by TILE_DIM
fn tiled_matmul_kernel[
    T: DType,
    TILE_DIM: Int
](c: Tensor[T], a: Tensor[T], b: Tensor[T]):
    // ... your full tiled matmul logic from Phase 2 goes here ...
    // It will use TILE_DIM throughout its implementation.
    ...
```

**2. Your Tuning Script (`tune.py` or `tune.mojo`)**

You'll write a separate script that imports the kernel and uses the `autotune`
library to test different values for `TILE_DIM`.

```python
import mojo_interop
from matmul_kernel import tiled_matmul_kernel
from autotune import search

# This is the function we want to tune.
# We wrap our kernel call in it.
def benchmark_matmul(tile_dim: int):
    # Setup your input tensors (a, b) and output tensor (c) on the GPU
    # ... tensor allocation and data copying logic ...

    # Time the kernel execution
    start_time = now()
    tiled_matmul_kernel[DType.float32, tile_dim](c, a, b)
    end_time = now()

    # Return the execution time
    return end_time - start_time

# The magic happens here!
# Tell Mojo to search over these specific values for the 'tile_dim' parameter.
best_config = search(benchmark_matmul, {"tile_dim": [8, 16, 32, 64, 128]})

# The autotuner runs benchmark_matmul with each tile_dim and finds the fastest.
print("Autotuning complete!")
print(f"The fastest TILE_DIM for this GPU is: {best_config['tile_dim']}")
```

### Your Workflow on Hackathon Day:

1. You will write `matmul_kernel.mojo` and `tune.py` on **Machine 1 (Dev
   Machine)**.
2. You will then `scp` (secure copy) these files over to **Machine 2 (Data/Tune
   Machine)**.
3. On **Machine 2**, you will run `python tune.py`.
4. You'll let it run for an hour or so. It will print out the best `TILE_DIM`.
5. You take that optimal value (let's say it prints `32`) and you hardcode it
   into the final version of the kernel you use in your application.

So, in short: **Mojo provides the brains (`autotune`), and Lambda Cloud provides
the brawn (the GPU).**
