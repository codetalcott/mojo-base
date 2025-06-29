# **An Expert's Guide to High-Performance Kernel Optimization in Mojo: Code Patterns for SIMD, Parallelization, and Tiling**

## Table of Contents

### Core Sections
- [Introduction](#introduction)
- [Section 1: Mastering Instruction-Level Parallelism with SIMD](#section-1-mastering-instruction-level-parallelism-with-simd)
  - [1.1. Foundational SIMD Patterns](#11-foundational-simd-patterns)
  - [1.2. Memory-to-Vector Patterns: Loading and Storing](#12-memory-to-vector-patterns-loading-and-storing)
  - [1.3. Advanced SIMD Algorithm Patterns](#13-advanced-simd-algorithm-patterns)
  - [1.4. Parameterized and Autotuned SIMD](#14-parameterized-and-autotuned-simd)
- [Section 2: Harnessing Task and Data Parallelism](#section-2-harnessing-task-and-data-parallelism)
  - [2.1. CPU-Level Parallelism with parallelize](#21-cpu-level-parallelism-with-parallelize)
  - [2.2. GPU Kernel Fundamentals: Grid, Block, and Thread Patterns](#22-gpu-kernel-fundamentals-grid-block-and-thread-patterns)
  - [2.3. Asynchronous Execution and Data Transfer Patterns](#23-asynchronous-execution-and-data-transfer-patterns)
- [Section 3: Optimizing Memory Access with Tiling (Loop Blocking)](#section-3-optimizing-memory-access-with-tiling-loop-blocking)
  - [3.1. The Canonical Tiling Pattern for CPU](#31-the-canonical-tiling-pattern-for-cpu)
  - [3.2. Idiomatic Tiling in Mojo with Parameterized Functions](#32-idiomatic-tiling-in-mojo-with-parameterized-functions)
  - [3.3. The GPU Shared Memory Tiling Pattern](#33-the-gpu-shared-memory-tiling-pattern)
- [Section 4: Synthesis: A Fully Optimized Matrix Multiplication Kernel](#section-4-synthesis-a-fully-optimized-matrix-multiplication-kernel)
- [Conclusion](#conclusion)
- [Works cited](#works-cited)

## Performance Patterns Quick Reference

### SIMD Optimization Progression
| Pattern | Technique | Performance Gain | Key Function |
|---------|-----------|-----------------|--------------|
| **Basic** | Element-wise operations | Baseline vectorization | `simd_arithmetic_patterns()` |
| **Memory** | Aligned load/store | Memory bandwidth utilization | `load_store_patterns()` |
| **Advanced** | Reductions & conditional logic | Complex algorithm vectorization | `advanced_simd_patterns()` |
| **Hardware-Agnostic** | Native width detection | Portable vectorization | `vectorized_add[nelts]()` |

### Parallelization Progression
| Pattern | Target | Technique | Key Function |
|---------|--------|-----------|--------------|
| **CPU Multi-core** | Task parallelism | `parallelize[]` | `parallel_cpu_work()` |
| **GPU Basic** | Massive parallelism | Grid/block/thread model | `vector_add_kernel()` |
| **GPU Advanced** | Memory hierarchy | Shared memory + tiling | `tiled_gpu_kernel()` |

### Matrix Multiplication Evolution
| Implementation | Optimization Focus | Performance Characteristics | Function Name |
|---------------|-------------------|---------------------------|---------------|
| **Naive** | Baseline | O(n³) with poor cache behavior | `matmul_naive()` |
| **Vectorized** | SIMD | 4-8x speedup from vectorization | `matmul_vectorized()` |
| **Parallel** | Multi-core CPU | Additional 4-16x from parallelization | `matmul_vectorized_parallel()` |
| **GPU Tiled** | Memory hierarchy | 10-100x speedup on appropriate workloads | `matmul_gpu_tiled()` |
| **Autotuned** | Hardware-portable | Optimal performance across platforms | `matmul_autotuned()` |

### Key Language Constructs for Performance
| Construct | Purpose | Performance Impact |
|-----------|---------|-------------------|
| `fn` | Type safety & optimization | Enables aggressive compiler optimizations |
| `SIMD[DType, width]` | Vectorization | Direct mapping to hardware vector instructions |
| `@parameter` | Compile-time specialization | Zero-cost abstractions |
| `parallelize[]` | CPU parallelism | Automatic work distribution |
| `@kernel` + `DeviceContext` | GPU compute | Massive parallel execution |
| `Shared[...]` | GPU shared memory | Fast on-chip memory access |
| `autotune()` | Parameter optimization | Hardware-specific tuning |

### **Introduction** {#introduction}

Mojo is a programming language engineered to address a foundational challenge in
modern computing: unifying the high-level usability of languages like Python
with the low-level performance and control of systems languages like C++, Rust,
and CUDA.1 It is not merely a "faster Python" or "Python++"; it is a new
language designed from first principles on the Multi-Level Intermediate
Representation (MLIR) compiler infrastructure.5 This design choice makes Mojo
uniquely suited for the demands of artificial intelligence (AI), machine
learning (ML), and high-performance computing (HPC) workloads, which require
efficient execution across heterogeneous hardware, including multi-core CPUs and
massively parallel GPUs.1

A central tenet of Mojo's performance philosophy is the principle of explicit
programmer control. Rather than relying solely on opaque, "heroic" compiler
heuristics to discover optimization opportunities, Mojo provides a rich set of
language constructs that empower developers to directly command hardware
behavior at compile time.7 This is achieved through a powerful metaprogramming
system that includes features like strictly-typed

fn functions, the @parameter decorator for compile-time specialization, the
alias keyword for compile-time constants, and the first-class SIMD type for
vectorization. These tools allow programmers to write code that is
simultaneously abstract, performance-portable, and capable of achieving
performance competitive with hand-tuned C++ and CUDA.1

The most effective optimization strategies in Mojo arise from the synergy of its
core features. The advanced techniques detailed in this guide—Single
Instruction, Multiple Data (SIMD), Parallelization, and Tiling—are not isolated
tricks but are deeply interconnected components of a holistic performance
strategy. Tiling, a memory optimization technique, improves data locality in
caches, which in turn prevents parallel threads and SIMD lanes from stalling
while waiting for data from main memory. This synergistic relationship can
produce a multiplicative, rather than additive, performance improvement.12 This
guide will build these concepts layer by layer, demonstrating how to combine
them to construct a fully optimized computational kernel.

The following table provides a high-level summary of the key Mojo primitives for
performance optimization, serving as a roadmap for the detailed code patterns
presented in this guide.

Table 1: Mojo Optimization Primitives and Their Purpose

| Construct | Role in Optimization                                                                                                                                                                       | Relevant Sources |
| :-------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------- |
| fn        | Enforces strict type-checking and value semantics, enabling aggressive compiler optimizations and preventing runtime errors. The preferred function type for performance-critical kernels. | 1                |

| SIMD | Represents a hardware vector type for explicit instruction-level
parallelism (data parallelism). The fundamental building block for
vectorization. | 9 |

| parallelize | A high-level standard library function for distributing
data-parallel tasks across available CPU cores. Simplifies task-based
parallelism on the CPU. | 20 |

| @kernel / DeviceContext | The decorator and context class for defining and
launching GPU-accelerated functions. Manages grid/block/thread hierarchy and
device interaction. | 15 |

| tile / Tiling Logic | A memory optimization technique (loop blocking) to
improve cache/memory locality. Can be implemented with helper functions or
explicit nested loops. | 8 |

| Shared[...] | Declares fast, on-chip shared memory for a cooperative thread
block on the GPU, essential for efficient tiling in GPU kernels. | 23 |

| @parameter | Enables compile-time metaprogramming, allowing functions to be
specialized for constant values (like tile sizes or SIMD widths), which unlocks
significant compiler optimizations. | 1 |

| alias | Defines compile-time constants. Crucial for parameterizing optimized
functions and providing values to @parameter constructs. | 27 |

| autotune | A utility to benchmark and programmatically select the best
compile-time parameters (e.g., tile sizes) for the specific target hardware,
ensuring performance portability. | 1 |

## **Section 1: Mastering Instruction-Level Parallelism with SIMD** {#section-1-mastering-instruction-level-parallelism-with-simd}

Instruction-level parallelism (ILP) is a class of parallel execution where
multiple instructions are processed simultaneously by a processor.30 The most
common and powerful form of ILP available to programmers is Single Instruction,
Multiple Data (SIMD), a hardware feature that allows a single operation (e.g.,
an addition or multiplication) to be applied to an entire vector of data
elements in one clock cycle.17 Mojo elevates SIMD from a low-level assembly
concern to a first-class language feature through its

SIMD type, providing an architecture-agnostic abstraction over hardware vector
registers.19

### **1.1. Foundational SIMD Patterns** {#11-foundational-simd-patterns}

The foundation of SIMD programming in Mojo is the declaration, initialization,
and manipulation of SIMD objects. These patterns are designed to be intuitive
and map directly to efficient underlying hardware instructions.

**Pattern 1.1.1: SIMD Declaration and Initialization**

A SIMD vector is declared with a specific element data type (DType) and a fixed
size. It can be initialized in several ways, each suited for different
scenarios.

```mojo
from DType import float32, int32
from builtin import SIMD

fn simd_initialization_patterns():
    # Pattern: Default initialization.
    # All elements are initialized to zero.
    # This is efficient as it can map to a register-clearing instruction.
    let zeros = SIMD[float32, 4]()
    print("Default (zeros):", zeros) # [0.0, 0.0, 0.0, 0.0]

    # Pattern: Splat initialization.
    # A single scalar value is "splatted" or broadcast across all elements.
    # This is useful for creating constant vectors for arithmetic operations.
    let ones = SIMD[float32, 4](1.0)
    print("Splat (ones):", ones) # [1.0, 1.0, 1.0, 1.0]

    # Pattern: Element-wise initialization.
    # Each element is specified explicitly. The number of arguments must
    # match the SIMD size.
    let sequence = SIMD[int32, 4](10, 20, 30, 40)
    print("Element-wise:", sequence) # [10, 20, 30, 40]
```

**Pattern 1.1.2: Element-wise Arithmetic and Comparisons**

Once initialized, standard arithmetic and comparison operators work element-wise
on SIMD vectors, generating a new SIMD vector as the result. This is the core of
SIMD's power, replacing a loop of scalar operations with a single vector
instruction.17

```mojo
from DType import float32, bool
from builtin import SIMD

fn simd_arithmetic_patterns():
    let a = SIMD[float32, 4](1.0, 2.0, 3.0, 4.0)
    let b = SIMD[float32, 4](5.0, 6.0, 7.0, 8.0)
    let c = SIMD[float32, 4](2.0) # Splatted scalar

    # Pattern: Vector-Vector Addition
    let sum_vec = a + b
    print("Vector-Vector Add:", sum_vec) # [6.0, 8.0, 10.0, 12.0]

    # Pattern: Vector-Scalar Multiplication
    # The scalar 'c' is effectively broadcast for the operation.
    let product_vec = a * c
    print("Vector-Scalar Mul:", product_vec) # [2.0, 4.0, 6.0, 8.0]

    # Pattern: Element-wise Comparison
    # This returns a SIMD vector of booleans (a mask).
    let comparison_mask: SIMD[bool, 4] = a > c
    print("Comparison Mask:", comparison_mask) # [False, False, True, True]
```

A subtle but powerful design decision in Mojo is that scalar types like Float32
are aliases for SIMD.17 This unification is profound because it allows a single
function to operate seamlessly on both scalars and vectors. For example, a
function defined to accept

SIMD can be called with a Float32 (where N=1) or a SIMD[float32, 8] (where N=8)
without any change to the function's code. This design principle dramatically
simplifies the creation of vectorized libraries and allows developers to
prototype algorithms with scalar values and later scale them to full vector
width simply by changing the type definition, directly supporting Mojo's goal of
progressive optimization.

### **1.2. Memory-to-Vector Patterns: Loading and Storing** {#12-memory-to-vector-patterns-loading-and-storing}

High-performance kernels spend much of their time moving data between main
memory and compute registers. For SIMD, this means loading data from a
contiguous memory block into a SIMD vector and storing the result back.
Performing this efficiently requires careful attention to memory alignment.
Modern CPUs read memory in chunks (cache lines, typically 64 bytes). If a vector
load or store crosses a cache line boundary, the hardware may need to perform
two separate memory accesses, effectively halving the memory bandwidth and
stalling the computation.18

**Pattern 1.2.1: Aligned Memory Allocation and Pointer Management**

Mojo provides tools to ensure that memory allocated for SIMD operations is
correctly aligned. The DTypePointer is the primary type for interacting with
memory buffers intended for SIMD operations.17

```mojo
from memory import DTypePointer, aligned_alloc
from DType import float32
from builtin import SIMD

fn memory_management_patterns():
    alias simd_width = 4
    alias alignment = simd_width * 4 # Alignment in bytes for 4 float32s (16 bytes)

    # Pattern: Allocate aligned memory.
    # This guarantees the returned pointer's address is a multiple of 'alignment'.
    let data_ptr = DTypePointer[float32].aligned_alloc(simd_width, alignment)

    #... use the pointer...

    # Pattern: Free the allocated memory.
    data_ptr.free()
```

**Pattern 1.2.2: Vectorized Load and Store**

The simd\_load and simd\_store methods are the bridge between memory
(DTypePointer) and registers (SIMD).

```mojo
from memory import DTypePointer
from DType import int32
from builtin import SIMD

fn load_store_patterns():
    alias vector_size = 8
    alias simd_width = 4

    # Allocate memory for 8 integers.
    let data_ptr = DTypePointer[int32].alloc(vector_size)
    for i in range(vector_size):
        data_ptr.store(i, i)

    # Pattern: Load a vector from memory at a specific offset.
    # Load the first 'simd_width' elements (elements 0-3).
    var vec1 = data_ptr.simd_load[simd_width](0)
    print("Loaded vec1:", vec1) # [0, 1, 2, 3]

    # Load the next 'simd_width' elements (elements 4-7).
    var vec2 = data_ptr.simd_load[simd_width](simd_width)
    print("Loaded vec2:", vec2) # [4, 5, 6, 7]

    # Perform a SIMD operation.
    let result_vec = vec1 + vec2
    print("Result vec:", result_vec) # [4, 6, 8, 10]

    # Pattern: Store a vector back to memory.
    # Store the result back into the first 4 elements of the buffer.
    data_ptr.simd_store(0, result_vec)

    # Verify the stored data.
    let final_vec = data_ptr.simd_load[simd_width](0)
    print("Stored result:", final_vec) # [4, 6, 8, 10]

    data_ptr.free()
```

### **1.3. Advanced SIMD Algorithm Patterns** {#13-advanced-simd-algorithm-patterns}

Beyond basic arithmetic, many algorithms require more sophisticated vector
manipulations. Mojo provides a rich set of SIMD intrinsics for reductions,
permutations, and conditional logic, enabling the vectorization of complex
control flow.

**Pattern 1.3.1: Reductions, Permutations, and Conditional Selection**

These patterns are essential for tasks like calculating dot products, applying
filters, or reordering data for specific computational stages.

```mojo
from DType import float32, int32, bool
from builtin import SIMD
from math import iota

fn advanced_simd_patterns():
    let f_vec = iota[float32, 8](0.0) # [0.0, 1.0,..., 7.0]
    let i_vec = SIMD[int32, 4](10, 20, 30, 40)

    # Pattern: Reduction.
    # 'reduce_add' sums all elements in the vector horizontally.
    # This is critical for dot products and calculating norms.
    let sum_val = f_vec.reduce_add()
    print("reduce_add:", sum_val) # 28.0

    # Pattern: Shuffle (Permutation).
    # Reorders elements based on compile-time indices.
    # Here, it reverses the vector.
    let shuffled_vec = i_vec.shuffle()
    print("shuffle:", shuffled_vec) # [40, 30, 20, 10]

    # Pattern: Conditional Selection (branch-free logic).
    # Creates a new vector by choosing elements from 'yes_vec' or 'no_vec'
    # based on the boolean 'mask'. This avoids costly conditional branches
    # inside a loop.
    let mask = SIMD[bool, 4](True, False, True, False)
    let yes_vec = SIMD[int32, 4](1, 2, 3, 4)
    let no_vec = SIMD[int32, 4](-1, -2, -3, -4)
    let selected_vec = mask.select(yes_vec, no_vec)
    print("select:", selected_vec) # [1, -2, 3, -4]
```

**Pattern 1.3.2: Case Study \- Parallel Prefix Sum**

The prefix sum (or scan) operation is a classic HPC algorithm that is
challenging to vectorize due to its inherent data dependency (output\[i\]
depends on output\[i-1\]). A naive loop is sequential. However, a parallel scan
algorithm can be implemented using SIMD, computing the result in a logarithmic
number of steps. This pattern showcases the power of combining SIMD shifts with
metaprogramming (unroll) for maximum efficiency.33

```mojo
from DType import uint8
from builtin import SIMD
from algorithm import unroll

# Computes an inclusive prefix sum on a single SIMD vector.
fn prefix_sum_simd_chunk(inout chunk: SIMD[uint8, 8]):
    # This parameterized function will be unrolled by the compiler.
    # The 'i' becomes a compile-time constant for each unrolled step.
    @parameter
    fn add_shifted[i: Int]():
        # The shift amount (1, 2, 4) is a compile-time constant,
        # which allows the compiler to generate the most efficient
        # shift instruction.
        chunk += chunk.shift_right[1 << i]()

    # Unroll the operation log2(8) = 3 times.
    unroll[3, add_shifted]()

fn simd_prefix_sum_example():
    var data = SIMD[uint8, 8](1, 1, 1, 1, 1, 1, 1, 1)
    print("Original data:", data)

    prefix_sum_simd_chunk(data)
    print("Prefix sum:", data) # [1, 2, 3, 4, 5, 6, 7, 8]
```

### **1.4. Parameterized and Autotuned SIMD** {#14-parameterized-and-autotuned-simd}

To write performance-portable code, it is crucial to avoid hardcoding
hardware-specific values like the SIMD vector width. A vector of 4 float32s (128
bits) is efficient on one architecture, while another might support 8 float32s
(256 bits, AVX2) or 16 (512 bits, AVX-512). Mojo's metaprogramming features
allow the code to adapt to the target hardware at compile time.

**Pattern 1.4.1: Hardware-Agnostic Vectorization**

This pattern uses simdwidthof to query the native vector width and parameterizes
the kernel to use that width, ensuring optimal performance across different
machines.

```mojo
from DType import float32
from builtin import SIMD, simdwidthof
from memory import DTypePointer

# This function is parameterized by the SIMD width.
@parameter
fn vectorized_add[nelts: Int](
    c_ptr: DTypePointer[float32],
    a_ptr: DTypePointer[float32],
    b_ptr: DTypePointer[float32],
    size: Int
):
    for i in range(0, size, nelts):
        # Load vectors of the native width.
        let a_vec = a_ptr.simd_load[nelts](i)
        let b_vec = b_ptr.simd_load[nelts](i)
        # Store the result.
        c_ptr.simd_store(i, a_vec + b_vec)

fn main_vectorized_add():
    # Determine the native SIMD width for float32 on this machine.
    alias native_width = simdwidthof[float32]()
    print("Using native SIMD width:", native_width)

    let size = 1024
    let a = DTypePointer[float32].alloc(size)
    let b = DTypePointer[float32].alloc(size)
    let c = DTypePointer[float32].alloc(size)
    # Initialize a and b...

    # The compiler generates a specialized version of vectorized_add
    # for the determined native_width.
    vectorized_add[native_width](c, a, b, size)

    #...
    a.free()
    b.free()
    c.free()
```

The use of compile-time parameters, as seen in the prefix-sum example's
shift_right[1 << i]() call, is a cornerstone of Mojo optimization.33 A standard
loop like

for i in range(N): vec.shift(i) would compel the compiler to generate code for a
"shift by variable amount," which is often a slower instruction on many CPUs
than a "shift by constant amount." By using @parameter and unroll, the loop is
eliminated at compile time, and the compiler sees a sequence of distinct calls:
vec.shift(1), vec.shift(2), vec.shift(4), etc. Each of these has a constant
shift amount, allowing the compiler to emit the fastest possible machine code
for each specific operation. This pattern of using compile-time constants to
enable strength reduction is fundamental to how Mojo achieves low-level
performance from high-level code.

## **Section 2: Harnessing Task and Data Parallelism** {#section-2-harnessing-task-and-data-parallelism}

While SIMD exploits parallelism within a single processor core, modern systems
gain most of their computational power from having multiple cores (on a CPU) or
thousands of simple processing units (on a GPU). Mojo provides distinct but
complementary abstractions for harnessing these two forms of hardware
parallelism.

### **2.1. CPU-Level Parallelism with parallelize** {#21-cpu-level-parallelism-with-parallelize}

For many data-parallel problems running on a CPU, Mojo's standard library offers
the algorithm.functional.parallelize function. This high-level utility
simplifies the process of distributing independent computations across all
available CPU cores, handling thread management and load balancing
automatically.2

**Pattern 2.1.1: Basic Data-Parallel Loop**

This pattern demonstrates how to parallelize a simple loop over a collection of
data. The parallelize function takes a callable (often a lambda) and the number
of work items. It then spawns a pool of worker threads and assigns each one a
portion of the work, blocking until all tasks are complete.20

```mojo
from algorithm import parallelize
from collections import List

fn parallel_cpu_work():
    var data = List[Int](1, 2, 3, 4, 5, 6, 7, 8)
    var results = List[Int](0, 0, 0, 0, 0, 0, 0, 0)
    let n_items = data.count()

    # Define the work to be done for a single item 'i'.
    # This function captures 'data' and 'results' by reference.
    fn process_item(i: Int):
        results[i] = data[i] * 2

    # Distribute the 'process_item' function across all available CPU cores.
    # 'parallelize' will call process_item(0), process_item(1),..., process_item(7)
    # in parallel.
    parallelize[process_item](n_items)

    print("Parallel results:", results)
```

**Pattern 2.1.2: Safe Parallel Reduction**

A common pitfall in parallel programming is a data race, where multiple threads
attempt to modify the same shared memory location simultaneously without
synchronization. A naive parallel sum like parallelize(lambda i: total\_sum \+=
data\[i\], n) will produce incorrect results because the \+= operation is not
atomic.20 The correct pattern requires each thread to accumulate its result into
a local variable, followed by a final, sequential reduction.

```mojo
# Note: This pattern is conceptual. Mojo's current standard library
# may require more explicit thread-local storage mechanisms for this to be
# robust. The core idea is to avoid direct writes to a shared accumulator.

from algorithm import parallelize
from collections import List
# This is a hypothetical utility for thread-safe operations.
# from threading import AtomicInt

fn safe_parallel_reduction():
    let data = List[Int](1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    let n_items = data.count()
    # In a real implementation, this would be an array of atomics or
    # a more sophisticated reduction object.
    var total_sum = 0

    # A more robust pattern would involve a thread-local accumulator
    # and a final reduction step. For simplicity, this demonstrates the
    # core issue. A truly safe implementation in current Mojo would be more complex.
    # The following is UNSAFE due to data races on 'total_sum'.
    # fn unsafe_add(i: Int):
    #     total_sum += data[i] # RACE CONDITION
    # parallelize[unsafe_add](n_items)

    print("A safe parallel reduction requires thread-local storage and a final reduction step.")
```

It is critical to distinguish Mojo's parallelize from vectorize. While both
achieve parallelism, they operate at different hardware levels. vectorize
implements instruction-level parallelism (SIMD) _within a single CPU core_.
parallelize implements task-level parallelism by distributing work _across
multiple CPU cores_.20 The most effective optimization strategies often combine
both:

parallelize is used to divide a large problem into coarse-grained chunks, and
vectorize is used within the processing of each chunk to accelerate the
fine-grained, computationally intensive loops. This hierarchical application of
parallelism is key to maximizing hardware utilization.

### **2.2. GPU Kernel Fundamentals: Grid, Block, and Thread Patterns** {#22-gpu-kernel-fundamentals-grid-block-and-thread-patterns}

GPU programming operates on a different model from CPU parallelism. Instead of a
small number of powerful cores, a GPU has thousands of simpler cores. A program,
called a **kernel**, is launched to run simultaneously on a vast number of
threads. These threads are organized into a hierarchy: threads are grouped into
**blocks**, and the collection of all blocks is called the **grid**.15

**Pattern 2.2.1: Kernel Definition and Launch**

A GPU kernel in Mojo is a standard fn (which cannot raise errors) that is
launched via a DeviceContext object. The grid\_dim and block\_dim parameters
control the size and shape of the thread hierarchy.15

```mojo
from gpu.host import DeviceContext
from gpu.id import thread_idx, block_idx

# A simple GPU kernel that runs on every thread.
fn my_gpu_kernel():
    print(
        "Hello from block:", block_idx.x,
        "thread:", thread_idx.x
    )

fn launch_gpu_kernel():
    try:
        var ctx = DeviceContext()
        # Launch a grid with 2 blocks, each containing 4 threads.
        # This will launch a total of 2 * 4 = 8 threads.
        ctx.enqueue_function[my_gpu_kernel](grid_dim=2, block_dim=4)
        ctx.synchronize() # Wait for the GPU to finish.
    except:
        print("No compatible GPU found.")
```

**Pattern 2.2.2: Global Thread Indexing**

The most fundamental pattern in any non-trivial kernel is calculating a unique
global index for each thread. This index is then used to map each thread to a
specific element of the data being processed.

```mojo
from gpu.id import thread_idx, block_idx, block_dim
from DType import int32
from memory import DTypePointer

# A kernel that performs element-wise vector addition.
fn vector_add_kernel(
    c: DTypePointer[int32],
    a: DTypePointer[int32],
    b: DTypePointer[int32],
    size: Int
):
    # Pattern: Calculate the unique global index for this thread.
    let idx = block_idx.x * block_dim.x + thread_idx.x

    # Pattern: Boundary check guard.
    # This is essential because the grid size (total threads) may be larger
    # than the data size. This prevents out-of-bounds memory access.
    if idx < size:
        c.store(idx, a.load(idx) + b.load(idx))
```

### **2.3. Asynchronous Execution and Data Transfer Patterns** {#23-asynchronous-execution-and-data-transfer-patterns}

Operations between the CPU (host) and GPU (device) are inherently asynchronous.
When the host enqueues a command, such as a memory copy or a kernel launch, it
does not wait for the command to complete. It immediately proceeds to the next
line of code. This requires explicit synchronization to ensure results are not
accessed before they are ready.15

**Pattern 2.3.1: GPU Memory Management and Synchronization**

This pattern shows the complete workflow: allocating memory on both host and
device, copying data to the device, launching the kernel, copying results back
to the host, and synchronizing execution.

```mojo
from gpu.host import DeviceContext
from DType import float32

# Assume vector_add_kernel from the previous pattern is defined.

fn gpu_workflow_pattern():
    let vector_size = 1024
    try:
        var ctx = DeviceContext()

        # 1. Allocate memory on the host (CPU) and device (GPU).
        let lhs_host = ctx.enqueue_create_host_buffer[float32](vector_size)
        let rhs_host = ctx.enqueue_create_host_buffer[float32](vector_size)
        let res_host = ctx.enqueue_create_host_buffer[float32](vector_size)

        let lhs_device = ctx.enqueue_create_buffer[float32](vector_size)
        let rhs_device = ctx.enqueue_create_buffer[float32](vector_size)
        let res_device = ctx.enqueue_create_buffer[float32](vector_size)

        # Initialize host data...
        for i in range(vector_size):
            lhs_host[i] = float(i)
            rhs_host[i] = float(i)

        # 2. Copy input data from host to device (H2D).
        ctx.enqueue_copy(dst_buf=lhs_device, src_buf=lhs_host)
        ctx.enqueue_copy(dst_buf=rhs_device, src_buf=rhs_host)

        # 3. Launch the kernel on the device.
        let block_size = 256
        let grid_size = (vector_size + block_size - 1) // block_size
        ctx.enqueue_function[vector_add_kernel](
            res_device.unsafe_ptr(),
            lhs_device.unsafe_ptr(),
            rhs_device.unsafe_ptr(),
            vector_size,
            grid_dim=grid_size,
            block_dim=block_size
        )

        # 4. Copy results from device back to host (D2H).
        ctx.enqueue_copy(dst_buf=res_host, src_buf=res_device)

        # 5. Synchronize.
        # This is a critical step. It forces the CPU to wait until all
        # previously enqueued GPU commands (copies and kernel) have completed.
        ctx.synchronize()

        # Now it is safe to access the results on the host.
        print("GPU computation complete. First result:", res_host)

    except:
        print("GPU workflow failed or no GPU found.")
```

## **Section 3: Optimizing Memory Access with Tiling (Loop Blocking)** {#section-3-optimizing-memory-access-with-tiling-loop-blocking}

Tiling, also known as loop blocking, is a crucial memory optimization technique
that restructures loops to improve data locality and maximize the reuse of data
in the cache.25 The memory hierarchy in modern computers consists of small, fast
caches and large, slow main memory. Accessing data from the cache is orders of
magnitude faster than fetching it from main memory. Tiling works by partitioning
a large dataset into smaller "tiles" or "blocks" that are sized to fit within
the cache. By processing one tile completely before moving to the next, the
algorithm can reuse the data within that tile multiple times while it is still
resident in the fast cache, dramatically reducing the number of expensive main
memory accesses.25

### **3.1. The Canonical Tiling Pattern for CPU** {#31-the-canonical-tiling-pattern-for-cpu}

For CPU-bound computations involving multi-dimensional arrays, like matrix
multiplication, tiling can provide a significant speedup. The standard
implementation involves adding outer loops that iterate over the tiles and
modifying the inner loops to operate only within the boundaries of the current
tile.

**Pattern 3.1.1: Tiled Matrix Transposition**

This pattern demonstrates tiling for a memory-bound operation. In a naive
transpose, accessing a\[j\]\[i\] leads to strided memory access with poor
spatial locality. Tiling improves this by processing small square blocks,
keeping accesses within a more localized memory region.25

```mojo
from collections import List
from DType import float32

fn tiled_transpose_cpu(
    b: List[List[float32]],
    a: List[List[float32]],
    N: Int
):
    alias TILE_SIZE = 16

    # Outer loops iterate over the tiles.
    for ii in range(0, N, TILE_SIZE):
        for jj in range(0, N, TILE_SIZE):
            # Inner loops iterate within the tile.
            for i in range(ii, min(ii + TILE_SIZE, N)):
                for j in range(jj, min(jj + TILE_SIZE, N)):
                    b[i][j] = a[j][i]
```

### **3.2. Idiomatic Tiling in Mojo with Parameterized Functions** {#32-idiomatic-tiling-in-mojo-with-parameterized-functions}

While manual loop nesting works, Mojo's metaprogramming capabilities enable a
more abstract, reusable, and optimizable approach to tiling. This is often
achieved by using a higher-order function that encapsulates the tiling logic and
takes a parameterized function as an argument to perform the computation on each
tile.8

**Pattern 3.2.1: Abstracted Tiling with a Higher-Order Function**

This pattern separates the "what" (the tile computation) from the "how" (the
iteration over tiles).

```mojo
from DType import int32
from memory import DTypePointer

# A higher-order function that handles the tiling logic.
fn tile_2d[tiled_fn: fn[tile_x: Int, tile_y: Int](Int, Int) -> None,
          tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            # Call the provided kernel function for each tile.
            tiled_fn[tile_x, tile_y](x, y)

# A parameterized function that defines the work for one tile.
@parameter
fn process_tile[tile_x: Int, tile_y: Int](x_offset: Int, y_offset: Int):
    # This function would contain the actual computation for a tile
    # starting at (x_offset, y_offset) with dimensions (tile_x, tile_y).
    print("Processing tile at:", x_offset, y_offset, "of size:", tile_x, "x", tile_y)

fn main_abstract_tiling():
    alias TILE_DIM_X = 32
    alias TILE_DIM_Y = 16
    let total_width = 128
    let total_height = 64

    # The compiler specializes 'tile_2d' and 'process_tile' with the
    # constant tile dimensions, enabling aggressive optimization.
    tile_2d(total_width, total_height)
```

### **3.3. The GPU Shared Memory Tiling Pattern** {#33-the-gpu-shared-memory-tiling-pattern}

On GPUs, tiling is not just an optimization; it is the fundamental technique for
achieving high memory bandwidth. It is inextricably linked with the use of
**shared memory**, a small, programmable, on-chip cache with extremely low
latency that is shared among all threads in a block.23 The goal is to minimize
traffic to the large but slow global device memory (DRAM).

The core strategy involves a cooperative effort by the threads within a block.
They work together to load a tile of data from global memory into shared memory,
perform computations on that tile using the fast shared memory, and then write
the final results back to global memory. This pattern dramatically increases
arithmetic intensity (the ratio of math operations to memory operations) and is
essential for turning memory-bound problems into compute-bound ones.

**Pattern 3.3.1: The Load-Sync-Compute-Store Workflow**

This pattern illustrates the canonical workflow for shared memory tiling in a
GPU kernel. It requires explicit synchronization using barrier() to coordinate
the actions of threads within a block.

```mojo
from gpu.id import thread_idx, block_idx, block_dim
from gpu.memory import barrier
from gpu.types import Shared
from DType import float32
from memory import DTypePointer

@parameter
fn tiled_gpu_kernel(
    output_ptr: DTypePointer[float32],
    input_ptr: DTypePointer[float32],
    size: Int
):
    # 1. Allocate Shared Memory.
    # This memory is shared by all threads in the current block.
    # Its size is known at compile time.
    shared tile_data = Shared()

    # Calculate the global index for the current thread.
    let idx = block_idx.x * block_dim.x + thread_idx.x

    # 2. Cooperative Load from Global to Shared Memory.
    # Each thread in the block loads one element from the input in
    # global memory and places it into the shared memory tile.
    if idx < size:
        tile_data[thread_idx.x] = input_ptr.load(idx)

    # 3. Synchronize.
    # 'barrier()' acts as a rendezvous point. No thread can proceed past
    # this point until ALL threads in the block have reached it.
    # This ensures that 'tile_data' is fully populated before anyone reads it.
    barrier()

    # 4. Compute using fast Shared Memory.
    # For this example, we'll do a simple operation, like reversing
    # the elements within the tile. In a real kernel (like matmul),
    # this would be the main computational loop.
    var value: float32 = 0
    if idx < size:
        let reversed_local_idx = block_dim.x - 1 - thread_idx.x
        value = tile_data[reversed_local_idx] * 2.0

    # Note: Another barrier() might be needed here if the computation involved
    # threads reading results written by other threads to shared memory.

    # 5. Store the result from a register back to Global Memory.
    if idx < size:
        output_ptr.store(idx, value)
```

This pattern reveals that advanced kernel optimization is an exercise in
orchestrating the hardware's hierarchies of parallelism (threads, blocks) and
memory (registers, shared memory, global memory). A naive parallel algorithm
where each thread independently accesses global memory will quickly become
memory-bound, as thousands of threads contend for the limited bandwidth of the
memory bus, leaving the powerful computational units idle.12 The tiled approach
transforms the problem. A block of threads is assigned a tile of work. They
perform a single, coordinated, and often coalesced read from slow global memory
into fast shared memory. After synchronization, the computation proceeds at the
much higher speed of the on-chip memory. This strategy is the key to unlocking
the full computational power of the GPU.

## **Section 4: Synthesis: A Fully Optimized Matrix Multiplication Kernel** {#section-4-synthesis-a-fully-optimized-matrix-multiplication-kernel}

Matrix multiplication (matmul) is the quintessential benchmark for
high-performance computing. Its computational intensity and regular memory
access patterns make it an ideal candidate for optimization, and it serves as
the core operation in countless AI and scientific workloads.27 This section
synthesizes the techniques of SIMD, parallelization, and tiling to incrementally
build a fully optimized

matmul kernel in Mojo.

### **Pattern 4.1: The Naive Baseline Implementation**

We begin with a simple, Pythonic implementation using def and nested loops. This
version is easy to read but will have poor performance due to dynamic typing and
sequential execution. It serves as our functional reference and performance
baseline.

```mojo
from collections import List
from DType import float32

# Naive, sequential matmul using Pythonic 'def' and lists.
def matmul_naive(
    C: List[List[float32]],
    A: List[List[float32]],
    B: List[List[float32]],
    M: Int, N: Int, K: Int
):
    for i in range(M):
        for j in range(N):
            var accumulator: float32 = 0.0
            for k in range(K):
                accumulator += A[i][k] * B[k][j]
            C[i][j] = accumulator
```

### **Pattern 4.2: Typed and Vectorized fn Kernel for CPU**

The first step in optimization is to move from dynamic def to a statically-typed
fn and apply SIMD vectorization to the innermost loop. This exploits
instruction-level parallelism within a single CPU core. We vectorize along the j
dimension (columns of C, rows of B) to ensure contiguous memory access for B if
it is stored in row-major order, which is crucial for efficient simd\_load
operations.13

```mojo
from DType import float32
from builtin import SIMD, simdwidthof
from memory import DTypePointer

@parameter
fn matmul_vectorized(
    C: DTypePointer[float32],
    A: DTypePointer[float32],
    B: DTypePointer[float32],
    M: Int, N: Int, K: Int
):
    alias nelts = simdwidthof[float32]()  # Native SIMD width
    for i in range(M):
        for j in range(0, N, nelts):
            # Accumulator is a SIMD vector, initialized to zeros.
            var c_vec = SIMD[float32, nelts](0)
            for k in range(K):
                # Load a scalar from A and splat it into a vector.
                let a_val = A.load(i * K + k)
                let a_vec = SIMD[float32, nelts](a_val)
                # Load a vector from B.
                let b_vec = B.simd_load[nelts](k * N + j)
                # Fused-Multiply-Add (FMA) is often implicitly generated here.
                c_vec += a_vec * b_vec
            # Store the final vector result to C.
            C.simd_store(i * N + j, c_vec)
```

### **Pattern 4.3: Parallelized CPU Kernel**

To engage multiple CPU cores, we wrap the vectorized kernel with parallelize.
The outer loop over the rows of the result matrix C is a natural candidate for
parallelization, as the calculation for each row is independent of the others.4

```mojo
from algorithm import parallelize

fn matmul_vectorized_parallel(
    C: DTypePointer[float32],
    A: DTypePointer[float32],
    B: DTypePointer[float32],
    M: Int, N: Int, K: Int
):
    alias nelts = simdwidthof[float32]()

    # Define the work for a single row 'i'.
    @parameter
    fn calc_row(i: Int):
        for j in range(0, N, nelts):
            var c_vec = SIMD[float32, nelts](0)
            for k in range(K):
                let a_vec = SIMD[float32, nelts](A.load(i * K + k))
                let b_vec = B.simd_load[nelts](k * N + j)
                c_vec += a_vec * b_vec
            C.simd_store(i * N + j, c_vec)

    # Distribute the row calculations across all CPU cores.
    parallelize[calc_row](M)
```

### **Pattern 4.4: GPU Kernel with Shared Memory Tiling**

This pattern represents the leap to maximum performance on GPU hardware. It
combines the grid/block/thread parallelism model with shared memory tiling to
maximize arithmetic intensity. Each thread block is responsible for computing
one tile of the output matrix C. To do this, it iteratively loads tiles from A
and B into shared memory, performs a matrix multiplication on the tiles in fast
memory, and accumulates the result in registers.8

```mojo
from gpu.id import thread_idx, block_idx, block_dim
from gpu.memory import barrier
from gpu.types import Shared
from DType import float32
from memory import DTypePointer

@parameter
fn matmul_gpu_tiled(
    C: DTypePointer[float32],
    A: DTypePointer[float32],
    B: DTypePointer[float32],
    M: Int, N: Int, K: Int
):
    # 1. Shared memory allocation for tiles from A and B.
    shared a_tile = Shared()
    shared b_tile = Shared()

    # Thread indices within the block.
    let tx = thread_idx.x
    let ty = thread_idx.y

    # Global row and column for this thread's primary output element.
    let row = block_idx.y * TILE_DIM + ty
    let col = block_idx.x * TILE_DIM + tx

    # Accumulator for this thread, stored in a register.
    var accumulator: float32 = 0.0

    # 2. Loop over tiles along the K dimension.
    for tile_k in range(0, K, TILE_DIM):
        # 3. Cooperative Load from Global to Shared Memory.
        # Each thread loads one element of A's tile and one of B's tile.
        let a_idx = row * K + (tile_k + tx)
        let b_idx = (tile_k + ty) * N + col
        if row < M and (tile_k + tx) < K:
            a_tile = A.load(a_idx)
        else:
            a_tile = 0.0
        if col < N and (tile_k + ty) < K:
            b_tile = B.load(b_idx)
        else:
            b_tile = 0.0

        # 4. Synchronize to ensure tiles are fully loaded.
        barrier()

        # 5. Compute: Multiply the tiles from shared memory.
        for k_inner in range(TILE_DIM):
            accumulator += a_tile * b_tile

        # 6. Synchronize before loading the next tile.
        barrier()

    # 7. Store the final result from the register to global memory.
    if row < M and col < N:
        C.store(row * N + col, accumulator)
```

### **Pattern 4.5: The Autotuned, Performance-Portable Kernel**

The optimal TILE\_DIM in the previous pattern is highly dependent on the
specific GPU's architecture (shared memory size, number of registers per thread,
warp size, etc.). Hardcoding this value leads to code that is only optimal for
one specific machine. The final and most advanced pattern uses Mojo's autotune
feature to make the kernel performance-portable. The autotune utility compiles
and benchmarks multiple versions of the kernel with different compile-time
parameters and automatically selects the fastest one for the target hardware.1

```mojo
from autotune import autotune, search
from time import now

# The matmul_gpu_tiled function from Pattern 4.4 is used here.

# The autotune block defines the search space and evaluation logic.
@adaptive
fn matmul_autotuned(
    C: DTypePointer[float32],
    A: DTypePointer[float32],
    B: DTypePointer[float32],
    M: Int, N: Int, K: Int
):
    # 1. Define the search space for the TILE_DIM parameter.
    # The compiler will generate a version of the kernel for each value.
    alias TILE_DIM = autotune(8, 16, 32)

    # 2. Define the evaluator function to benchmark each version.
    fn evaluator(
        funcs: DTypePointer[fn() -> None],
        num_candidates: Int
    ) -> Int:
        var best_time: float64 = 1e9
        var best_idx = 0
        for i in range(num_candidates):
            # Warm-up run to avoid JIT overhead in measurement.
            funcs.load(i)()
            # Benchmark the candidate function.
            let start = now()
            funcs.load(i)()
            let end = now()
            let time = (end - start).to_float64()

            if time < best_time:
                best_time = time
                best_idx = i
        return best_idx

    # 3. Use 'search' to find the best function and execute it.
    # This captures the runtime arguments for the kernel.
    alias best_matmul_fn = search[
        matmul_gpu_tiled[TILE_DIM](C, A, B, M, N, K)
    ](evaluator)
    best_matmul_fn()
```

This final pattern represents the pinnacle of Mojo's optimization philosophy. It
combines explicit, low-level control over GPU hardware (via shared memory
tiling) with high-level, compile-time automation (autotune). The result is a
single source file that can produce a highly specialized and performant kernel
that is automatically adapted to the specific hardware it is compiled on,
achieving the twin goals of extreme performance and portability.

### **Conclusion** {#conclusion}

The advanced optimization techniques available in Mojo—SIMD, parallelization,
and tiling—are not merely features but form a comprehensive toolkit for systems
programming. They provide the developer with explicit and granular control over
the underlying hardware, enabling the systematic deconstruction and optimization
of performance-critical kernels. The journey from a naive, sequential
implementation to a fully optimized, autotuned GPU kernel for matrix
multiplication illustrates a clear, repeatable methodology:

1. **Establish Correctness and Baseline:** Begin with a simple, readable
   implementation to ensure functional correctness.
2. **Apply Instruction-Level Parallelism:** Use Mojo's first-class SIMD types to
   vectorize the innermost computational loops, maximizing the throughput of a
   single core.
3. **Introduce Task-Level Parallelism:** Employ parallelize for multi-core CPU
   execution or the GPU's grid-block-thread model to distribute independent work
   across multiple processing units.
4. **Optimize Memory Access:** Implement tiling to improve cache locality on
   CPUs or, critically, to leverage fast shared memory on GPUs. This step is
   essential for preventing memory bottlenecks and allowing the parallel compute
   units to operate at full capacity.
5. **Parameterize and Autotune:** Abstract hardware-specific details like tile
   sizes into compile-time parameters and use autotune to programmatically
   discover the optimal configuration for any given target machine.

This layered approach, enabled by Mojo's unique fusion of Pythonic syntax with
powerful compile-time metaprogramming, allows developers to bridge the
historical gap between productivity and performance. By providing direct control
over vectorization, parallelism, and memory layout, Mojo empowers programmers to
write code that is not only highly performant but also abstract, reusable, and
portable across the diverse landscape of modern AI hardware.

#### **Works cited** {#works-cited}

1. Meet Mojo: The Language That Could Replace Python, C++, and CUDA -
   HackerNoon, accessed June 28, 2025,
   [https://hackernoon.com/meet-mojo-the-language-that-could-replace-python-c-and-cuda](https://hackernoon.com/meet-mojo-the-language-that-could-replace-python-c-and-cuda)
2. Introduction to Mojo Programming Language - W3Schools, accessed June 28,
   2025,
   [https://www.w3schools.in/mojo/introduction](https://www.w3schools.in/mojo/introduction)
3. Mojo v/s Python In Performance-Critical AI Applications | Blog - Cubet,
   accessed June 28, 2025,
   [https://cubettech.com/resources/blog/mojo-v-s-python-in-performance-critical-ai-applications/](https://cubettech.com/resources/blog/mojo-v-s-python-in-performance-critical-ai-applications/)
4. Unveiling the Mojo Launch with Jeremy Howard - Toolify.ai, accessed June 28,
   2025,
   [https://www.toolify.ai/ai-news/unveiling-the-mojo-launch-with-jeremy-howard-17723](https://www.toolify.ai/ai-news/unveiling-the-mojo-launch-with-jeremy-howard-17723)
5. Mojo, The Next-Gen Programming Language | by guna S D | Jun, 2025 | Medium,
   accessed June 28, 2025,
   [https://medium.com/@sdgunaa/mojo-the-next-gen-programming-language-ebbde84705c9](https://medium.com/@sdgunaa/mojo-the-next-gen-programming-language-ebbde84705c9)
6. Mojo - A New Programming Language for AI - Refine dev, accessed June 28,
   2025,
   [https://refine.dev/blog/mojo-programming-language/](https://refine.dev/blog/mojo-programming-language/)
7. 2024 LLVM Dev Mtg - Simplifying GPU Programming with Parametric Tile-Level
   Tensors In Mojo - YouTube, accessed June 28, 2025,
   [https://www.youtube.com/watch?v=sOZWhPVvRdw](https://www.youtube.com/watch?v=sOZWhPVvRdw)
8. dsharlet/mojo_comments: Experiments with mojo - GitHub, accessed June 28,
   2025,
   [https://github.com/dsharlet/mojo_comments](https://github.com/dsharlet/mojo_comments)
9. Mojo : Powerful CPU+GPU Programming - Modular, accessed June 28, 2025,
   [https://www.modular.com/mojo](https://www.modular.com/mojo)
10. [Feature Request] Enhanced Compile-Time and Runtime Integration for Function
    Optimization in Mojo · Issue #1731 · modular/max - GitHub, accessed June 28,
    2025,
    [https://github.com/modularml/mojo/issues/1731](https://github.com/modularml/mojo/issues/1731)
11. Highly efficient matrix transpose in Mojo | simons blog, accessed June 28,
    2025,
    [https://veitner.bearblog.dev/highly-efficient-matrix-transpose-in-mojo/](https://veitner.bearblog.dev/highly-efficient-matrix-transpose-in-mojo/)
12. Modular Tech Talk: Kernel Programming and Mojo - YouTube, accessed June 28,
    2025,
    [https://www.youtube.com/watch?v=Invd_dxC2RU](https://www.youtube.com/watch?v=Invd_dxC2RU)
13. Optimizing Matrix Multiplication for ML with Mojo | by Benny - Medium,
    accessed June 28, 2025,
    [https://medium.com/@bennynottonson/optimizing-matrix-multiplication-for-ml-with-mojo-bfc428112360](https://medium.com/@bennynottonson/optimizing-matrix-multiplication-for-ml-with-mojo-bfc428112360)
14. Mojo Tutorial | PDF | Computers | Technology & Engineering - Scribd,
    accessed June 28, 2025,
    [https://www.scribd.com/document/649592378/Mojo-Tutorial](https://www.scribd.com/document/649592378/Mojo-Tutorial)
15. Basics of GPU programming with Mojo | Modular, accessed June 28, 2025,
    [https://docs.modular.com/mojo/manual/gpu/basics/](https://docs.modular.com/mojo/manual/gpu/basics/)
16. Get started with GPU programming - Modular docs, accessed June 28, 2025,
    [https://docs.modular.com/mojo/manual/gpu/intro-tutorial/](https://docs.modular.com/mojo/manual/gpu/intro-tutorial/)
17. mojo-learning/tutorials/simd.md at main - GitHub, accessed June 28, 2025,
    [https://github.com/rd4com/mojo-learning/blob/main/tutorials/simd.md](https://github.com/rd4com/mojo-learning/blob/main/tutorials/simd.md)
18. Manual vectorization techniques - Efficient Systems Programming with Mojo:
    Python to Low-Level | StudyRaid, accessed June 28, 2025,
    [https://app.studyraid.com/en/read/12633/409992/manual-vectorization-techniques](https://app.studyraid.com/en/read/12633/409992/manual-vectorization-techniques)
19. simd - Modular docs, accessed June 28, 2025,
    [https://docs.modular.com/mojo/stdlib/builtin/simd/](https://docs.modular.com/mojo/stdlib/builtin/simd/)
20. Parallelism concepts in Mojo - Efficient Systems Programming with Mojo:
    Python to Low-Level | StudyRaid, accessed June 28, 2025,
    [https://app.studyraid.com/en/read/12633/409985/parallelism-concepts-in-mojo](https://app.studyraid.com/en/read/12633/409985/parallelism-concepts-in-mojo)
21. parallelize - Modular docs, accessed June 28, 2025,
    [https://docs.modular.com/mojo/stdlib/algorithm/functional/parallelize/](https://docs.modular.com/mojo/stdlib/algorithm/functional/parallelize/)
22. vectorize vs parallelize in Mojo - Stack Overflow, accessed June 28, 2025,
    [https://stackoverflow.com/questions/76562547/vectorize-vs-parallelize-in-mojo](https://stackoverflow.com/questions/76562547/vectorize-vs-parallelize-in-mojo)
23. GPU programming basics - Efficient Systems Programming with Mojo: Python to
    Low-Level, accessed June 28, 2025,
    [https://app.studyraid.com/en/read/12633/409993/gpu-programming-basics](https://app.studyraid.com/en/read/12633/409993/gpu-programming-basics)
24. Optimize custom ops for GPUs with Mojo - Modular docs, accessed June 28,
    2025,
    [https://docs.modular.com/max/tutorials/custom-ops-matmul/](https://docs.modular.com/max/tutorials/custom-ops-matmul/)
25. Loop tiling | Code Guidelines for Correctness, Modernization, Security,
    Portability, and Optimization - Open Catalog, accessed June 28, 2025,
    [https://open-catalog.codee.com/Glossary/Loop-tiling/](https://open-catalog.codee.com/Glossary/Loop-tiling/)
26. Loop Tiling or Blocking - Aussie AI, accessed June 28, 2025,
    [https://www.aussieai.com/book/ch15-loop-tiling-blocking](https://www.aussieai.com/book/ch15-loop-tiling-blocking)
27. Exploring Mojo : The Emerging High-Performance Language With Impressive
    Speeds, But Not Without Competition | by Zaza Zakaria | Better Programming
    - Medium, accessed June 28, 2025,
      [https://medium.com/better-programming/exploring-mojo-the-emerging-high-performance-language-with-impressive-speeds-but-not-without-acdbbbed09f2](https://medium.com/better-programming/exploring-mojo-the-emerging-high-performance-language-with-impressive-speeds-but-not-without-acdbbbed09f2)
28. mojo-learning/tutorials/autotuned-parametrized-tests.md at main - GitHub,
    accessed June 28, 2025,
    [https://github.com/rd4com/mojo-learning/blob/main/tutorials/autotuned-parametrized-tests.md](https://github.com/rd4com/mojo-learning/blob/main/tutorials/autotuned-parametrized-tests.md)
29. A brief guide to the Mojo n-body example - Modular, accessed June 28, 2025,
    [https://www.modular.com/blog/a-brief-guide-to-the-mojo-n-body-example](https://www.modular.com/blog/a-brief-guide-to-the-mojo-n-body-example)
30. Instruction-level parallelism - Wikipedia, accessed June 28, 2025,
    [https://en.wikipedia.org/wiki/Instruction-level_parallelism](https://en.wikipedia.org/wiki/Instruction-level_parallelism)
31. What is a SIMD operation? #mojo #python #java #programming #cpu #gpu #llvm
    #mlir #github #javascript - YouTube, accessed June 28, 2025,
    [https://www.youtube.com/shorts/CO8CwN1hJb4](https://www.youtube.com/shorts/CO8CwN1hJb4)
32. Counting chars with SIMD in Mojo - Maxim Zaks - Medium, accessed June 28,
    2025,
    [https://mzaks.medium.com/counting-chars-with-simd-in-mojo-140ee730bd4d](https://mzaks.medium.com/counting-chars-with-simd-in-mojo-140ee730bd4d)
33. Faster prefix sum computation with SIMD and Mojo | by Maxim Zaks - Medium,
    accessed June 28, 2025,
    [https://mzaks.medium.com/faster-prefix-sum-computation-with-simd-and-mojo-39bdc25e49b3](https://mzaks.medium.com/faster-prefix-sum-computation-with-simd-and-mojo-39bdc25e49b3)
34. [Docs] Any multithreading or concurrency explanation · Issue #1660 - GitHub,
    accessed June 28, 2025,
    [https://github.com/modularml/mojo/issues/1660](https://github.com/modularml/mojo/issues/1660)
35. Introduction - Mojo GPU Puzzles - Code with Modular, accessed June 28, 2025,
    [https://builds.modular.com/puzzles/introduction.html](https://builds.modular.com/puzzles/introduction.html)
36. How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog
    - siboehm, accessed June 28, 2025,
      [https://siboehm.com/articles/22/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM)
37. How to Be Confident in Your Performance Benchmarking - Modular, accessed
    June 28, 2025,
    [https://www.modular.com/blog/how-to-be-confident-in-your-performance-benchmarking](https://www.modular.com/blog/how-to-be-confident-in-your-performance-benchmarking)
38. 2023 LLVM Dev Mtg - Mojo : A system programming language for heterogenous
    computing - YouTube, accessed June 28, 2025,
    [https://www.youtube.com/watch?v=SEwTjZvy8vw](https://www.youtube.com/watch?v=SEwTjZvy8vw)
