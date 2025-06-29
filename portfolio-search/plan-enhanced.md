### **Hackathon Plan 2.0: Real-Time Semantic Code Search with Mojo**

#### **🎯 Core Vision & Pitch**

For any developer with a growing portfolio of projects, finding past work is a
constant struggle. Standard search tools rely on keywords, failing to capture
the _intent_ or _concept_ behind the code. This project tackles that head-on. We
are building a semantic search engine that understands what your code _does_,
not just what it says.

We will achieve sub-50ms, as-you-type search speeds by identifying the core
computational bottlenecks of a state-of-the-art AI embedding model and replacing
them with hyper-optimized Mojo kernels. This isn't just a faster grep; it's a
"project brain"—an instantly accessible, collective intelligence of your entire
coding history that can find concepts, patterns, and specific implementations in
the time it takes to type a query.

### **System Architecture: A Pragmatic Approach**

The key to a successful hackathon is focusing innovation where it delivers the
most impact. Instead of designing a new attention model from scratch (a research
project in itself), we will port a proven, pre-trained code embedding model to
run with Mojo-accelerated inference. This gives us state-of-the-art semantic
understanding combined with unparalleled performance.

Data Flow (Indexing):\
Codebase ➞ Tree-sitter ➞ Pre-trained Tokenizer ➞ Mojo-Accelerated Model ➞ Vector
Store\
(Python Script) (AST Parsing) (Hugging Face) (Python ➞ Mojo Bridge) (In-memory
NumPy/Mojo Tensor)\
Search Flow (Querying):\
User Query ➞ Tokenizer ➞ Mojo Model ➞ Query Vector ➞ Mojo SIMD Search ➞ Ranked
Results

### **Phase 1: The Core End-to-End Pipeline (Hours 1-6)**

The primary goal of this phase is to establish a functional, end-to-end
pipeline. We need to prove that a single code snippet can be tokenized, embedded
using a Python-Mojo hybrid model, and searched against a tiny corpus. Speed is
not the concern here; correctness and integration are everything.

**1.1. Environment & Model Setup**

- **Action:** Set up a dedicated Python virtual environment. Install core
  dependencies: pip install mojo torch transformers "tree-sitter\~=0.20" numpy
  flask. You will also need to install language-specific grammars for
  Tree-sitter (e.g., git clone
  <https://github.com/tree-sitter/tree-sitter-python>).
- **Model Choice:** Select a compact, high-performance pre-trained model. For a
  hackathon, starting small is key.
  - **Recommended:** **GTE-small (General Text Embeddings)**. It offers an
    excellent balance of embedding quality and computational efficiency
    (384-dim), making it ideal for rapid prototyping and demonstrating speed
    gains.
  - **Alternatives:** MiniLM-L6-v2 is a viable classic. Code-specific models
    like CodeBERT are powerful but their larger size might complicate
    optimization efforts within the hackathon timeline.
- **Goal:** Write a self-contained Python script embed\_test.py. This script
  should load the GTE-small model from Hugging Face, take a hardcoded string of
  Python code, and successfully print a 384-dimension vector (NumPy array) to
  the console. This validates that the foundational embedding logic is working.

**1.2. Kernel Scaffolding: From PyTorch to Mojo**

- **Identify the Bottleneck:** Use torch.profiler to analyze the forward pass of
  your chosen model. The profiler will clearly show that the vast majority of
  computation time is spent in matrix multiplications (torch.matmul within
  nn.Linear layers) and the associated operations in the attention mechanism.
  These are our primary targets for Mojo optimization.
- **Mojo MatMul Kernel (Initial Version):**
  - **Input:** Two Tensor\[T\] objects representing the input matrices.
  - **Output:** The resulting Tensor\[T\].
  - **Initial Goal:** Focus entirely on correctness. Implement a naive,
    nested-loop matrix multiplication in a .mojo file. This slow but simple
    version will confirm that you can pass data to Mojo and compute a result.
- **Python-Mojo Bridge:**
  - In a Python script, import your Mojo module. The key task is to convert
    PyTorch tensors into a format Mojo can understand (e.g., a pointer to the
    underlying data buffer). Write a Python function that takes two NumPy
    arrays, passes them to your Mojo MatMul kernel, receives the result, and
    converts it back into a NumPy array. Assert that the output of your Mojo
    kernel matches the output of numpy.matmul for the same inputs. This proves
    the bridge is stable.

### **Phase 2: Optimization & Scaling (Hours 7-16)**

With a working-but-slow pipeline, this phase is about two things: making it fast
with Mojo and feeding it a meaningful amount of data.

**2.1. Kernel Optimization: The Mojo Magic**

- **Action:** Iteratively enhance your MatMul kernel. This is the core of the
  hackathon's technical innovation.
  - **SIMD (Single Instruction, Multiple Data):** Refactor your inner loop to
    use Mojo's SIMD\[DType, Width\] type. This will allow you to load multiple
    floating-point values from both matrices and perform multiple
    multiplications and additions in a single CPU instruction, providing a
    significant, immediate speedup.
  - **Parallelization:** Wrap your outer loop with the parallelize function from
    Mojo's standard library. This will automatically distribute the work of
    calculating rows of the output matrix across all available CPU cores.
  - **Tiling:** For larger matrices, implement tiling (or loop blocking). This
    involves breaking the matrices into smaller sub-matrices that fit neatly
    into the CPU's cache. By processing one tile at a time, you drastically
    reduce cache misses, which is often a major performance bottleneck.
- **Target:** Create a benchmark.py script. It should compare the execution time
  of your fully optimized Mojo kernel against torch.matmul on matrices of
  various sizes relevant to your model. Your goal is to generate a graph showing
  a 5x-10x speed improvement. **This benchmark is your core demo point.**

**2.2. The "Vectorize-All-The-Things" Pipeline**

- **Code Parsing with Tree-sitter:** Write a robust Python script that
  recursively walks your project directories. For each source file, it should
  use the appropriate Tree-sitter grammar to parse the file into an AST.
  Traverse the AST to intelligently extract logical code chunks: functions,
  classes, and methods, including their docstrings. This is superior to naive
  line-based chunking as it preserves semantic context.
- **Batch Embedding & Storage:** Process these extracted chunks in batches for
  efficiency. Feed each batch to your Mojo-accelerated embedding model. Store
  the results in two files:
  1. A binary file (corpus\_embeddings.npy) containing the raw NumPy array of
     all vectors.
  2. A JSON file (corpus\_metadata.json) containing a list of objects, where
     each object holds the file\_path, project\_name, function\_name, start/end
     line numbers, and the raw code for the corresponding vector. The index of
     the list maps directly to the row number in the NumPy array.

**2.3. The SIMD Search Kernel**

- **Action:** Create a new, highly specialized Mojo function for the search
  operation itself. The task is to compute the cosine similarity between a
  single query vector and the entire corpus matrix of N vectors.
- **Optimization:** This is a perfect use case for SIMD. The kernel should load
  the query vector into SIMD registers. Then, it can loop through the corpus
  matrix, loading chunks of each document vector into other registers,
  performing the dot product, and accumulating the result in parallel.
- **Target:** The function should return the top K indices and their similarity
  scores. Benchmark this function: it should be able to scan over 100,000
  vectors in well under a millisecond.

### **Phase 3: Interface & Demo Prep (Hours 17-20)**

This is about presentation. We need a simple, intuitive interface and a
compelling story that highlights the "magic" of the technology.

**3.1. API & Simple UI**

- **Action:** Wrap your search logic in a simple Flask or FastAPI endpoint.
  - A POST /search endpoint accepts a JSON body: {"query": "your code query",
    "top\_k": 10}.
  - It should call your embedding model to vectorize the query, use the SIMD
    Search Kernel to find the best matches, retrieve the corresponding metadata
    from the JSON file, and return a ranked list of results.
- **Web Interface:** Build a minimal, clean frontend with HTML and vanilla
  JavaScript (or a simple library like Preact).
  - A single, prominent search box.
  - Use a debounce function on the onkeyup event to fire API requests \~150ms
    after the user stops typing. This feels responsive without overwhelming the
    backend.
  - The results area should render each match with syntax highlighting (using a
    library like highlight.js) and display the project, file path, and
    similarity score. Make the file path a clickable link to the file on GitHub.

**3.2. Demo Storytelling**

- **Prepare your scenarios:** Use your excellent existing scenarios (API
  patterns, DB connections, auth middleware). For each one, have a specific,
  non-trivial query ready.
- **Create the "Bake-Off":** This is your most powerful narrative tool.
  - Record a short screen capture of yourself trying to find an API pattern
    using your standard VSCode search (Ctrl+Shift+F). Show the fumbling, the
    irrelevant keyword matches, the frustration.
  - Immediately switch to a live demo of your application. Type a conceptual
    query like "http client with exponential backoff" and show the correct,
    semantically relevant results appearing _instantly_.
  - Conclude with the benchmark graph: **PyTorch MatMul vs. Your Optimized Mojo
    MatMul**. This provides the "how" behind the magic. Explain that this kernel
    is the engine that makes the instantaneous experience possible.

### **🌟 Post-Hackathon / Stretch Goals**

- **Advanced Ranker:** Implement a post-search re-ranking step. The formula
  could be final\_score \= similarity\_score \* (1 \+ recency\_boost) \* (1 \+
  project\_relevance\_boost). Recency can be sourced from Git metadata, and
  project relevance can be determined if the result is from the user's currently
  active project.
- **VSCode Extension:** This is the ideal form factor for this tool. Use the
  VSCode API to create a custom view with a webview that hosts your search UI,
  making it a seamless part of the development workflow.
- **Hybrid Search:** Augment the vector search with a traditional keyword-based
  index (e.g., Tantivy). This handles cases where a user needs to find an exact,
  but rare, variable name or string literal that semantic search might overlook.
- **Publish Kernels as a Library:** Clean up, document, and publish your
  optimized MatMul and SIMD-Search kernels. Accompany this with a blog post
  detailing the optimization process and benchmarks. This provides immense value
  back to the Modular community and establishes you as an expert.
