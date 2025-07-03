# **Gold-Standard Code Patterns for High-Performance Semantic Search on the Modular Platform**

## **Section 1: Optimal Embedding Generation with MAX Engine**

The foundation of any high-performance semantic search system is the rapid and
efficient generation of vector embeddings from text. This process, which
converts unstructured text into meaningful numerical representations, is the
most computationally intensive part of the pipeline. The Modular MAX Engine is
specifically designed to accelerate this step. This section details the
recommended model architecture, verified code patterns for inference, and
performance tuning parameters to achieve maximum throughput.

### **1.1 Recommended Embedding Model Architecture: BAAI BGE-BASE-EN-V1.5 (ONNX)**

The selection of an appropriate embedding model is critical for both search
relevance and system performance. Based on official documentation and benchmarks
from Modular, the recommended model is BAAI/bge-base-en-v1.5.1 This model is a
leading performer on the MTEB (Massive Text Embedding Benchmark) leaderboard,
offering an excellent balance of high accuracy, a standard 768-dimensional
embedding output, and a manageable disk footprint of 416MB.1

For integration with the Modular platform, the definitive format for this model
is ONNX (Open Neural Network Exchange). The MAX Engine is architected with an
"Importer" component that takes serialized models from standard frameworks like
TensorFlow, PyTorch, and ONNX and converts them into an internal graph
representation that can be aggressively optimized. The official, end-to-end
semantic search example from Modular demonstrates loading the bge-base-en-v1.5
model directly from its .onnx file.1 This establishes ONNX as the most stable,
verified, and high-performance bridge between the general machine learning
ecosystem, where models are trained and shared, and the specialized MAX
execution environment. For building the fastest possible semantic search engine,
standardizing on the ONNX format is the most robust and documented path,
providing a clear and unambiguous directive for implementation.

### **1.2 Code Pattern: High-Throughput Model Loading and Inference**

The core of high-performance inference on the Modular platform is the max.engine
Python module. The primary pattern for its use involves two key classes:
InferenceSession, which manages the runtime environment and hardware resources,
and Model, which represents the loaded and compiled model graph ready for
execution.

The verified code pattern for embedding generation is as follows:

1. Instantiate an engine.InferenceSession() to prepare the execution
   environment.
2. Load the ONNX model into the session using
   session.load("path/to/model.onnx"). This returns a Model object.
3. (Optional but recommended) Inspect the model's input\_metadata and
   output\_metadata properties to programmatically determine the required tensor
   names, shapes, and data types (dtype).
4. To maximize throughput, batch the input data. The official example
   effectively uses torch.utils.data.DataLoader to group tokenized sentences
   into batches, which is a critical optimization for amortizing inference
   overhead.1
5. Execute inference by calling maxmodel.execute() within a loop over the data
   loader. Inputs must be passed as keyword arguments where the keys match the
   model's input tensor names (e.g., input\_ids, attention\_mask).
6. Extract the desired embedding from the output tensor. For transformer-based
   models like BGE, this is typically the embedding of the \`\` token, which
   corresponds to the first token in the sequence (index 0\) of the
   last\_hidden\_state output.1

### **1.3 Performance Analysis: Verifiable Speed Advantage of MAX Engine**

The primary justification for using the Modular platform is its verifiable
performance advantage in the inference step. Quantitative benchmarks published
by Modular, conducted on an AWS c5.12xlarge CPU instance, demonstrate that the
MAX Engine significantly outperforms both native PyTorch and the standard ONNX
Runtime, especially as batch sizes increase.1 This acceleration is the main
driver of overall application speed.

| Batch Size | MAX Engine Latency (ms) | PyTorch Latency (ms) | ONNX Runtime Latency (ms) | MAX Speedup vs. PyTorch | MAX Speedup vs. ONNX Runtime |
| :--------- | :---------------------- | :------------------- | :------------------------ | :---------------------- | :--------------------------- |
| 1          | 24.3                    | 38.2                 | 68.3                      | 1.6x                    | 2.8x                         |
| 8          | 101.4                   | 151.7                | 200.7                     | 1.5x                    | 2.0x                         |
| 32         | 344.9                   | 536.2                | 623.6                     | 1.6x                    | 1.8x                         |
| 128        | 1317.0                  | 2605.0               | 2397.0                    | 2.0x                    | 1.8x                         |
| 512        | 5250.0                  | 10420.0              | 9588.0                    | 2.0x                    | 1.8x                         |

Table 1.1: MAX Engine Inference Performance Comparison (CPU). Data sourced from
official Modular benchmarks.1

### **1.4 Tuning Parameters for Embedding Generation**

To extract maximum performance, the MAX Engine offers several tuning parameters.
While the basic code pattern provides substantial speedups, these configurations
allow for fine-tuning based on specific hardware and workload characteristics.

| Parameter                | API/CLI          | Description                                                                                                         | Recommended Strategy for Semantic Search                                                                                                                 |
| :----------------------- | :--------------- | :------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| devices                  | InferenceSession | Specifies the hardware devices for inference (e.g., \['cpu'\], \['gpu:0'\]).                                        | Set explicitly to target available hardware (e.g., \['gpu:0'\] if a compatible GPU is present).                                                          |
| num\_threads             | InferenceSession | Sets the number of CPU threads for the session. Defaults to the number of physical cores.                           | For CPU-only inference, leave as default or tune based on workload to avoid oversubscription.                                                            |
| \--max-batch-size        | max CLI          | The maximum number of sequences processed in a single batch.                                                        | Tune to the largest value that fits in memory to maximize throughput. Start with values like 128 or 256\.                                                |
| \--quantization-encoding | max CLI          | Specifies the weight quantization type (e.g., q4\_k, gptq) to reduce model size and potentially speed up inference. | For GGUF models, use a supported quantization like q4\_k for a balance of speed and accuracy. For ONNX, quantization is typically done prior to loading. |
| \--max-length            | max CLI          | Maximum input sequence length for the model.                                                                        | Set to the model's maximum supported length (e.g., 512 for BGE) to avoid truncation of long documents.                                                   |

_Table 1.2: MAX Engine Performance Tuning Parameters. Parameters sourced from
MAX Engine and CLI documentation._

## **Section 2: High-Speed Vector Indexing and Search**

Once embeddings are generated, they must be stored in a specialized vector
database that can perform fast similarity searches. This section outlines the
optimal strategy for integrating a vector store and the recommended algorithm
and tuning parameters for achieving the lowest possible query latency.

### **2.1 Vector Store Integration Strategy: Python Interoperability**

The current gold-standard architecture for building applications on the Modular
platform is a hybrid "Python-First, Mojo-Accelerated" paradigm. While Mojo is a
powerful, high-performance language, its native ecosystem for specialized
libraries like vector databases is still developing. Conversely, Python has a
mature and extensive ecosystem of robust, production-ready tools.

Mojo provides a seamless and highly efficient Python interoperability layer,
allowing Mojo code to import and call any Python library directly using the
Python.import\_module("module\_name") function. The official semantic search
example from Modular leverages this exact feature to integrate with chromadb, a
popular Python-native vector database.1 While Mojo has a Foreign Function
Interface (FFI) for C/C++, there are no verified, documented examples of its use
with complex C++ vector search libraries like Faiss or HNSWlib, making that path
experimental and not a gold standard.2

Therefore, the recommended and verified pattern is to use Python for high-level
application logic and integration with its rich ecosystem, while delegating the
performance-critical, computationally-bound task of inference to the MAX Engine.
This approach leverages the strengths of both languages and represents the
intended design of the Modular ecosystem.

### **2.2 Recommended Indexing Algorithm: HNSW for Maximum Search Speed**

Vector databases employ specialized indexing algorithms to avoid slow,
brute-force searches. The two most common are Inverted File (IVFFlat) and
Hierarchical Navigable Small World (HNSW). For applications where search speed
is the absolute priority, **HNSW is the unambiguous choice**.

HNSW builds a multi-layered graph of vectors, allowing for a highly efficient,
logarithmic-time search that dramatically outperforms other methods in
low-latency query scenarios. The official Modular semantic search example
configures its chromadb instance to use an HNSW index with a cosine distance
metric, reinforcing this as the recommended practice.1 The primary trade-offs
are that HNSW requires more memory to store the graph structure and a longer,
more computationally intensive build process compared to IVFFlat. However, for a
production system where query latency is paramount, these are acceptable
one-time (or infrequent) costs.

| Attribute                  | HNSW                                                                 | IVFFlat                                                        |
| :------------------------- | :------------------------------------------------------------------- | :------------------------------------------------------------- |
| **Search Speed**           | **Very High**. Logarithmic complexity scales well with dataset size. | **High**. Speed is linearly dependent on the nprobe parameter. |
| **Build Time**             | Slower. Complex graph construction process.                          | Faster. Based on k-means clustering.                           |
| **Memory Usage**           | Higher. Stores graph connections for each point.                     | Lower. Stores centroids and vector lists.                      |
| **Recall-Speed Trade-off** | Tuned at query time with ef\_search parameter.                       | Tuned at query time with nprobe parameter.                     |
| **Data Distribution**      | Robust and less sensitive to skewed data.                            | Performance can degrade with unbalanced clusters.              |

_Table 2.1: Vector Index Algorithm Trade-offs (HNSW vs. IVFFlat). Data sourced
from technical analyses of vector indexing algorithms._

### **2.3 Code Pattern: Implementing a High-Speed HNSW Index with ChromaDB**

Implementing a high-speed HNSW index via Python interoperability is
straightforward. The pattern involves using the chromadb client library.

1. Import the library: chroma\_client \=
   Python.import\_module("chromadb").Client().
2. Create a collection, critically specifying the HNSW index algorithm and the
   distance metric in the metadata: collection \=
   chroma\_client.create\_collection(name="my\_hnsw\_collection",
   metadata={"hnsw:space": "cosine"}). Using "cosine" is standard for normalized
   text embeddings.1
3. Add documents and their embeddings to the collection using
   collection.upsert(). This method efficiently adds new data or updates
   existing entries.
4. Perform searches using collection.query(), passing the query embedding and
   the desired number of results (n\_results).

### **2.4 Tuning Parameters for Vector Search**

The HNSW algorithm offers several key parameters for tuning the trade-off
between search speed, recall (accuracy), and memory usage.

| Parameter        | Context    | Description                                                                     | Impact on Performance/Recall                                                                                                                                |
| :--------------- | :--------- | :------------------------------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| m                | Build Time | The maximum number of connections per node in the graph layers.                 | Higher values create a denser, more accurate graph, improving recall but increasing memory usage and build time.                                            |
| ef\_construction | Build Time | The size of the dynamic list for candidate neighbors during graph construction. | Higher values lead to a higher quality index (better recall) at the cost of a longer build time.                                                            |
| ef\_search       | Query Time | The size of the dynamic list for candidate neighbors during search.             | This is the primary query-time tuning knob. Increasing ef\_search improves recall but increases query latency. For the _fastest_ search, use a lower value. |
| distance         | Build Time | The distance metric used to measure vector similarity (e.g., cosine, l2, ip).1  | Must match the nature of the embeddings. cosine is standard for sentence transformers.                                                                      |

Table 2.2: HNSW Index Tuning Parameters. Parameters sourced from vector database
and algorithm documentation.1

## **Section 3: End-to-End Implementation Module for LLM Agent**

This section synthesizes the gold-standard patterns into a complete, actionable
implementation plan designed for direct use by an LLM code agent. It provides a
full code module structure and the necessary environment setup instructions.

### **3.1 Complete, Verified Code Module**

The following describes the structure for a single, cohesive Python file
(semantic\_search.py) that encapsulates the entire high-performance search
pipeline.

- **Class Definition**: A class named SemanticSearchEngine should be defined to
  manage all components.
- **\_\_init\_\_(self, model\_path)**: The constructor should:
  - Initialize the max.engine.InferenceSession.
  - Load the specified ONNX model using session.load(model\_path).
  - Initialize the transformers tokenizer for the model.
  - Initialize the chromadb client and create or get a collection configured for
    HNSW with a cosine distance metric.
- **index\_documents(self, documents: list\[str\], batch\_size: int \= 32\)**:
  This method should:
  - Take a list of strings as input.
  - Tokenize the documents.
  - Use a DataLoader to create batches.
  - Iterate through the batches, generating embeddings using the MAX Engine.
  - upsert the documents, embeddings, and unique IDs into the ChromaDB
    collection.
- **search(self, query: str, n\_results: int \= 10\)**: This method should:
  - Take a single query string.
  - Generate the embedding for the query using the MAX Engine.
  - Call collection.query() on the ChromaDB collection to retrieve the
    n\_results most similar documents.
  - Return the search results.
- **if \_\_name\_\_ \== "\_\_main\_\_": block**: This block should provide a
  simple demonstration of instantiating the SemanticSearchEngine, indexing a
  sample set of documents, and performing a search.

### **3.2 Configuration and Environment Setup**

The following steps are required to prepare the environment for running the
semantic search engine:

1. **Install the Modular SDK**: Follow the official instructions to install the
   Modular platform, which includes the MAX Engine and Mojo SDK.
2. **Create Project Environment**: Create a project directory and set up a
   Python virtual environment.\
   Bash\
   mkdir modular\_search && cd modular\_search\
   python3 \-m venv.venv\
   source.venv/bin/activate

3. **Install Dependencies**: Install the necessary Python packages using pip.\
   Bash\
   pip install modular-max chromadb transformers torch

4. **Download ONNX Model**: Download the bge-base-en-v1.5 model from the Hugging
   Face Hub. Ensure you download the files from the onnx revision of the
   repository.\
   Bash\
   \# Ensure git-lfs is installed\
   git lfs install\
   git clone https://huggingface.co/BAAI/bge-base-en-v1.5 \--branch onnx

   The model will be located at bge-base-en-v1.5/model.onnx.

## **Section 4: Conclusion and Definitive Recommendations**

This report has detailed the verified, gold-standard patterns for building a
high-performance semantic search engine on the Modular platform. The analysis
yields a clear and robust architectural recommendation.

### **4.1 Summary of the Gold-Standard Architecture**

The fastest, most reliable, and officially documented architecture for semantic
search within the Modular ecosystem is a **hybrid Python/Mojo system** that
leverages the distinct strengths of each component:

- **Inference Acceleration**: The core performance gain comes from using the
  **MAX Engine Python API (max.engine)** to execute a pre-trained **ONNX
  embedding model**. Maximizing throughput requires implementing **batched
  inference**, which significantly amortizes the overhead of model execution.
- **Ecosystem Integration**: The system should use **Mojo's Python
  interoperability layer** to call a mature and production-ready Python vector
  database library, such as chromadb. This avoids experimental, undocumented
  paths and leverages a vast ecosystem of tools.
- **Indexing Algorithm**: For the lowest possible query latency, the vector
  database must be configured to use an **HNSW index** with a **cosine**
  distance metric. This algorithm is purpose-built for high-speed, high-recall
  approximate nearest neighbor search.1

### **4.2 Final Recommendations for the LLM Agent**

To construct the fastest possible semantic search engine, the LLM agent must
adhere to the following directives:

1. **Prioritize Inference Optimization**: Focus all performance tuning efforts
   on the MAX Engine execution. This includes maximizing batch size, ensuring
   correct device placement (CPU/GPU), and using the optimized model loading and
   execution patterns detailed in Section 1\. This is where the most significant
   speed improvements are realized.
2. **Embrace Python Interoperability**: Do not attempt to find, write, or use a
   native Mojo vector search library. The verified, supported, and gold-standard
   pattern is to call an established Python library via Python.import\_module().
   This is a design feature of the Modular ecosystem, not a limitation.
3. **Default to HNSW for Speed**: The HNSW indexing algorithm should be the
   default choice for all speed-critical applications. The ef\_search parameter
   should be exposed as a configurable option to allow end-users to balance the
   trade-off between search speed and recall.
4. **Standardize on the ONNX Format**: The ONNX model format should be used as
   the stable and performant interface between the embedding model and the MAX
   Engine. The agent's workflow should assume the model is available in this
   format.

#### **Works cited**

1. Semantic Search with MAX Engine \- Modular, accessed July 3, 2025,
   [https://www.modular.com/blog/semantic-search-with-max-engine](https://www.modular.com/blog/semantic-search-with-max-engine)
2. ihnorton/mojo-ffi: Mojo FFI demos: dynamic linking methods ... \- GitHub,
   accessed July 3, 2025,
   [https://github.com/ihnorton/mojo-ffi](https://github.com/ihnorton/mojo-ffi)
