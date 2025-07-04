"""
MAX Graph API Implementation for Semantic Search
Leverages MAX's graph optimizations while maintaining compatibility with our autotuning framework
"""

import max.graph as g
from max.graph import ops, TensorType, DeviceRef
from max.dtype import DType
from max.engine import InferenceSession
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass

@dataclass
class MaxGraphConfig:
    """Configuration for MAX Graph semantic search."""
    corpus_size: int
    vector_dims: int = 768
    batch_size: int = 1
    device: str = "cpu"  # or "gpu"
    use_fp16: bool = False
    enable_fusion: Optional[bool] = None  # Auto-detect based on device capabilities
    
    def __post_init__(self):
        # Future-proof adaptive fusion based on device capabilities
        if self.enable_fusion is None:
            self.enable_fusion = self._detect_optimal_fusion_setting()
    
    def _detect_optimal_fusion_setting(self) -> bool:
        """
        Detect optimal fusion setting based on device capabilities.
        Future-proof for Apple Metal and other GPU architectures.
        """
        # Check if device has GPU-like parallel processing capabilities
        if self._is_parallel_compute_device():
            return True  # GPU-like devices benefit from fusion
        else:
            return False  # CPU-like devices show minimal/negative benefit
    
    def _is_parallel_compute_device(self) -> bool:
        """
        Detect if device has parallel compute capabilities.
        Handles current and future GPU architectures automatically.
        """
        device_lower = self.device.lower()
        
        # Known parallel compute indicators
        parallel_indicators = [
            'gpu',           # Generic GPU
            'cuda',          # NVIDIA CUDA
            'metal',         # Apple Metal (future)
            'opencl',        # OpenCL devices
            'vulkan',        # Vulkan compute
            'rocm',          # AMD ROCm
            'dml',           # DirectML
            'tensorrt',      # TensorRT optimized
            'mlx',           # Apple MLX framework
        ]
        
        # Check if device string contains any parallel compute indicators
        return any(indicator in device_lower for indicator in parallel_indicators)

class MaxSemanticSearchGraph:
    """
    MAX Graph implementation of semantic search with automatic optimizations.
    
    This implementation leverages MAX's:
    - Automatic kernel fusion
    - Graph-level optimizations
    - Hardware-agnostic execution
    - Memory optimization
    """
    
    def __init__(self, config: MaxGraphConfig):
        self.config = config
        self.graph = None
        self.session = None
        self.model = None
        self._build_graph()
    
    def _build_graph(self):
        """Build the MAX Graph for semantic search computation."""
        print("🔧 Building MAX Graph for semantic search...")
        
        # Define input tensors using correct imports
        dtype = DType.float16 if self.config.use_fp16 else DType.float32
        device = DeviceRef.CPU() if self.config.device == "cpu" else DeviceRef.GPU()
        
        # Query tensor: [batch_size, vector_dims]
        self.query_input = TensorType(dtype, [self.config.batch_size, self.config.vector_dims], device)
        
        # Corpus tensor: [corpus_size, vector_dims]
        self.corpus_input = TensorType(dtype, [self.config.corpus_size, self.config.vector_dims], device)
        
        # Build computational graph with MAX optimizations
        self.graph = self._create_optimized_similarity_graph()
        
        print(f"✅ MAX Graph built for {self.config.corpus_size:,} vectors ({self.config.vector_dims}D)")
    
    def _create_optimized_similarity_graph(self) -> g.Graph:
        """Create optimized similarity computation graph using MAX Graph API."""
        
        def forward(query, corpus):
            # Simplified approach - just do matrix multiplication without normalization first
            # to isolate the issue
            corpus_transposed = ops.transpose(corpus, axis_1=0, axis_2=1)
            similarities = ops.matmul(query, corpus_transposed)
            return similarities
        
        # Create the graph with forward function and input types
        graph = g.Graph(
            name="semantic_search_graph",
            forward=forward,
            input_types=[self.query_input, self.corpus_input]
        )
        
        return graph
    
    def compile(self, device_target: Optional[str] = None) -> None:
        """Compile the graph for target device with MAX optimizations."""
        device = device_target or self.config.device
        
        print(f"🚀 Compiling MAX Graph for {device}...")
        start_time = time.time()
        
        try:
            # CORRECT PATTERN: Create session, then load graph to get model
            self.session = InferenceSession()
            self.model = self.session.load(self.graph)
            
            compile_time = time.time() - start_time
            print(f"✅ MAX Graph compiled in {compile_time:.3f}s with automatic optimizations")
            
        except Exception as e:
            compile_time = time.time() - start_time
            print(f"❌ MAX Graph compilation failed: {e}")
            self.session = None
            self.model = None
    
    def search_similarity(self, query_embedding: np.ndarray, 
                         corpus_embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform semantic similarity search using MAX Graph.
        
        Args:
            query_embedding: Query vector [1, vector_dims] or [vector_dims]
            corpus_embeddings: Corpus vectors [corpus_size, vector_dims]
            
        Returns:
            Dictionary with similarity scores and optional probabilities
        """
        if self.model is None:
            raise RuntimeError("Graph not compiled. Call compile() first.")
        
        # Ensure correct input shapes
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Validate input dimensions
        assert query_embedding.shape[1] == self.config.vector_dims, \
            f"Query dimension mismatch: {query_embedding.shape[1]} vs {self.config.vector_dims}"
        assert corpus_embeddings.shape == (self.config.corpus_size, self.config.vector_dims), \
            f"Corpus shape mismatch: {corpus_embeddings.shape} vs ({self.config.corpus_size}, {self.config.vector_dims})"
        
        # Convert to appropriate dtype
        if self.config.use_fp16:
            query_embedding = query_embedding.astype(np.float16)
            corpus_embeddings = corpus_embeddings.astype(np.float16)
        else:
            query_embedding = query_embedding.astype(np.float32)
            corpus_embeddings = corpus_embeddings.astype(np.float32)
        
        # Execute using CORRECT PATTERN: individual numpy arrays as arguments
        start_time = time.time()
        
        outputs = self.model.execute(query_embedding, corpus_embeddings)
        
        execution_time = time.time() - start_time
        
        # Parse outputs - MAX returns list of max.driver.Tensor objects
        # Convert back to numpy arrays for compatibility
        def tensor_to_numpy(tensor):
            """Convert max.driver.Tensor to numpy array."""
            if hasattr(tensor, 'to_numpy'):
                return tensor.to_numpy()
            elif hasattr(tensor, 'numpy'):
                return tensor.numpy()
            elif hasattr(tensor, '__array__'):
                return np.array(tensor)
            else:
                # Fallback - try to convert directly
                return np.array(tensor)
        
        similarities = tensor_to_numpy(outputs[0])
        result = {
            'similarities': similarities,
            'execution_time_ms': execution_time * 1000
        }
        
        if len(outputs) > 1:
            result['probabilities'] = tensor_to_numpy(outputs[1])
        
        return result
    
    def get_top_k_results(self, similarities: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Get top-k most similar results."""
        if similarities.ndim == 2:
            similarities = similarities.flatten()
        
        # Get indices of top-k similarities
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores

class MaxSemanticSearchBenchmark:
    """
    Benchmarking wrapper that integrates MAX Graph with our autotuning framework.
    Allows performance comparison between MAX Graph and manual Mojo implementations.
    """
    
    def __init__(self, config: MaxGraphConfig):
        self.config = config
        self.max_search = MaxSemanticSearchGraph(config)
        self.performance_metrics = {}
    
    def setup_for_autotuning(self, autotuning_config: Dict[str, Any]) -> None:
        """Configure MAX Graph based on autotuning parameters."""
        
        # Update MAX configuration based on autotuning parameters
        if 'use_fp16' in autotuning_config:
            self.config.use_fp16 = autotuning_config['use_fp16']
        
        if 'enable_fusion' in autotuning_config:
            self.config.enable_fusion = autotuning_config['enable_fusion']
        
        if 'device' in autotuning_config:
            self.config.device = autotuning_config['device']
        
        # Rebuild and recompile graph with new configuration
        self.max_search = MaxSemanticSearchGraph(self.config)
        self.max_search.compile()
    
    def benchmark_configuration(self, query_embeddings: np.ndarray, 
                              corpus_embeddings: np.ndarray,
                              iterations: int = 5) -> Dict[str, float]:
        """
        Benchmark current MAX Graph configuration.
        Compatible with our existing autotuning framework.
        """
        
        latencies = []
        throughputs = []
        
        print(f"🔧 Benchmarking MAX Graph configuration...")
        print(f"   Corpus: {self.config.corpus_size:,} vectors")
        print(f"   Device: {self.config.device}")
        print(f"   FP16: {self.config.use_fp16}")
        print(f"   Fusion: {self.config.enable_fusion}")
        
        for i in range(iterations):
            result = self.max_search.search_similarity(query_embeddings[0], corpus_embeddings)
            
            latency_ms = result['execution_time_ms']
            latencies.append(latency_ms)
            
            # Calculate throughput (vectors processed per second)
            vectors_processed = self.config.corpus_size
            throughput = vectors_processed / (latency_ms / 1000.0)
            throughputs.append(throughput)
            
            print(f"     Iteration {i+1}: {latency_ms:.3f}ms")
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        avg_throughput = np.mean(throughputs)
        
        metrics = {
            'avg_latency_ms': avg_latency,
            'std_latency_ms': std_latency,
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'avg_throughput_vectors_per_sec': avg_throughput,
            'corpus_size': self.config.corpus_size,
            'vector_dims': self.config.vector_dims,
            'device': self.config.device,
            'use_fp16': self.config.use_fp16,
            'enable_fusion': self.config.enable_fusion
        }
        
        self.performance_metrics = metrics
        
        print(f"✅ MAX Graph benchmark complete:")
        print(f"   Avg latency: {avg_latency:.3f}ms ± {std_latency:.3f}ms")
        print(f"   Throughput: {avg_throughput:.1f} vectors/sec")
        
        return metrics
    
    def compare_with_baseline(self, baseline_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compare MAX Graph performance with baseline (manual Mojo) implementation."""
        
        if not self.performance_metrics:
            raise RuntimeError("Run benchmark_configuration() first")
        
        max_latency = self.performance_metrics['avg_latency_ms']
        baseline_latency = baseline_metrics.get('avg_latency_ms', 0)
        
        if baseline_latency > 0:
            speedup = baseline_latency / max_latency
            improvement_pct = ((baseline_latency - max_latency) / baseline_latency) * 100
        else:
            speedup = 1.0
            improvement_pct = 0.0
        
        comparison = {
            'max_graph_latency_ms': max_latency,
            'baseline_latency_ms': baseline_latency,
            'speedup_factor': speedup,
            'improvement_percent': improvement_pct,
            'max_graph_throughput': self.performance_metrics['avg_throughput_vectors_per_sec'],
            'baseline_throughput': baseline_metrics.get('avg_throughput_vectors_per_sec', 0),
            'recommendation': 'MAX Graph' if speedup > 1.1 else 'Baseline' if speedup < 0.9 else 'Similar'
        }
        
        print(f"\n📊 MAX Graph vs Baseline Comparison:")
        print(f"   MAX Graph: {max_latency:.3f}ms")
        print(f"   Baseline:  {baseline_latency:.3f}ms")
        print(f"   Speedup:   {speedup:.2f}x")
        print(f"   Improvement: {improvement_pct:+.1f}%")
        print(f"   Recommendation: {comparison['recommendation']}")
        
        return comparison

def create_test_data(corpus_size: int, vector_dims: int = 768) -> Tuple[np.ndarray, np.ndarray]:
    """Create test data for benchmarking."""
    np.random.seed(42)  # Reproducible results
    
    # Generate random normalized vectors
    query_embeddings = np.random.randn(5, vector_dims).astype(np.float32)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    
    corpus_embeddings = np.random.randn(corpus_size, vector_dims).astype(np.float32)
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    
    return query_embeddings, corpus_embeddings

# Example usage and testing
if __name__ == "__main__":
    print("🧪 MAX Graph Semantic Search Implementation")
    print("==========================================")
    
    # Test configuration
    config = MaxGraphConfig(
        corpus_size=10000,
        vector_dims=768,
        device="cpu",
        use_fp16=False,
        enable_fusion=True
    )
    
    # Create test data
    query_embeddings, corpus_embeddings = create_test_data(config.corpus_size, config.vector_dims)
    
    # Initialize and compile MAX Graph
    max_search = MaxSemanticSearchGraph(config)
    max_search.compile()
    
    # Test search functionality
    result = max_search.search_similarity(query_embeddings[0], corpus_embeddings)
    top_indices, top_scores = max_search.get_top_k_results(result['similarities'], k=5)
    
    print(f"\n🔍 Search Results:")
    print(f"   Execution time: {result['execution_time_ms']:.3f}ms")
    print(f"   Top 5 similarities: {top_scores}")
    print(f"   Top 5 indices: {top_indices}")
    
    # Test benchmarking functionality
    benchmark = MaxSemanticSearchBenchmark(config)
    metrics = benchmark.benchmark_configuration(query_embeddings, corpus_embeddings)
    
    print(f"\n✅ MAX Graph implementation ready for integration with autotuning framework!")