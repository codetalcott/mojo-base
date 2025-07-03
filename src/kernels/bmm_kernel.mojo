"""
Batched Matrix Multiplication (BMM) Kernel for Similarity Search.
Ultra-fast SIMD-accelerated similarity computation for semantic search.
"""

from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from algorithm import parallelize, vectorize
from memory import UnsafePointer
from math import sqrt
from sys import simdwidthof
from gpu import (
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    global_idx,
    lane_id,
    thread_idx,
)
from gpu.host import DeviceContext, FuncAttribute

@parameter
struct BMMKernel:
    """
    Batched Matrix Multiplication kernel optimized for similarity search.
    
    Optimizations:
    - SIMD acceleration for parallel computation
    - Memory-aligned access patterns
    - Cache-friendly tiling for large datasets
    - Fused operations (normalization + similarity)
    """
    alias embed_dim: Int = 768
    alias nelts = simdwidthof[DType.float32]()
    alias tile_size: Int = 64  # Tile size for cache optimization
    
    var corpus_embeddings: UnsafePointer[Float32]  # N x 768 matrix
    var corpus_norms: UnsafePointer[Float32]       # Precomputed L2 norms
    var corpus_size: Int
    var is_normalized: Bool
    
    fn __init__(out self, corpus_size: Int) raises:
        """Initialize BMM kernel with corpus capacity."""
        if corpus_size <= 0:
            raise Error("Corpus size must be positive")
        
        self.corpus_size = corpus_size
        self.is_normalized = False
        
        # Allocate aligned memory for optimal SIMD performance
        var alignment = self.nelts * 4  # 4 bytes per float32
        try:
            self.corpus_embeddings = UnsafePointer[Float32].alloc(
                corpus_size * self.embed_dim
            )
            self.corpus_norms = UnsafePointer[Float32].alloc(
                corpus_size
            )
        except:
            raise Error("Failed to allocate memory for BMM kernel")
    
    fn __del__(owned self):
        """Clean up allocated memory."""
        self.corpus_embeddings.free()
        self.corpus_norms.free()
    
    fn load_corpus(mut self, embeddings: NDBuffer[_, 2, _, _]) raises:
        """Load corpus embeddings and precompute norms."""
        # Validate input dimensions
        if embeddings.shape()[0] > self.corpus_size:
            raise Error("Input embeddings exceed corpus capacity")
        if embeddings.shape()[1] != self.embed_dim:
            raise Error("Embedding dimension mismatch")
        
        # Copy embeddings to aligned memory with bounds checking
        var actual_corpus_size = min(embeddings.shape()[0], self.corpus_size)
        for i in range(actual_corpus_size):
            for j in range(self.embed_dim):
                var idx = i * self.embed_dim + j
                self.corpus_embeddings.store(idx, embeddings[i, j])
        
        # Precompute L2 norms for normalization
        self._precompute_norms()
        self.is_normalized = True
    
    @parameter
    fn _precompute_norms(mut self):
        """Precompute L2 norms for all corpus embeddings."""
        @parameter
        fn compute_norm(i: Int):
            var norm_squared: Float32 = 0.0
            
            # SIMD-accelerated norm computation
            @parameter
            fn accumulate_norm(j: Int):
                var vec = self.corpus_embeddings.simd_load[self.nelts](
                    i * self.embed_dim + j
                )
                norm_squared += (vec * vec).reduce_add()
            
            vectorize[self.nelts, accumulate_norm](self.embed_dim)
            
            # Store inverse norm for efficient division
            var norm = sqrt(norm_squared)
            self.corpus_norms.store(i, rsqrt(norm_squared) if norm > 1e-8 else 0.0)
        
        parallelize[compute_norm](self.corpus_size)
    
    @parameter
    fn cosine_similarity_batch(self, 
                              query: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Compute cosine similarity between query and all corpus embeddings.
        
        Args:
            query: Single query embedding [embed_dim]
            
        Returns:
            Similarity scores for all corpus embeddings [corpus_size]
        """
        var similarities = Tensor[DType.float32](self.corpus_size)
        
        # Normalize query vector
        let query_norm = self._compute_query_norm(query)
        
        # Compute similarities in tiles for cache efficiency
        @parameter
        fn process_tile(tile_start: Int):
            let tile_end = min(tile_start + self.tile_size, self.corpus_size)
            
            for i in range(tile_start, tile_end):
                let similarity = self._compute_cosine_similarity_single(
                    query, i, query_norm
                )
                similarities[i] = similarity
        
        # Process corpus in tiles
        let num_tiles = (self.corpus_size + self.tile_size - 1) // self.tile_size
        parallelize[process_tile](num_tiles)
        
        return similarities
    
    @parameter
    fn _compute_query_norm(self, query: Tensor[DType.float32]) -> Float32:
        """Compute L2 norm of query vector with SIMD."""
        var norm_squared: Float32 = 0.0
        
        @parameter
        fn accumulate_norm(i: Int):
            let vec = query.simd_load[self.nelts](i)
            norm_squared += (vec * vec).reduce_add()
        
        vectorize[self.nelts, accumulate_norm](self.embed_dim)
        return rsqrt(norm_squared) if norm_squared > 1e-8 else 0.0
    
    @parameter
    fn _compute_cosine_similarity_single(self,
                                       query: Tensor[DType.float32],
                                       corpus_idx: Int,
                                       query_norm: Float32) -> Float32:
        """Compute cosine similarity between query and single corpus embedding."""
        var dot_product: Float32 = 0.0
        let corpus_offset = corpus_idx * self.embed_dim
        
        # SIMD-accelerated dot product
        @parameter
        fn compute_dot_product(i: Int):
            let query_vec = query.simd_load[self.nelts](i)
            let corpus_vec = self.corpus_embeddings.simd_load[self.nelts](
                corpus_offset + i
            )
            dot_product += (query_vec * corpus_vec).reduce_add()
        
        vectorize[self.nelts, compute_dot_product](self.embed_dim)
        
        # Return normalized similarity
        let corpus_norm = self.corpus_norms.load(corpus_idx)
        return dot_product * query_norm * corpus_norm
    
    @parameter
    fn batched_similarity_top_k(self,
                               query: Tensor[DType.float32],
                               k: Int) -> Tuple[Tensor[DType.float32], Tensor[DType.int32]]:
        """
        Compute top-k most similar embeddings efficiently.
        
        Args:
            query: Query embedding [embed_dim]
            k: Number of top results to return
            
        Returns:
            Tuple of (scores, indices) for top-k results
        """
        # Compute all similarities
        let all_similarities = self.cosine_similarity_batch(query)
        
        # Find top-k using partial sort
        var top_scores = Tensor[DType.float32](k)
        var top_indices = Tensor[DType.int32](k)
        
        # Initialize with first k elements
        for i in range(k):
            top_scores[i] = all_similarities[i]
            top_indices[i] = i
        
        # Simple selection for remaining elements (can be optimized with heap)
        for i in range(k, self.corpus_size):
            let current_score = all_similarities[i]
            
            # Find minimum in current top-k
            var min_idx = 0
            var min_score = top_scores[0]
            
            for j in range(1, k):
                if top_scores[j] < min_score:
                    min_score = top_scores[j]
                    min_idx = j
            
            # Replace if current score is better
            if current_score > min_score:
                top_scores[min_idx] = current_score
                top_indices[min_idx] = i
        
        return (top_scores, top_indices)
    
    @parameter
    fn fused_search_and_rank(self,
                            query: Tensor[DType.float32],
                            project_filter: String = "",
                            k: Int = 10) -> Tensor[DType.float32]:
        """
        Fused operation: similarity search + project filtering + ranking.
        
        This combines multiple operations for maximum efficiency:
        1. Similarity computation
        2. Project-based filtering  
        3. Top-k selection
        4. Context-aware ranking
        """
        # TODO: Implement project filtering logic
        # TODO: Add context-aware ranking weights
        
        let (scores, indices) = self.batched_similarity_top_k(query, k)
        return scores
    
    fn get_performance_metrics(self) -> String:
        """Return performance and memory usage statistics."""
        var memory_usage_mb = (self.corpus_size * self.embed_dim * 4) / (1024 * 1024)
        
        return (
            "BMM Kernel Stats:\n" +
            "- Corpus Size: " + str(self.corpus_size) + "\n" +
            "- Embedding Dim: " + str(self.embed_dim) + "\n" +
            "- Memory Usage: " + str(memory_usage_mb) + " MB\n" +
            "- SIMD Width: " + str(self.nelts) + "\n" +
            "- Tile Size: " + str(self.tile_size) + "\n" +
            "- Normalized: " + str(self.is_normalized)
        )

# High-level API functions
@parameter
fn create_bmm_kernel(corpus_size: Int) -> BMMKernel:
    """Factory function to create optimized BMM kernel."""
    return BMMKernel(corpus_size)

@parameter
fn benchmark_bmm_kernel(kernel: BMMKernel, num_queries: Int = 100) -> Float64:
    """Benchmark BMM kernel search performance."""
    # Create test query
    var test_query = Tensor[DType.float32](BMMKernel.embed_dim)
    for i in range(BMMKernel.embed_dim):
        test_query[i] = Float32(i) / 1000.0
    
    let start_time = time.now()
    
    # Run search benchmark
    for _ in range(num_queries):
        let _ = kernel.cosine_similarity_batch(test_query)
    
    let end_time = time.now()
    let total_time = (end_time - start_time).to_float64()
    
    return total_time / Float64(num_queries)  # Average time per query

@parameter
fn benchmark_memory_bandwidth(kernel: BMMKernel) -> Float64:
    """Benchmark memory bandwidth utilization."""
    # Create test query
    var test_query = Tensor[DType.float32](BMMKernel.embed_dim)
    for i in range(BMMKernel.embed_dim):
        test_query[i] = 1.0
    
    let iterations = 100
    let start_time = time.now()
    
    for _ in range(iterations):
        let _ = kernel.cosine_similarity_batch(test_query)
    
    let end_time = time.now()
    let total_time = (end_time - start_time).to_float64()
    
    # Calculate bandwidth (bytes transferred per second)
    let bytes_per_iteration = kernel.corpus_size * BMMKernel.embed_dim * 4  # 4 bytes per float
    let total_bytes = Float64(bytes_per_iteration * iterations)
    
    return total_bytes / total_time  # Bytes per second