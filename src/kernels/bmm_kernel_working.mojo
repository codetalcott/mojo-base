"""
Working BMM Kernel for Similarity Search
Uses only confirmed available Mojo features
"""

from memory import UnsafePointer
from math import sqrt
from sys import simdwidthof

# Working version using confirmed features
struct BMMKernel:
    """
    Batched Matrix Multiplication kernel for similarity search.
    Simplified implementation using only available Mojo features.
    """
    alias embed_dim: Int = 768
    alias simd_width = simdwidthof[DType.float32]()
    
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
        
        # Allocate memory for embeddings and norms
        self.corpus_embeddings = UnsafePointer[Float32].alloc(
            corpus_size * self.embed_dim
        )
        self.corpus_norms = UnsafePointer[Float32].alloc(corpus_size)
        
        # Initialize to zero
        for i in range(corpus_size * self.embed_dim):
            self.corpus_embeddings[i] = 0.0
        
        for i in range(corpus_size):
            self.corpus_norms[i] = 0.0
    
    fn __del__(owned self):
        """Clean up allocated memory."""
        self.corpus_embeddings.free()
        self.corpus_norms.free()
    
    fn load_corpus_data(mut self, embeddings_data: UnsafePointer[Float32], num_vectors: Int) raises:
        """Load corpus embeddings from external data."""
        if num_vectors > self.corpus_size:
            raise Error("Too many vectors for corpus capacity")
        
        # Copy embeddings
        for i in range(num_vectors):
            for j in range(self.embed_dim):
                var idx = i * self.embed_dim + j
                self.corpus_embeddings[idx] = embeddings_data[idx]
        
        # Precompute norms
        self._precompute_norms(num_vectors)
        self.is_normalized = True
    
    fn _precompute_norms(mut self, num_vectors: Int):
        """Precompute L2 norms for all corpus embeddings."""
        for i in range(num_vectors):
            var norm_squared: Float32 = 0.0
            
            # Compute norm for this vector
            for j in range(self.embed_dim):
                var idx = i * self.embed_dim + j
                var val = self.corpus_embeddings[idx]
                norm_squared += val * val
            
            # Store the norm (not squared)
            var norm = sqrt(norm_squared)
            self.corpus_norms[i] = norm if norm > 1e-8 else 1e-8  # Avoid division by zero
    
    fn cosine_similarity_single(self, query: UnsafePointer[Float32], corpus_idx: Int) -> Float32:
        """Compute cosine similarity between query and one corpus vector."""
        if not self.is_normalized:
            return 0.0
        
        # Compute dot product
        var dot_product: Float32 = 0.0
        var corpus_offset = corpus_idx * self.embed_dim
        
        for i in range(self.embed_dim):
            dot_product += query[i] * self.corpus_embeddings[corpus_offset + i]
        
        # Compute query norm
        var query_norm_squared: Float32 = 0.0
        for i in range(self.embed_dim):
            query_norm_squared += query[i] * query[i]
        
        var query_norm = sqrt(query_norm_squared)
        var corpus_norm = self.corpus_norms[corpus_idx]
        
        # Compute cosine similarity
        if query_norm > 1e-8 and corpus_norm > 1e-8:
            return dot_product / (query_norm * corpus_norm)
        else:
            return 0.0
    
    fn cosine_similarity_batch(self, query: UnsafePointer[Float32], results: UnsafePointer[Float32], num_vectors: Int):
        """Compute cosine similarity between query and all corpus vectors."""
        if not self.is_normalized:
            return
        
        # Precompute query norm
        var query_norm_squared: Float32 = 0.0
        for i in range(self.embed_dim):
            query_norm_squared += query[i] * query[i]
        var query_norm = sqrt(query_norm_squared)
        
        # Compute similarities
        for vec_idx in range(num_vectors):
            var dot_product: Float32 = 0.0
            var corpus_offset = vec_idx * self.embed_dim
            
            # Dot product computation
            for dim_idx in range(self.embed_dim):
                dot_product += query[dim_idx] * self.corpus_embeddings[corpus_offset + dim_idx]
            
            # Cosine similarity
            var corpus_norm = self.corpus_norms[vec_idx]
            if query_norm > 1e-8 and corpus_norm > 1e-8:
                results[vec_idx] = dot_product / (query_norm * corpus_norm)
            else:
                results[vec_idx] = 0.0
    
    fn find_top_k(self, query: UnsafePointer[Float32], k: Int, num_vectors: Int, 
                  top_indices: UnsafePointer[Int], top_scores: UnsafePointer[Float32]):
        """Find top-k most similar vectors."""
        # Allocate temporary results
        var all_similarities = UnsafePointer[Float32].alloc(num_vectors)
        
        # Compute all similarities
        self.cosine_similarity_batch(query, all_similarities, num_vectors)
        
        # Simple selection sort for top-k (could be optimized)
        for i in range(k):
            var max_idx = 0
            var max_score: Float32 = -1.0
            
            # Find maximum in remaining elements
            for j in range(num_vectors):
                if all_similarities[j] > max_score:
                    max_score = all_similarities[j]
                    max_idx = j
            
            # Store result
            top_indices[i] = max_idx
            top_scores[i] = max_score
            
            # Mark as used
            all_similarities[max_idx] = -2.0  # Lower than any valid similarity
        
        all_similarities.free()
    
    fn get_stats(self):
        """Get kernel statistics."""
        print("BMM Kernel:", self.corpus_size, "vectors,", self.embed_dim, "dimensions")

# Test function to verify the kernel works
fn test_bmm_kernel():
    """Test the BMM kernel with sample data."""
    print("🧪 Testing BMM Kernel")
    print("=====================")
    
    var corpus_size = 100
    var embed_dim = 768
    
    # Create kernel
    try:
        var kernel = BMMKernel(corpus_size)
        
        # Create test data
        var test_embeddings = UnsafePointer[Float32].alloc(corpus_size * embed_dim)
        var query = UnsafePointer[Float32].alloc(embed_dim)
        
        # Initialize with some test data
        for i in range(corpus_size):
            for j in range(embed_dim):
                var idx = i * embed_dim + j
                test_embeddings[idx] = Float32(i + j) / Float32(embed_dim)
        
        for j in range(embed_dim):
            query[j] = Float32(j) / Float32(embed_dim)
        
        # Load data into kernel
        kernel.load_corpus_data(test_embeddings, corpus_size)
        
        # Test single similarity
        var similarity = kernel.cosine_similarity_single(query, 0)
        print("✅ Single similarity:", similarity)
        
        # Test batch similarities
        var results = UnsafePointer[Float32].alloc(corpus_size)
        kernel.cosine_similarity_batch(query, results, corpus_size)
        print("✅ Batch similarities[0-2]:", results[0], results[1], results[2])
        
        # Test top-k
        var k = 5
        var top_indices = UnsafePointer[Int].alloc(k)
        var top_scores = UnsafePointer[Float32].alloc(k)
        
        kernel.find_top_k(query, k, corpus_size, top_indices, top_scores)
        print("✅ Top-3 indices:", top_indices[0], top_indices[1], top_indices[2])
        print("✅ Top-3 scores:", top_scores[0], top_scores[1], top_scores[2])
        
        # Cleanup
        test_embeddings.free()
        query.free()
        results.free()
        top_indices.free()
        top_scores.free()
        
        print("✅ BMM Kernel test completed successfully!")
        
    except e:
        print("❌ BMM Kernel test failed:", e)

fn main():
    """Main test function."""
    test_bmm_kernel()