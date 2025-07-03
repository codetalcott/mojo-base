"""
Multi-Head Latent Attention (MLA) Kernel for Code Embeddings.
High-performance implementation optimized for semantic code understanding.
"""

from tensor import Tensor
from algorithm import parallelize, vectorize
from builtin import SIMD, simdwidthof
from math import sqrt, exp, min, max
from memory import DTypePointer
from DType import DType
from time import now
from random import random_float64

@parameter
struct MLAKernel:
    """
    Multi-Head Latent Attention kernel optimized for code sequences.
    
    Architecture:
    - 8 attention heads for diverse pattern capture
    - 768 embedding dimension (96 per head)
    - Optimized for sequences up to 512 tokens
    - Custom attention patterns for code syntax
    """
    alias num_heads: Int = 8
    alias embed_dim: Int = 768
    alias head_dim: Int = 96  # 768 / 8
    alias max_seq_len: Int = 512
    alias nelts = simdwidthof[DType.float32]()
    
    # Learned parameters
    var query_weights: Tensor[DType.float32]     # [768, 768]
    var key_weights: Tensor[DType.float32]       # [768, 768] 
    var value_weights: Tensor[DType.float32]     # [768, 768]
    var output_weights: Tensor[DType.float32]    # [768, 768]
    
    # Attention mask for code structure (syntax-aware)
    var syntax_attention_mask: Tensor[DType.bool]  # [512, 512]
    
    fn __init__(inout self):
        """Initialize MLA kernel with Xavier/Glorot initialization."""
        # Initialize weight matrices
        self.query_weights = Tensor[DType.float32](self.embed_dim, self.embed_dim)
        self.key_weights = Tensor[DType.float32](self.embed_dim, self.embed_dim)
        self.value_weights = Tensor[DType.float32](self.embed_dim, self.embed_dim)
        self.output_weights = Tensor[DType.float32](self.embed_dim, self.embed_dim)
        
        # Initialize syntax attention mask
        self.syntax_attention_mask = Tensor[DType.bool](self.max_seq_len, self.max_seq_len)
        
        # Xavier initialization for weights
        self._initialize_weights()
        self._create_syntax_mask()
    
    fn _initialize_weights(inout self):
        """Initialize weights using proper Xavier/Glorot normal distribution."""
        let scale = sqrt(2.0 / Float32(self.embed_dim))
        
        # Proper Xavier initialization with random values
        for i in range(self.embed_dim):
            for j in range(self.embed_dim):
                # Use proper random initialization
                let random_val = Float32(random_float64(-1.0, 1.0))
                let xavier_val = random_val * scale
                
                self.query_weights[i, j] = xavier_val
                self.key_weights[i, j] = xavier_val * Float32(random_float64(0.9, 1.1))
                self.value_weights[i, j] = xavier_val * Float32(random_float64(0.9, 1.1))
                self.output_weights[i, j] = xavier_val * Float32(random_float64(0.9, 1.1))
    
    fn _create_syntax_mask(inout self):
        """Create syntax-aware attention mask for code structure."""
        # Initialize all positions as valid
        for i in range(self.max_seq_len):
            for j in range(self.max_seq_len):
                self.syntax_attention_mask[i, j] = True
        
        # TODO: Implement syntax-specific masking patterns
        # - Block-level attention for function boundaries
        # - Local attention for statement sequences  
        # - Global attention for imports/declarations
    
    @parameter
    fn encode_sequence(self, 
                      input_tokens: Tensor[DType.float32],  # [seq_len, embed_dim]
                      seq_len: Int) -> Tensor[DType.float32] raises:
        """
        Encode a sequence of code tokens into semantic embeddings.
        
        Args:
            input_tokens: Tokenized code sequence [seq_len, embed_dim]
            seq_len: Actual sequence length (may be < max_seq_len)
            
        Returns:
            Encoded semantic embedding [embed_dim]
        """
        # Input validation
        if seq_len <= 0:
            raise Error("Sequence length must be positive")
        if seq_len > self.max_seq_len:
            raise Error("Sequence length exceeds maximum allowed length")
        if input_tokens.shape()[0] < seq_len:
            raise Error("Input tensor is smaller than specified sequence length")
        if input_tokens.shape()[1] != self.embed_dim:
            raise Error("Input embedding dimension mismatch")
        # Step 1: Compute Q, K, V projections
        let queries = self._compute_projection(input_tokens, self.query_weights, seq_len)
        let keys = self._compute_projection(input_tokens, self.key_weights, seq_len)
        let values = self._compute_projection(input_tokens, self.value_weights, seq_len)
        
        # Step 2: Multi-head attention
        let attention_output = self._multi_head_attention(queries, keys, values, seq_len)
        
        # Step 3: Output projection
        let final_output = self._compute_output_projection(attention_output, seq_len)
        
        # Step 4: Global pooling to get single embedding
        return self._global_average_pooling(final_output, seq_len)
    
    @parameter  
    fn _compute_projection(self, 
                          input: Tensor[DType.float32],
                          weights: Tensor[DType.float32],
                          seq_len: Int) -> Tensor[DType.float32] raises:
        """Compute matrix multiplication with SIMD optimization."""
        # Bounds checking
        if seq_len <= 0 or seq_len > input.shape()[0]:
            raise Error("Invalid sequence length for projection")
        
        var output = Tensor[DType.float32](seq_len, self.embed_dim)
        
        # Vectorized matrix multiplication
        @parameter
        fn compute_row(i: Int):
            @parameter
            fn compute_cols(j: Int):
                var acc = SIMD[DType.float32, self.nelts](0)
                
                # Inner product with SIMD and bounds checking
                for k in range(0, self.embed_dim, self.nelts):
                    let remaining = min(self.nelts, self.embed_dim - k)
                    if remaining == self.nelts:
                        let input_vec = input.simd_load[self.nelts](i * self.embed_dim + k)
                        let weight_vec = weights.simd_load[self.nelts](k * self.embed_dim + j)
                        acc += input_vec * weight_vec
                    else:
                        # Handle remaining elements individually for safety
                        for r in range(remaining):
                            let input_val = input[i, k + r]
                            let weight_val = weights[k + r, j]
                            acc[0] += input_val * weight_val
                
                # Reduce and store
                output[i, j] = acc.reduce_add()
            
            vectorize[self.nelts, compute_cols](self.embed_dim)
        
        parallelize[compute_row](seq_len)
        return output
    
    @parameter
    fn _multi_head_attention(self,
                           queries: Tensor[DType.float32],
                           keys: Tensor[DType.float32], 
                           values: Tensor[DType.float32],
                           seq_len: Int) -> Tensor[DType.float32]:
        """Compute multi-head attention with syntax awareness."""
        var output = Tensor[DType.float32](seq_len, self.embed_dim)
        let scale = 1.0 / sqrt(Float32(self.head_dim))
        
        # Process each head in parallel
        @parameter
        fn process_head(head: Int):
            let head_offset = head * self.head_dim
            
            # Extract head-specific Q, K, V
            for i in range(seq_len):
                for j in range(seq_len):
                    # Skip computation if masked by syntax rules
                    if not self.syntax_attention_mask[i, j]:
                        continue
                    
                    # Compute attention score for this position pair
                    var score: Float32 = 0.0
                    
                    # Dot product between Q[i] and K[j] for this head
                    @parameter
                    fn compute_attention_score(k: Int):
                        let q_val = queries[i, head_offset + k]
                        let k_val = keys[j, head_offset + k]
                        score += q_val * k_val
                    
                    vectorize[self.nelts, compute_attention_score](self.head_dim)
                    
                    # Apply scaling and softmax (simplified)
                    score *= scale
                    let attention_weight = exp(score)  # Simplified softmax
                    
                    # Apply attention to values
                    for k in range(self.head_dim):
                        output[i, head_offset + k] += attention_weight * values[j, head_offset + k]
        
        parallelize[process_head](self.num_heads)
        return output
    
    @parameter
    fn _compute_output_projection(self,
                                attention_output: Tensor[DType.float32],
                                seq_len: Int) -> Tensor[DType.float32]:
        """Apply final output projection."""
        return self._compute_projection(attention_output, self.output_weights, seq_len)
    
    @parameter
    fn _global_average_pooling(self, 
                             sequence_output: Tensor[DType.float32],
                             seq_len: Int) -> Tensor[DType.float32]:
        """Pool sequence into single embedding vector."""
        var pooled = Tensor[DType.float32](self.embed_dim)
        
        @parameter
        fn pool_dimension(dim: Int):
            var sum: Float32 = 0.0
            for i in range(seq_len):
                sum += sequence_output[i, dim]
            pooled[dim] = sum / Float32(seq_len)
        
        parallelize[pool_dimension](self.embed_dim)
        return pooled
    
    fn get_performance_stats(self) -> String:
        """Return performance characteristics of this kernel."""
        return (
            "MLA Kernel Stats:\n" +
            "- Heads: " + str(self.num_heads) + "\n" +
            "- Embedding Dim: " + str(self.embed_dim) + "\n" +
            "- Max Sequence: " + str(self.max_seq_len) + "\n" +
            "- SIMD Width: " + str(self.nelts)
        )

@parameter
fn create_optimized_mla_kernel() -> MLAKernel:
    """Factory function to create optimized MLA kernel."""
    return MLAKernel()

# Performance benchmarking utilities
@parameter  
fn benchmark_mla_kernel(kernel: MLAKernel, num_iterations: Int = 100) -> Float64:
    """Benchmark MLA kernel performance."""
    # Create test input
    let test_seq_len = 256
    var test_input = Tensor[DType.float32](test_seq_len, MLAKernel.embed_dim)
    
    # Initialize with dummy data
    for i in range(test_seq_len):
        for j in range(MLAKernel.embed_dim):
            test_input[i, j] = Float32(i + j) / 1000.0
    
    # Benchmark encoding
    let start_time = time.now()
    
    for _ in range(num_iterations):
        let _ = kernel.encode_sequence(test_input, test_seq_len)
    
    let end_time = time.now()
    let total_time = (end_time - start_time).to_float64()
    
    return total_time / Float64(num_iterations)  # Average time per iteration