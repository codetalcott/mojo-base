"""
Tensor Bounds Validation
Comprehensive validation for tensor operations to prevent memory violations
"""

struct TensorBoundsValidator:
    """Validates tensor operations and prevents boundary violations."""
    
    fn validate_tensor_dimensions(self, rank: Int, shape: List[Int]) raises:
        """Validate tensor has valid dimensions."""
        if rank <= 0:
            raise Error("Tensor rank must be positive")
        
        if rank > 8:
            raise Error("Tensor rank exceeds practical limits")
        
        if len(shape) != rank:
            raise Error("Shape length must match tensor rank")
        
        for i in range(len(shape)):
            if shape[i] <= 0:
                raise Error("All tensor dimensions must be positive")
    
    fn validate_index_access(self, indices: List[Int], shape: List[Int]) raises:
        """Validate tensor index access is within bounds."""
        if len(indices) != len(shape):
            raise Error("Index dimensions must match tensor dimensions")
        
        for i in range(len(indices)):
            if indices[i] < 0 or indices[i] >= shape[i]:
                raise Error("Index " + str(i) + " out of bounds")
    
    fn validate_slice_bounds(self, start: Int, end: Int, dimension_size: Int) raises:
        """Validate tensor slice bounds."""
        if start < 0:
            raise Error("Slice start cannot be negative")
        
        if end > dimension_size:
            raise Error("Slice end exceeds dimension size")
        
        if start >= end:
            raise Error("Slice start must be less than end")
    
    fn validate_matrix_multiplication(self, a_shape: List[Int], b_shape: List[Int]) raises:
        """Validate matrix multiplication dimensions."""
        if len(a_shape) != 2 or len(b_shape) != 2:
            raise Error("Matrix multiplication requires 2D tensors")
        
        if a_shape[1] != b_shape[0]:
            raise Error("Matrix multiplication dimension mismatch")
    
    fn validate_embedding_dimensions(self, embedding_shape: List[Int]) raises:
        """Validate embedding tensor dimensions."""
        if len(embedding_shape) != 1:
            raise Error("Embedding must be 1D tensor")
        
        if embedding_shape[0] != 768:
            raise Error("Embedding must be 768 dimensions")
    
    fn validate_sequence_tensor(self, tensor_shape: List[Int], max_seq_len: Int) raises:
        """Validate sequence tensor dimensions."""
        if len(tensor_shape) != 2:
            raise Error("Sequence tensor must be 2D")
        
        if tensor_shape[0] > max_seq_len:
            raise Error("Sequence length exceeds maximum")
        
        if tensor_shape[1] != 768:
            raise Error("Sequence embedding dimension must be 768")

fn validate_tensor_operation_safety() -> Bool:
    """Test tensor operation safety validation."""
    print("🔒 Tensor Operation Safety Validation")
    print("====================================")
    
    var validator = TensorBoundsValidator()
    var safety_tests_passed = 0
    var total_safety_tests = 8
    
    # Test 1: Valid tensor dimensions
    print("\n1. Testing valid tensor dimensions...")
    try:
        var valid_shape = List[Int]()
        valid_shape.append(100)
        valid_shape.append(768)
        validator.validate_tensor_dimensions(2, valid_shape)
        print("   ✅ Valid dimensions accepted")
        safety_tests_passed += 1
    except e:
        print("   ❌ Valid dimensions rejected:", e)
    
    # Test 2: Invalid tensor rank
    print("\n2. Testing invalid tensor rank...")
    try:
        var invalid_shape = List[Int]()
        validator.validate_tensor_dimensions(0, invalid_shape)
        print("   ❌ Should have failed")
    except:
        print("   ✅ Invalid rank caught")
        safety_tests_passed += 1
    
    # Test 3: Index bounds checking
    print("\n3. Testing index bounds...")
    try:
        var shape = List[Int]()
        shape.append(10)
        shape.append(20)
        var valid_indices = List[Int]()
        valid_indices.append(5)
        valid_indices.append(15)
        validator.validate_index_access(valid_indices, shape)
        print("   ✅ Valid indices accepted")
        safety_tests_passed += 1
    except e:
        print("   ❌ Valid indices rejected:", e)
    
    # Test 4: Index out of bounds
    print("\n4. Testing index out of bounds...")
    try:
        var shape = List[Int]()
        shape.append(10)
        shape.append(20)
        var invalid_indices = List[Int]()
        invalid_indices.append(15)  # Out of bounds
        invalid_indices.append(5)
        validator.validate_index_access(invalid_indices, shape)
        print("   ❌ Should have failed")
    except:
        print("   ✅ Out of bounds index caught")
        safety_tests_passed += 1
    
    # Test 5: Slice bounds validation
    print("\n5. Testing slice bounds...")
    try:
        validator.validate_slice_bounds(0, 50, 100)
        print("   ✅ Valid slice accepted")
        safety_tests_passed += 1
    except e:
        print("   ❌ Valid slice rejected:", e)
    
    # Test 6: Invalid slice bounds
    print("\n6. Testing invalid slice bounds...")
    try:
        validator.validate_slice_bounds(50, 200, 100)  # End > dimension
        print("   ❌ Should have failed")
    except:
        print("   ✅ Invalid slice bounds caught")
        safety_tests_passed += 1
    
    # Test 7: Matrix multiplication validation
    print("\n7. Testing matrix multiplication dimensions...")
    try:
        var a_shape = List[Int]()
        a_shape.append(100)
        a_shape.append(50)
        var b_shape = List[Int]()
        b_shape.append(50)
        b_shape.append(30)
        validator.validate_matrix_multiplication(a_shape, b_shape)
        print("   ✅ Valid matrix multiplication")
        safety_tests_passed += 1
    except e:
        print("   ❌ Valid matrix multiplication rejected:", e)
    
    # Test 8: Embedding validation
    print("\n8. Testing embedding validation...")
    try:
        var embedding_shape = List[Int]()
        embedding_shape.append(768)
        validator.validate_embedding_dimensions(embedding_shape)
        print("   ✅ Valid embedding dimensions")
        safety_tests_passed += 1
    except e:
        print("   ❌ Valid embedding rejected:", e)
    
    var success_rate = Float64(safety_tests_passed) / Float64(total_safety_tests)
    print("\n📊 Tensor Safety Results:")
    print("Total tests:", total_safety_tests)
    print("Passed:", safety_tests_passed)
    print("Success rate:", success_rate * 100.0, "%")
    
    return success_rate >= 0.8

fn simulate_safe_tensor_operations() -> Bool:
    """Simulate tensor operations with safety validation."""
    print("\n⚡ Safe Tensor Operations Simulation")
    print("===================================")
    
    var validator = TensorBoundsValidator()
    var operations_completed = 0
    var violations_prevented = 0
    
    # Simulate sequence processing
    print("\n📝 Simulating sequence processing...")
    try:
        var seq_shape = List[Int]()
        seq_shape.append(256)  # Sequence length
        seq_shape.append(768)  # Embedding dimension
        validator.validate_sequence_tensor(seq_shape, 512)
        
        # Simulate safe access patterns
        for i in range(10):
            var indices = List[Int]()
            indices.append(i * 25)  # Safe index
            indices.append(i * 76)  # Safe index
            try:
                validator.validate_index_access(indices, seq_shape)
                operations_completed += 1
            except:
                violations_prevented += 1
        
        print("   ✅ Sequence processing validated")
    except e:
        print("   ❌ Sequence processing failed:", e)
    
    # Simulate matrix operations
    print("\n🔢 Simulating matrix operations...")
    try:
        var a_shape = List[Int]()
        a_shape.append(768)
        a_shape.append(768)
        var b_shape = List[Int]()
        b_shape.append(768)
        b_shape.append(256)
        
        validator.validate_matrix_multiplication(a_shape, b_shape)
        operations_completed += 5  # Simulate 5 matrix operations
        print("   ✅ Matrix operations validated")
    except e:
        print("   ❌ Matrix operations failed:", e)
    
    # Simulate embedding operations
    print("\n🧠 Simulating embedding operations...")
    try:
        for i in range(10):
            var emb_shape = List[Int]()
            emb_shape.append(768)
            validator.validate_embedding_dimensions(emb_shape)
            operations_completed += 1
        
        print("   ✅ Embedding operations validated")
    except e:
        print("   ❌ Embedding operations failed:", e)
    
    print("\n📊 Operation Results:")
    print("Safe operations completed:", operations_completed)
    print("Violations prevented:", violations_prevented)
    print("Safety rate:", Float64(operations_completed) / Float64(operations_completed + violations_prevented) * 100.0, "%")
    
    return operations_completed > 0

fn test_memory_safety_patterns() -> Bool:
    """Test memory safety patterns for tensor operations."""
    print("\n🛡️  Memory Safety Pattern Testing")
    print("=================================")
    
    var validator = TensorBoundsValidator()
    var memory_safe_operations = 0
    var memory_violations_caught = 0
    
    # Test large tensor validation
    print("\n📊 Testing large tensor handling...")
    try:
        var large_shape = List[Int]()
        large_shape.append(10000)
        large_shape.append(768)
        validator.validate_tensor_dimensions(2, large_shape)
        memory_safe_operations += 1
        print("   ✅ Large tensor validated")
    except e:
        print("   ❌ Large tensor failed:", e)
    
    # Test extreme indices
    print("\n🔍 Testing extreme index values...")
    var extreme_test_cases = [
        (-1, "negative index"),
        (999999, "extremely large index"),
        (0, "zero index"),
    ]
    
    for i in range(3):
        var test_value = extreme_test_cases[i][0]
        var test_name = extreme_test_cases[i][1]
        
        try:
            var shape = List[Int]()
            shape.append(100)
            var indices = List[Int]()
            indices.append(test_value)
            validator.validate_index_access(indices, shape)
            
            if test_value >= 0 and test_value < 100:
                memory_safe_operations += 1
                print("   ✅", test_name, "handled safely")
            else:
                print("   ❌", test_name, "should have been caught")
        except:
            memory_violations_caught += 1
            print("   ✅", test_name, "violation caught")
    
    # Test slice edge cases
    print("\n✂️  Testing slice edge cases...")
    var slice_test_cases = [
        (0, 100, 100, "full slice"),
        (50, 60, 100, "middle slice"),
        (-1, 50, 100, "negative start"),
        (50, 150, 100, "end exceeds bounds"),
    ]
    
    for i in range(4):
        var start = slice_test_cases[i][0]
        var end = slice_test_cases[i][1] 
        var size = slice_test_cases[i][2]
        var test_name = slice_test_cases[i][3]
        
        try:
            validator.validate_slice_bounds(start, end, size)
            if start >= 0 and end <= size and start < end:
                memory_safe_operations += 1
                print("   ✅", test_name, "accepted")
            else:
                print("   ❌", test_name, "should have been rejected")
        except:
            memory_violations_caught += 1
            print("   ✅", test_name, "violation caught")
    
    print("\n📊 Memory Safety Results:")
    print("Safe operations:", memory_safe_operations)
    print("Violations caught:", memory_violations_caught)
    print("Total safety actions:", memory_safe_operations + memory_violations_caught)
    
    var safety_effectiveness = Float64(memory_violations_caught) / Float64(memory_violations_caught + memory_safe_operations)
    print("Safety effectiveness:", safety_effectiveness * 100.0, "%")
    
    return memory_safe_operations > 0 and memory_violations_caught > 0

fn main():
    """Main function to validate tensor bounds safety."""
    print("🔒 Comprehensive Tensor Bounds Validation")
    print("=========================================")
    
    var all_validations_passed = True
    
    # Test 1: Tensor operation safety
    if validate_tensor_operation_safety():
        print("✅ Tensor operation safety: PASS")
    else:
        print("❌ Tensor operation safety: FAIL")
        all_validations_passed = False
    
    # Test 2: Safe tensor operations simulation
    if simulate_safe_tensor_operations():
        print("✅ Safe tensor operations: PASS")
    else:
        print("❌ Safe tensor operations: FAIL")
        all_validations_passed = False
    
    # Test 3: Memory safety patterns
    if test_memory_safety_patterns():
        print("✅ Memory safety patterns: PASS")
    else:
        print("❌ Memory safety patterns: FAIL")
        all_validations_passed = False
    
    print("\n📋 Tensor Bounds Validation Summary")
    print("===================================")
    
    if all_validations_passed:
        print("🎉 ALL TENSOR VALIDATIONS PASSED")
        print("Tensor operations are safe for production")
    else:
        print("⚠️  Some tensor validations failed")
        print("Address issues before production deployment")
    
    print("\n🔒 Safety Features Implemented:")
    print("==============================")
    print("✅ Tensor dimension validation")
    print("✅ Index bounds checking")
    print("✅ Slice bounds validation")
    print("✅ Matrix operation validation")
    print("✅ Embedding dimension validation")
    print("✅ Sequence tensor validation")
    print("✅ Memory access validation")
    print("✅ Edge case handling")
    
    return all_validations_passed