#!/usr/bin/env python3
"""
Check MAX modules for execution patterns
"""

import max

print("=== MAX Top Level Modules ===")
max_modules = [x for x in dir(max) if not x.startswith('_')]
for module in sorted(max_modules):
    print(f"  - {module}")

print()

# Check if there's an engine or execution module
execution_modules = [x for x in max_modules if any(keyword in x.lower() for keyword in ['engine', 'exec', 'session', 'infer', 'run'])]
print("Execution related modules:")
for module in execution_modules:
    print(f"  - {module}")

# Check if graph can be called directly
print()
print("Checking if Graph has call methods:")
import max.graph as g

# Create a simple test graph
dtype = max.dtype.DType.float32
device = g.DeviceRef.CPU()
tensor_type = g.TensorType(dtype, [2, 2], device)

def simple_forward(x):
    return g.ops.add(x, x)

test_graph = g.Graph(
    name="test_call",
    forward=simple_forward,
    input_types=[tensor_type]
)

print(f"Graph type: {type(test_graph)}")
print(f"Graph callable: {callable(test_graph)}")

# Check if graph has __call__ method
if hasattr(test_graph, '__call__'):
    print("Graph has __call__ method")
else:
    print("Graph does not have __call__ method")

# Look for execution-related methods on the graph
graph_methods = [x for x in dir(test_graph) if not x.startswith('_')]
execution_methods = [x for x in graph_methods if any(keyword in x.lower() for keyword in ['exec', 'run', 'call', 'infer', 'compile'])]
print(f"Execution methods on graph: {execution_methods}")