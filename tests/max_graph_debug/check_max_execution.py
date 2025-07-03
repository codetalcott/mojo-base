#!/usr/bin/env python3
"""
Check MAX Graph execution methods
"""

import max.graph as g

print("=== MAX Graph Execution Methods ===")
print()

print("Graph methods:")
graph_methods = [x for x in dir(g.Graph) if not x.startswith('_')]
for method in sorted(graph_methods):
    print(f"  - {method}")

print()

print("Looking for execution/session related:")
all_attrs = [x for x in dir(g) if not x.startswith('_')]
execution_attrs = [x for x in all_attrs if any(keyword in x.lower() for keyword in ['session', 'infer', 'exec', 'run', 'compile', 'engine'])]
for attr in sorted(execution_attrs):
    print(f"  - {attr}")

print()

print("All top-level max.graph attributes:")
for attr in sorted(all_attrs):
    print(f"  - {attr}")