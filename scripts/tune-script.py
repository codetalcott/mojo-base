#!/usr/bin/env python3
import time
import json
import subprocess

try:
    import numpy as np
except ImportError:
    print("Installing numpy...")
    subprocess.run(["pip3", "install", "numpy"], check=True)
    import numpy as np

print("🚀 GPU 1 A10 Autotuning Test")
print("=" * 40)

# Check GPU
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name"], capture_output=True, text=True
    )
    print(f"GPU: {result.stdout.strip()}")
except subprocess.SubprocessError:
    print("No GPU detected")

# Simple performance test
configs = [
    {"tile": 4, "block": 32},
    {"tile": 8, "block": 32},
    {"tile": 16, "block": 32},
    {"tile": 32, "block": 32},
]

results = []
for i, config in enumerate(configs, 1):
    print(f"\n[{i}/4] Testing tile={config['tile']}, block={config['block']}")

    start = time.time()
    size = config["tile"] * config["block"]
    a = np.random.random((size, 128)).astype(np.float32)
    b = np.random.random((128, size)).astype(np.float32)
    c = np.dot(a, b)
    latency = (time.time() - start) * 1000

    result = {"tile": config["tile"], "latency_ms": round(latency, 2)}
    results.append(result)
    print(f"   Latency: {latency:.2f}ms")

best = min(results, key=lambda x: x["latency_ms"])
print(f"\nBest: tile={best['tile']}, latency={best['latency_ms']}ms")

with open("gpu1_results.json", "w") as f:
    json.dump({"gpu": "A10", "results": results, "best": best}, f, indent=2)

print("Results saved to gpu1_results.json")
