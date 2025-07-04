#!/usr/bin/env python3
"""
Real-time Performance Dashboard
Live monitoring and visualization for presentation
"""

import asyncio
import time
import sys
import requests
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import statistics


@dataclass
class MetricPoint:
    """Single performance measurement."""

    timestamp: datetime
    latency_ms: float
    results_count: int
    success: bool
    query: str


class PerformanceDashboard:
    """Real-time performance monitoring dashboard."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.metrics: List[MetricPoint] = []
        self.max_history = 100  # Keep last 100 measurements
        self.monitoring = False

        self.test_queries = [
            "authentication patterns",
            "React components",
            "async functions",
            "error handling",
            "API endpoints",
            "database queries",
            "TypeScript interfaces",
            "middleware functions",
        ]

    def clear_screen(self):
        """Clear terminal screen."""
        print("\033[2J\033[H", end="")

    def format_latency(self, ms: float) -> str:
        """Format latency with color coding."""
        if ms < 10:
            return f"\033[92m{ms:6.1f}ms\033[0m"  # Green
        elif ms < 25:
            return f"\033[93m{ms:6.1f}ms\033[0m"  # Yellow
        else:
            return f"\033[91m{ms:6.1f}ms\033[0m"  # Red

    def format_success_rate(self, rate: float) -> str:
        """Format success rate with color coding."""
        if rate >= 0.95:
            return f"\033[92m{rate:.1%}\033[0m"  # Green
        elif rate >= 0.85:
            return f"\033[93m{rate:.1%}\033[0m"  # Yellow
        else:
            return f"\033[91m{rate:.1%}\033[0m"  # Red

    def create_histogram(self, values: List[float], width: int = 40) -> str:
        """Create ASCII histogram."""
        if not values:
            return "No data"

        min_val = min(values)
        max_val = max(values)
        if min_val == max_val:
            return "▆" * width

        # Create buckets
        bucket_count = min(width, len(values))
        bucket_size = (max_val - min_val) / bucket_count
        buckets = [0] * bucket_count

        for value in values:
            bucket_idx = min(bucket_count - 1, int((value - min_val) / bucket_size))
            buckets[bucket_idx] += 1

        # Normalize to display height
        max_count = max(buckets) if buckets else 1
        histogram = ""
        for count in buckets:
            height = int((count / max_count) * 8)
            if height == 0 and count > 0:
                height = 1
            histogram += "▁▂▃▄▅▆▇█"[height - 1] if height > 0 else " "

        return histogram

    async def measure_performance(self) -> Optional[MetricPoint]:
        """Measure single search performance."""
        query = self.test_queries[len(self.metrics) % len(self.test_queries)]
        start_time = time.time()

        try:
            response = requests.post(
                f"{self.api_url}/search",
                json={"query": query, "max_results": 5, "include_mcp": True, "use_cache": True},
                timeout=5.0,
            )

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])

                return MetricPoint(
                    timestamp=datetime.now(),
                    latency_ms=latency_ms,
                    results_count=len(results),
                    success=True,
                    query=query,
                )
            else:
                return MetricPoint(
                    timestamp=datetime.now(),
                    latency_ms=latency_ms,
                    results_count=0,
                    success=False,
                    query=query,
                )

        except requests.RequestException:
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            return MetricPoint(
                timestamp=datetime.now(),
                latency_ms=latency_ms,
                results_count=0,
                success=False,
                query=query,
            )

    def calculate_stats(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Calculate performance statistics for time window."""
        if not self.metrics:
            return {}

        # Filter to time window
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]

        if not recent_metrics:
            recent_metrics = self.metrics[-10:]  # Fallback to last 10

        successful_metrics = [m for m in recent_metrics if m.success]

        if not successful_metrics:
            return {
                "window_minutes": window_minutes,
                "total_requests": len(recent_metrics),
                "success_rate": 0,
                "error": "No successful requests",
            }

        latencies = [m.latency_ms for m in successful_metrics]
        result_counts = [m.results_count for m in successful_metrics]

        return {
            "window_minutes": window_minutes,
            "total_requests": len(recent_metrics),
            "successful_requests": len(successful_metrics),
            "success_rate": len(successful_metrics) / len(recent_metrics),
            "latency": {
                "avg": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "p95": (
                    sorted(latencies)[int(len(latencies) * 0.95)]
                    if len(latencies) > 1
                    else latencies[0]
                ),
            },
            "results": {
                "avg_count": statistics.mean(result_counts),
                "total_results": sum(result_counts),
            },
            "requests_per_minute": (
                len(recent_metrics) / window_minutes if window_minutes > 0 else 0
            ),
        }

    def render_dashboard(self):
        """Render the performance dashboard."""
        self.clear_screen()

        print("🚀 MOJO SEMANTIC SEARCH - PERFORMANCE DASHBOARD")
        print("=" * 70)
        print(
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Monitoring: {'🟢 ON' if self.monitoring else '🔴 OFF'}"
        )
        print()

        if not self.metrics:
            print("📊 No data collected yet...")
            print("   Starting performance monitoring...")
            return

        # Current status
        last_metric = self.metrics[-1]
        status_icon = "✅" if last_metric.success else "❌"
        print(f"📊 Current Status: {status_icon}")
        print(f"   Last Query: {last_metric.query[:50]}")
        print(f"   Latency: {self.format_latency(last_metric.latency_ms)}")
        print(f"   Results: {last_metric.results_count}")
        print()

        # Performance statistics
        stats_1min = self.calculate_stats(1)
        stats_5min = self.calculate_stats(5)

        if "error" not in stats_1min and "error" not in stats_5min:
            print("📈 Performance Metrics:")
            print(f"                    1 Minute    5 Minutes")
            print(
                f"   Requests:        {stats_1min['total_requests']:8d}    {stats_5min['total_requests']:8d}"
            )
            print(
                f"   Success Rate:    {self.format_success_rate(stats_1min['success_rate']):>15} {self.format_success_rate(stats_5min['success_rate']):>15}"
            )
            print(
                f"   Avg Latency:     {stats_1min['latency']['avg']:8.1f}ms   {stats_5min['latency']['avg']:8.1f}ms"
            )
            print(
                f"   P95 Latency:     {stats_1min['latency']['p95']:8.1f}ms   {stats_5min['latency']['p95']:8.1f}ms"
            )
            print(
                f"   Req/Min:         {stats_1min['requests_per_minute']:8.1f}     {stats_5min['requests_per_minute']:8.1f}"
            )
            print()

            # Target achievement
            avg_latency = stats_5min["latency"]["avg"]
            success_rate = stats_5min["success_rate"]

            print("🎯 Target Achievement:")
            latency_status = (
                "✅" if avg_latency < 20 else "⚠️" if avg_latency < 50 else "❌"
            )
            success_status = (
                "✅" if success_rate >= 0.95 else "⚠️" if success_rate >= 0.85 else "❌"
            )

            print(f"   Sub-20ms Latency: {latency_status} {avg_latency:.1f}ms")
            print(f"   95%+ Success:     {success_status} {success_rate:.1%}")
            print()

            # Latency histogram
            recent_latencies = [m.latency_ms for m in self.metrics[-20:] if m.success]
            if recent_latencies:
                print(f"📊 Latency Distribution (last 20 requests):")
                histogram = self.create_histogram(recent_latencies, 50)
                print(
                    f"   {min(recent_latencies):4.0f}ms ▕{histogram}▏ {max(recent_latencies):4.0f}ms"
                )
                print()

        # Recent activity log
        print("📋 Recent Activity (last 10):")
        recent_metrics = self.metrics[-10:]
        for i, metric in enumerate(reversed(recent_metrics), 1):
            status = "✅" if metric.success else "❌"
            timestamp = metric.timestamp.strftime("%H:%M:%S")
            latency_str = self.format_latency(metric.latency_ms)

            print(
                f"   {i:2d}. {timestamp} {status} {latency_str} | "
                f"{metric.results_count:2d} results | {metric.query[:30]:30}"
            )

        print()
        print("🔧 System Status:")

        # Check system health
        try:
            response = requests.get(f"{self.api_url}/health", timeout=1)
            if response.status_code == 200:
                print("   ✅ API Server: Healthy")
            else:
                print(f"   ⚠️  API Server: HTTP {response.status_code}")
        except:
            print("   ❌ API Server: Unreachable")

        try:
            response = requests.get("http://localhost:8080", timeout=1)
            if response.status_code == 200:
                print("   ✅ Web Interface: Running")
            else:
                print("   ⚠️  Web Interface: Error")
        except:
            print("   ❌ Web Interface: Down")

        print()
        print("💡 Demo Highlights:")
        if stats_5min and "error" not in stats_5min:
            print(f"   🚀 Average latency: {stats_5min['latency']['avg']:.1f}ms")
            print(f"   📊 Success rate: {stats_5min['success_rate']:.1%}")
            print(f"   🎯 Total queries: {len(self.metrics)}")
        # Get dynamic corpus info from API
        try:
            api_info = requests.get(f"{self.api_url}/", timeout=1).json()
            corpus_size = api_info.get('corpus_size', 'Unknown')
            projects = api_info.get('source_projects', 'Unknown')
            print(f"   📚 Corpus size: {corpus_size} vectors")
            print(f"   🧠 Projects: {projects}")
        except:
            print("   📚 Corpus: Dynamic size")
            print("   🧠 Projects: Auto-detected")
        print("   🔗 Optimized MCP integration")

        print()
        print("Press Ctrl+C to stop monitoring")

    async def start_monitoring(self, refresh_interval: float = 2.0):
        """Start real-time performance monitoring."""
        self.monitoring = True

        try:
            while self.monitoring:
                # Take measurement
                metric = await self.measure_performance()
                if metric:
                    self.metrics.append(metric)

                    # Trim history
                    if len(self.metrics) > self.max_history:
                        self.metrics = self.metrics[-self.max_history :]

                # Render dashboard
                self.render_dashboard()

                # Wait for next update
                await asyncio.sleep(refresh_interval)

        except KeyboardInterrupt:
            self.monitoring = False
            print("\n🛑 Monitoring stopped")

            # Final summary
            if self.metrics:
                final_stats = self.calculate_stats(
                    max(1, int(len(self.metrics) / 10))
                )  # Assume ~10 req/min
                if "error" not in final_stats:
                    print(f"\n📊 Final Summary:")
                    print(f"   Total requests: {len(self.metrics)}")
                    print(f"   Average latency: {final_stats['latency']['avg']:.1f}ms")
                    print(f"   Success rate: {final_stats['success_rate']:.1%}")
                    print(f"   P95 latency: {final_stats['latency']['p95']:.1f}ms")


async def main():
    """Main dashboard function."""
    dashboard = PerformanceDashboard()

    print("🎯 Starting Performance Dashboard...")
    print("   This will continuously monitor search performance")
    print("   Perfect for live demo during presentation")
    print()

    # Check if API is available
    try:
        response = requests.get(f"{dashboard.api_url}/health", timeout=3)
        if response.status_code == 200:
            print("✅ API server is running - starting dashboard...")
            await asyncio.sleep(1)
        else:
            print("❌ API server error - dashboard may not work properly")
    except:
        print("❌ API server not available")
        print("   Please start: python3 api/semantic_search_api_v2.py")
        print("   Continue anyway? (y/n): ", end="")
        if input().lower() != "y":
            return

    await dashboard.start_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
