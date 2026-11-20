"""Time the full HTTP round trip on this machine."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sessrec.data import load_dataset  # noqa: E402
from sessrec.paths import LATENCY_FILE, RUNTIME_FILE  # noqa: E402

HOST = "127.0.0.1"
WARMUP = 50
REQUESTS = 1000
LENGTHS = (5, 20, 50, 100)


def free_port() -> int:
    """Prefer 8000. If another program is using it, take the next free port."""

    for port in range(8000, 8020):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((HOST, port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free local port between 8000 and 8019")


def percentile(values: list[float], mark: float) -> float:
    return float(np.percentile(np.asarray(values), mark))


def hardware() -> dict:
    return {
        "cpu": platform.processor() or platform.machine(),
        "cores": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }


def wait_until_ready(process: subprocess.Popen, port: int, timeout: float = 180) -> None:
    deadline = time.perf_counter() + timeout
    last_error = ""
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited early: {last_error}")
        try:
            response = httpx.get(f"http://{HOST}:{port}/health", timeout=2)
            if response.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_error = str(exc)
        time.sleep(0.3)
    raise TimeoutError("server did not become ready")


def time_requests(client: httpx.Client, port: int, history: list[int], count: int) -> list[float]:
    payload = {"history": history, "k": 5}
    samples = []
    for _ in range(count):
        started = time.perf_counter()
        response = client.post(f"http://{HOST}:{port}/recommend", json=payload)
        response.raise_for_status()
        samples.append((time.perf_counter() - started) * 1000)
    return samples


def main() -> None:
    dataset = load_dataset()
    long_history = dataset.test_histories()[0][-100:]
    raw_history = [dataset.index_to_raw[item] for item in long_history]
    port = free_port()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "sessrec.serve.app:app",
            "--host",
            HOST,
            "--port",
            str(port),
            "--workers",
            "1",
        ],
        cwd=ROOT,
        env=env,
    )
    try:
        wait_until_ready(process, port)
        by_length = {}
        with httpx.Client(timeout=30) as client:
            for length in LENGTHS:
                history = raw_history[-length:]
                time_requests(client, port, history, WARMUP)
                samples = time_requests(client, port, history, REQUESTS)
                by_length[str(length)] = {
                    "p50": percentile(samples, 50),
                    "p95": percentile(samples, 95),
                    "p99": percentile(samples, 99),
                }
                print(length, by_length[str(length)], flush=True)
        runtime = {}
        if RUNTIME_FILE.exists():
            runtime = json.loads(RUNTIME_FILE.read_text(encoding="utf-8"))
        report = {
            "hardware": hardware(),
            "port": port,
            "warmup": WARMUP,
            "requests_per_length": REQUESTS,
            "gate_ms": 50,
            "gate_met": all(item["p95"] < 50 for item in by_length.values()),
            "torch_threads": runtime.get("torch_threads"),
            "by_history_length": by_length,
            "note": "p95 means 95% of requests finished in this time or less. This is localhost on a catalog of a few thousand films.",
        }
        LATENCY_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    main()
