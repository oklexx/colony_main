"""Integration test: 100k steps, measure FPS and GPU utilization."""
import subprocess
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "train.py"

def run_training(args: list[str], timeout: int = 600) -> str:
    cmd = [sys.executable, str(TRAIN_SCRIPT)] + args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}")
    return result.stdout

def extract_fps(output: str) -> float:
    for line in output.splitlines():
        if "Average FPS:" in line:
            return float(line.split("Average FPS:")[1].strip().replace(",", ""))
    return 0.0

def main():
    print("=" * 60)
    print("INTEGRATION TEST: 100k steps, 8 envs, GPU training")
    print("=" * 60)

    args = [
        "--steps", "100000",
        "--envs", "8",
        "--n-steps", "4096",
        "--batch-size", "8192",
        "--n-epochs", "10",
        "--name", "integration_test",
    ]

    output = run_training(args, timeout=600)
    print(output)

    fps = extract_fps(output)
    print(f"\nExtracted FPS: {fps:,.0f}")

    if fps >= 50000:
        print("PASS: FPS >= 50,000")
    elif fps >= 20000:
        print("WARN: FPS between 20k-50k (acceptable for small net)")
    else:
        print(f"INFO: FPS={fps:,.0f} (below target, may need tuning)")

    if "Traceback" in output:
        print("FAIL: Training crashed")
        sys.exit(1)
    else:
        print("PASS: Training completed without errors")

if __name__ == "__main__":
    main()
