import subprocess, sys, time, os

PROJECT = r"C:\Users\oklex\OneDrive\Documentos\sakhalin_colony_main"
PYTHON = sys.executable

args = [
    PYTHON, "-u",  # unbuffered!
    os.path.join(PROJECT, "train.py"),
    "--name", "exp_25_minimap",
    "--steps", "10000000",
    "--envs", "32",
    "--compile",
    "--n-steps", "2048",
    "--batch-size", "32768",
    "--n-epochs", "10",
    "--lr", "3e-4",
    "--ent-coef", "0.01",
    "--vf-coef", "1.0",
    "--max-grad-norm", "1.0",
    "--gamma", "0.99",
    "--gae-lambda", "0.95",
    "--clip-range", "0.2",
    "--net-arch", "512",
    "--obs-mode", "minimap",
    "--seed", "42",
    "--eval-episodes", "10",
    "--amp", "bfloat16",
    "--log-actions",
]

print(f"Starting: {' '.join(args[-15:])}")
proc = subprocess.Popen(args, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

while True:
    line = proc.stdout.readline()
    if not line and proc.poll() is not None:
        break
    if line:
        print(line, end="", flush=True)

print(f"\nExit code: {proc.returncode}")
