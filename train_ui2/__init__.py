"""Sakhalin Colony — Training UI 2.0.

Compact, thematically grouped interface. Reuses train_ui/worker.py (same
JSONL protocol) and train_ui/model registry, so it can run IN PARALLEL with
the old UI: each window spawns its own worker process and keeps its own
state file (~/colony_runs/sakhalin_colony_ui2/config.json).
"""
__version__ = "2.0"
