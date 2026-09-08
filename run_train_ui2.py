#!/usr/bin/env python3
"""Launch Training UI 2.0 (compact themed UI).

Safe to run in parallel with run_train_ui.py: UI 2.0 keeps its own state in
~/colony_runs/sakhalin_colony_ui2/config.json and uses its own temp files.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_ui2.app import main

if __name__ == "__main__":
    sys.exit(main())
