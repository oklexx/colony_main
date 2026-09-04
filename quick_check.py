# -*- coding: utf-8 -*-
"""Быстрая проверка UI перед запуском."""

import sys
from pathlib import Path

print("=" * 60)
print("UI Improvements - Quick Pre-flight Check")
print("=" * 60)

checks = []

# 1. Check imports
try:
    from train_ui.protocol import ProgressMsg, CommandMsg
    checks.append(("Protocol", True, "OK"))
except Exception as e:
    checks.append(("Protocol", False, str(e)[:50]))

try:
    from train_ui.kl_status_widget import KLStatusWidget
    widget = KLStatusWidget()
    checks.append(("KL Status Widget", True, "Instantiated OK"))
except Exception as e:
    checks.append(("KL Status Widget", False, str(e)[:50]))

try:
    from train_ui.curriculum_progress_widget import CurriculumProgressWidget
    widget = CurriculumProgressWidget()
    checks.append(("Curriculum Progress Widget", True, "Instantiated OK"))
except Exception as e:
    checks.append(("Curriculum Progress Widget", False, str(e)[:50]))

try:
    from train_ui.quick_actions_widget import QuickActionsWidget
    widget = QuickActionsWidget()
    checks.append(("Quick Actions Widget", True, "Instantiated OK"))
except Exception as e:
    checks.append(("Quick Actions Widget", False, str(e)[:50]))

# 2. Check MainWindow exists
try:
    from train_ui.main_window import MainWindow
    checks.append(("MainWindow", True, "Exists"))
except Exception as e:
    checks.append(("MainWindow", False, str(e)[:50]))

# Print results
print("\nCheck Results:")
print("-" * 60)
all_ok = True
for name, ok, msg in checks:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: {msg}")
    if not ok:
        all_ok = False

print("-" * 60)
if all_ok:
    print("\nAll checks passed!")
    print("\nYou can now run:")
    print("  python run_train_ui.py")
else:
    print("\nSome checks failed. See errors above.")

print("=" * 60)
