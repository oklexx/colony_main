# Train UI Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up old `train_ui` package, launcher, and build/cache trash files, ensuring `train_ui2` and `run_train_ui2.py` remain fully intact and operational.

**Architecture:** Remove legacy `train_ui/` directory and `run_train_ui.py`. Clean up build logs, temporary cache files (`__pycache__`, `.pyc`, debug logs, `.tlog`, `.obj`), and verify `train_ui2`.

**Tech Stack:** Python, CMake / MSBuild build artifacts cleanup.

**Spec:** Project cleanup instructions specified by Model 1 task.

## Global Constraints
- Preserve `train_ui2/` and `run_train_ui2.py` completely.
- Remove old `train_ui/` and `run_train_ui.py`.
- Clean up temporary cache and build logs (`__pycache__`, root debug logs, `.tlog`, `.obj`, `.pyc`).

---

### Task 1: Remove Legacy train_ui and run_train_ui.py

**Files:**
- Delete: `train_ui/` (entire directory and contents)
- Delete: `run_train_ui.py`

- [ ] **Step 1: Delete train_ui directory and run_train_ui.py**

Run: `Remove-Item -Recurse -Force "train_ui", "run_train_ui.py"`
Expected: Successful removal without error.

- [ ] **Step 2: Verify train_ui2 and run_train_ui2.py remain intact**

Run: `Test-Path "train_ui2", "run_train_ui2.py"`
Expected: True / True.

### Task 2: Clean Up Build Logs and Temporary Cache Files

**Files:**
- Delete: `__pycache__` directories, `*.pyc`, root debug logs (`ai_debug_gui.log`, `reward_debug.log`, `eval_*.log`, etc.), and MSBuild/CMake artifacts (`*.tlog`, `*.obj`, build clean logs).

- [ ] **Step 1: Remove __pycache__ and .pyc files**

Run: `Get-ChildItem -Recurse -Include __pycache__, *.pyc | Remove-Item -Recurse -Force`
Expected: All Python cache folders and bytecode files removed.

- [ ] **Step 2: Remove root debug logs and build artifact files/directories**

Run: Remove temporary logs and .tlog/.obj/.dir artifacts in root.
Expected: Clean working directory.
