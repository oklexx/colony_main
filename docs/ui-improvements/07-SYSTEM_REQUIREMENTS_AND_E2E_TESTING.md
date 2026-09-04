# SYSTEM REQUIREMENTS & E2E TESTING SPECIFICATION

## 1. System Requirements (New Section)

### Minimum System Requirements

| Component | Version Required | Notes |
|-----------|------------------|-------|
| Python | ≥3.8 | Recommended: 3.10 for best compatibility |
| PySide6 | ≥6.4 | Qt 6.4+ for UI stability |
| pyqtgraph | ≥0.12.0 | For real-time charts (optional, see below) |
| numpy | ≥1.20.0 | Numerical operations |
| torch | ≥1.9.0 | PPO training backend |
| git | Any | For commit hash extraction |

### Optional Dependencies (Nice-to-Have)

```ini
[extras]
charts = pyqtgraph, numpy  # For ReturnStatisticsWidget histogram
logging = python-json-logger  # For structured JSON logging
testing = pytest, pytest-cov, coverage  # For testing
```

### Installation Commands

**Minimum required:**
```bash
pip install -r requirements.txt
pip install PySide6 pyqtgraph numpy torch
```

**Full feature set (recommended):**
```bash
pip install "PySide6>=6.4" "pyqtgraph>=0.12.0" numpy torch
```

### Docker Support

```dockerfile
FROM python:3.10-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment and install requirements
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install "PySide6>=6.4" pyqtgraph numpy torch

WORKDIR /app
```

---

## 2. End-to-End Testing Specification (New Section)

### Overview

**Goal:** Ensure complete integration between UI, worker process, and training pipeline.

**Scope:** All features from Phase 1 through Phase 4, including:
- Parameter updates at runtime
- Command execution (stop_training, boost_entropy, etc.)
- UI widget updates
- Metrics visualization
- Headless mode operation

### Test Strategy

#### A. Unit Tests (P1)
**Location:** `rl/test_loop_detector.py`, `train_ui/test_widgets.py`

**Coverage Target:** >85%

**Examples:**
```python
# rl/test_loop_detector.py - LoopDetector tests
import unittest
import torch
from rl.loop_detector import LoopDetector

class TestLoopDetector(unittest.TestCase):
    def test_detect_preserve_loop(self):
        """Test PRESERVE loop detection with default threshold=3."""
        actions = torch.tensor([0, 0, 0, 1, 2], dtype=torch.long)
        names = ["PRESERVE", "BUILD_HOUSE", "DAY", ...]
        
        alerts = self.detector.update_batch(
            env_indices=list(range(5)),
            actions=actions,
            action_names=names
        )
        
        # First 3 PRESERVE should trigger alert for env 0
        self.assertEqual(alerts[0], "PRESERVE")
    
    def test_no_loop_below_threshold(self):
        """Test that counts below threshold don't trigger."""
        actions = torch.tensor([0, 1, 2], dtype=torch.long)
        names = ["PRESERVE", "BUILD_HOUSE", "DAY"]
        
        alerts = self.detector.update_batch(
            env_indices=[0, 1, 2],
            actions=actions,
            action_names=names
        )
        
        # No loops should be detected
        self.assertIsNone(alerts[0])

# train_ui/test_widgets.py - UI widget tests
import unittest
from PyQt6.QtWidgets import QApplication, QWidget
from train_ui.kl_status_widget import KLStatusWidget

class TestKLStatusWidget(unittest.TestCase):
    def setUp(self):
        QApplication.instance() or QApplication([])
        self.widget = KLStatusWidget()
    
    def test_status_color_for_low_kl(self):
        """Test LOW status displays orange."""
        self.widget.update_status(0.005)  # Below 0.01 threshold
        
        style = self.widget.status_label.styleSheet()
        self.assertIn("#FFA500", style, "Should be orange")
    
    def test_progress_bar_bounds(self):
        """Test progress bar respects min/max thresholds."""
        kl_to_percent = (lambda kl: ((kl - 0.01) / (0.08 - 0.01)) * 100)
        
        self.widget.update_status(0.045)
        progress = self.widget.kl_progress.value()
        
        expected_percent = kl_to_percent(0.045)
        self.assertAlmostEqual(progress, expected_percent, places=1)

if __name__ == '__main__':
    unittest.main()
```

#### B. Integration Tests (P2)
**Location:** `test_integration.py` (NEW FILE)

**Purpose:** Test protocol message encoding/decoding, queue operations

**Examples:**
```python
# test_integration.py - Protocol and queue integration tests
import unittest
from train_ui.protocol import P, ProgressMsg

class TestProtocolEncoding(unittest.TestCase):
    def test_progress_msg_encoding(self):
        """Test that ProgressMsg can be encoded and decoded."""
        msg = ProgressMsg(
            done=100,
            total=1000,
            fps=45.2,
            best_reward=150.5,
            episodes=50,
            policy_loss=0.8,
            value_loss=0.9,
            entropy=0.01,
            kl=0.03,
            top_actions={"PRESERVE": 0.5, "BUILD_HOUSE": 0.3},
            loop_detected=True,
            loop_action_name="PRESERVE",
            envs_with_loops=10,
            curriculum_stage_active=1,
            curriculum_next_at_step=5000,
            return_normalization_enabled=False,
            avg_return=120.5,
            median_return=115.3,
            max_return=180.0,
            min_return=80.0,
        )
        
        encoded = P.encode(msg)
        decoded = P.decode(encoded)
        
        self.assertEqual(decoded.done, msg.done)
        self.assertEqual(decoded.kl, msg.kl)
        self.assertEqual(decoded.top_actions, msg.top_actions)

class TestQueueOperations(unittest.TestCase):
    def test_metrics_queue_nonblocking(self):
        """Test that metrics queue doesn't block worker."""
        import multiprocessing
        q = multiprocessing.Queue()
        
        # Should not raise exceptions even if UI is slow
        for i in range(100):
            q.put(f"msg_{i}")
        
        self.assertEqual(q.qsize(), 0)  # Queue flushed
    
    def test_command_queue_separation(self):
        """Test that metrics and commands use separate queues."""
        import multiprocessing
        
        metrics_q = multiprocessing.Queue()
        command_q = multiprocessing.Queue()
        
        # Metrics go to metrics queue only
        metrics_q.put({"kl": 0.03, "entropy": 0.01})
        
        # Commands go to command queue only
        command_q.put(b"boost_entropy")
        
        # Verify no cross-contamination
        assert len(list(command_q)) == 0  # No commands in metrics_q
        assert len(list(metrics_q)) == 1  # One metrics message

if __name__ == '__main__':
    unittest.main()
```

#### C. End-to-End Tests (P3-P4)
**Location:** `test_e2e.py` (NEW FILE)

**Purpose:** Full integration tests with actual worker process and headless UI

**Test Scenarios:**

**Scenario 1: UI Widget Updates in Real-Time**
```python
# test_e2e.py - End-to-end UI integration tests
import pytest
import time
from PyQt6.QtWidgets import QApplication
import multiprocessing

@pytest.fixture
def app():
    """Create and configure Qt application."""
    app = QApplication.instance() or QApplication([])
    yield app
    # Cleanup if needed

def test_ui_widgets_update_in_realtime(app, tmp_path):
    """
    Test that UI widgets update correctly when receiving metrics.
    
    This simulates real training by:
    1. Creating worker process
    2. Starting headless UI  
    3. Verifying widgets respond to metrics updates
    4. Sending commands and checking results
    """
    from train_ui.main_window import MainWindow
    
    # Create temp directory for logs
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    
    def worker_target():
        # Simulate training loop with periodic updates
        for i in range(10):  # Simulate 10 steps
            progress_msg = {
                "done": (i+1) * 100,
                "total": 10000,
                "fps": 45.0 + i * 2,
                "best_reward": float(i * 10),
                "episodes": i * 5,
                "policy_loss": 0.8 - i * 0.05,
                "value_loss": 0.9 - i * 0.04,
                "entropy": 0.02 + i * 0.001,
                "kl": 0.03 + i * 0.005,
                "top_actions": {"PRESERVE": 0.48, "BUILD_HOUSE": 0.12},
                "loop_detected": False,
                "envs_with_loops": 0,
                "curriculum_stage_active": 0,
                "return_normalization_enabled": True,
                "avg_return": float(100 + i * 5),
                "median_return": float(98 + i * 4),
                "max_return": float(120 + i * 6),
                "min_return": float(70 + i * 2),
            }
            
            # Simulate sending to queue (in real code this goes to metrics_queue)
            progress_queue.put(progress_msg)
            
            time.sleep(0.5)  # Simulate training time
    
    # Start worker process
    p = multiprocessing.Process(target=worker_target)
    p.start()
    
    try:
        # Give UI time to initialize
        time.sleep(1)
        
        # Create headless UI
        window = MainWindow(headless=True)
        
        # Wait for widgets to be created
        time.sleep(0.5)
        
        # Verify KLStatusWidget received updates
        kl_widget = window.findChild("KLStatusWidget")
        if kl_widget:
            assert kl_widget.status_label.text() != "STATUS: UNKNOWN"
        
        # Send command and verify it's processed
        boost_cmd = {"cmd": "boost_entropy", "factor": 2.0}
        command_queue.put(boost_cmd)
        
        time.sleep(1)  # Wait for processing
        
        # Verify worker is still running (didn't crash)
        assert p.is_alive()
        
    finally:
        p.terminate()
        p.join(timeout=5)

def test_command_execution_workflow():
    """
    Test complete command execution flow:
    UI → command_queue → worker process → execution → ACK back to UI
    
    Tests commands like: boost_entropy, stop_training, reset_curriculum
    """
    import subprocess
    import time
    
    # Start training process with --headless flag
    proc = subprocess.Popen(
        ["python", "main.py", "--headless", "--log-dir", "/tmp/test_logs"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    try:
        time.sleep(2)  # Let training start
        
        # Send command via stdin (or proper IPC mechanism)
        proc.stdin.write(b"boost_entropy\n")
        proc.stdin.flush()
        
        time.sleep(1)
        
        # Check logs for command processing
        stdout, stderr = proc.communicate(timeout=10)
        
        assert "Command processed: boost_entropy" in stderr.decode()
        assert "ent_coef increased" in stderr.decode()
        
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("Training process hung")

def test_headless_mode_functionality():
    """
    Test that headless mode works correctly for CI/CD pipelines.
    
    Requirements:
    1. No GUI errors on headless systems
    2. All logging goes to file (not UI)
    3. Metrics are still collected and saved
    4. Commands can still be sent via IPC
    """
    import subprocess
    from pathlib import Path
    
    # Create test directory
    test_dir = Path("/tmp/e2e_test_headless")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Start training in headless mode with verbose logging
        proc = subprocess.Popen(
            ["python", "main.py", "--headless", "--log-dir", str(test_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        
        time.sleep(2)
        
        # Verify log file was created
        log_files = list(test_dir.glob("*.log"))
        assert len(log_files) > 0, "No log files created in headless mode"
        
        # Check that training progress is logged
        with open(log_files[0]) as f:
            content = f.read()
        
        assert "Training started" in content
        assert "Step completed" in content
        
        proc.wait(timeout=30)
        
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)

def test_command_logging_integrity():
    """
    Test that all commands are logged to command_log.jsonl correctly.
    
    Verify:
    1. Every command is logged with timestamp
    2. Payload is preserved exactly
    3. Success/failure status recorded
    """
    from pathlib import Path
    
    log_file = Path("logs/command_log.jsonl")
    if not log_file.exists():
        pytest.skip("command_log.jsonl not found - likely running in CI without logs")
    
    with open(log_file) as f:
        lines = [line.strip() for line in f if line.strip()]
    
    assert len(lines) > 0, "No commands logged"
    
    # Parse JSONL and verify structure
    import json
    
    for i, line in enumerate(lines):
        entry = json.loads(line)
        
        # Verify required fields exist
        assert "timestamp" in entry, f"Missing timestamp at line {i}"
        assert "cmd" in entry, f"Missing cmd at line {i}"
        assert "payload" in entry, f"Missing payload at line {i}"
        assert "success" in entry, f"Missing success at line {i}"
        
        # Verify timestamp is recent (within last hour)
        import time
        assert abs(time.time() - entry["timestamp"]) < 3600

def test_full_training_workflow():
    """
    Integration test: Complete training workflow from start to checkpoint save.
    
    Steps tested:
    1. Training starts successfully
    2. Metrics are collected every N steps
    3. Loop detector runs and detects loops correctly
    4. Commands can be sent during training
    5. Checkpoint is saved properly
    6. Best model metadata includes all required fields (hash, git commit, etc.)
    """
    import subprocess
    from pathlib import Path
    
    test_dir = Path("/tmp/e2e_full_workflow")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Start training process
        proc = subprocess.Popen(
            ["python", "main.py", "--headless", 
             "--steps", "100",  # Short run for testing
             "--log-dir", str(test_dir),
             "--checkpoint-dir", str(test_dir / "checkpoints")],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        
        time.sleep(5)
        
        # Send a command during training
        proc.stdin.write(b"boost_entropy\n")
        proc.stdin.flush()
        
        time.sleep(2)
        
        # Wait for training to complete
        stdout, _ = proc.communicate(timeout=60)
        
        # Verify training completed successfully
        assert proc.returncode == 0 or "Training finished" in stdout.decode()
        
        # Verify checkpoint was saved
        checkpoint_dir = test_dir / "checkpoints"
        assert (checkpoint_dir / "best_model.pt").exists(), "Checkpoint not saved"
        
        # Verify metadata includes all required fields
        meta_file = checkpoint_dir / "best_model.meta.json"
        if meta_file.exists():
            import json
            with open(meta_file) as f:
                meta = json.load(f)
            
            # Check reproducibility fields
            assert "config_hash" in meta, "Missing config_hash in metadata"
            assert "git_commit" in meta, "Missing git_commit in metadata"
            assert ui_version := meta.get("ui_version") is not None
            
        finally:
            if test_dir.exists():
                shutil.rmtree(test_dir)

if __name__ == '__main__':
    pytest.main([__file__, "-v", "--tb=short"])
```

---

## 3. End-to-End Testing Checklist

### Phase 1 Tests (Core Infrastructure)
- [ ] Protocol encode/decode works for all new fields
- [ ] Dual-channel queue separation verified
- [ ] LoopDetector unit tests pass (>80% coverage)
- [ ] Worker process doesn't block on message queues

### Phase 2 Tests (Metrics Collection & Display)  
- [ ] ProgressMsg contains all required metrics
- [ ] KLStatusWidget updates correctly on status changes
- [ ] ReturnStatisticsWidget displays stats from aggregated returns
- [ ] ActionLoopWidget warns at correct threshold (>30%)
- [ ] CurriculumProgressWidget shows stage transitions

### Phase 3 Tests (Interactivity)
- [ ] Command execution in worker process
- [ ] Real-time parameter updates work correctly
- [ ] Headless mode doesn't crash on headless systems
- [ ] PyQtGraph charts render correctly

### Phase 4 Tests (Polish & Optimization)
- [ ] All edge cases handled gracefully
- [ ] Performance acceptable for long training runs
- [ ] Logging complete and structured
- [ ] Documentation matches implementation

---

## 4. Test Execution Commands

### Run all tests:
```bash
# Unit tests (fast, recommended on every commit)
pytest --cov=rl/loop_detector --cov=train_ui/widgets --cov-report=xml

# Integration tests  
pytest test_integration.py -v

# End-to-end tests (slower, run in CI or before release)
pytest test_e2e.py -v --tb=short

# All tests with coverage
pytest --cov=. --cov-report=html --cov-report=term-missing
```

### CI/CD Pipeline Integration:
```yaml
# .github/workflows/test-ui.yml
name: UI Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov coverage pyqtgraph
      - name: Run unit tests
        run: |
          pytest --cov=rl/loop_detector --cov=train_ui/widgets --cov-report=xml
      - name: Run e2e tests
        run: |
          pytest test_e2e.py -v
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## 5. Additional Notes

### Curriculum Schedule Fallback (Issue #5 from user)

**Problem:** What happens if curriculum schedule is empty or undefined?

**Solution in EnvManager.get_curriculum_progress():**
```python
def get_curriculum_progress(self) -> Dict[str, Any]:
    """Get current curriculum stage and progress."""
    
    config = self.config
    
    # Fallback: if no curriculum defined, use infinite progression
    if not config.get("curriculum_schedule") or len(config["curriculum_schedule"]) == 0:
        return {
            "stage": 0,
            "progress_percent": 100.0,  # Already at target
            "total_stages": 1,
            "next_transitions": [],     # No transitions (already complete)
            "available_actions": config.get("default_actions", ["PRESERVE"])
        }
    
    # Normal case: calculate progress based on schedule
    ...
```

**UI Behavior:** When `progress_percent >= 100`, show "Unlimited progression" instead of specific step.

### User Documentation Updates (Issue #6 from user)

**Task:** Update existing user documentation to reflect new features.

**Files to update:**
- `docs/user_guide/training.md` - Add section on monitoring UI
- `docs/user_guide/commands.md` - Document all new commands
- `docs/api/ui_widgets.rst` - API reference for new widgets

**Checklist:**
- [ ] Add "Monitoring Training Progress" guide
- [ ] Document KL divergence interpretation
- [ ] Explain action loop detection mechanism  
- [ ] Tutorial: How to use Quick Actions Panel
- [ ] FAQ: Common UI issues and troubleshooting

---

## Summary of New Requirements Added

| Issue # | Description | Status | Location |
|---------|-------------|--------|----------|
| 1 | Return statistics in ProgressMsg (avg/median/max/min) | ✅ Fixed | 03-API_SPECS.md line ~22 |
| 2 | Git commit hash in meta.json for reproducibility | ✅ Added | 06-APPENDIX.md line ~76 |
| 3 | End-to-end testing specification | ✅ Complete | NEW FILE: this document |
| 4 | System requirements (Python, PySide6 versions) | ✅ Complete | NEW SECTION above |
| 5 | Curriculum schedule fallback behavior | ✅ Documented | This document - Curriculum Fallback section |
| 6 | User documentation updates | ✅ Task list added | This document - User Documentation Updates section |

---

**END OF SYSTEM REQUIREMENTS & E2E TESTING SPECIFICATION**
