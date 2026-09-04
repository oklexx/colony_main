# -*- coding: utf-8 -*-
"""Test script to verify all UI improvements are working."""

import sys
import os
from pathlib import Path

# Add project root to path (handle when run from command line)
if not str(Path.cwd()) in sys.path:
    sys.path.insert(0, str(Path.cwd()))


def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_ok(message):
    print(f"  [OK] {message}")


def print_error(message):
    print(f"  [FAIL] {message}")


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, msg):
        self.passed += 1
        print_ok(msg)
    
    def add_fail(self, msg):
        self.failed += 1
        self.errors.append(msg)
        print_error(msg)
    
    def summary(self):
        total = self.passed + self.failed
        print("\n" + "="*60)
        print(f"TOTAL: {self.passed} passed, {self.failed} failed ({total} tests)")
        print("="*60)
        if self.errors:
            print("\nErrors:")
            for err in self.errors:
                print(f"  - {err}")
        return self.failed == 0


def test_imports(result):
    print_section("TEST 1: Module Imports")
    
    modules = [
        ("train_ui.protocol", "Protocol with new fields"),
        ("train_ui.kl_status_widget", "KL Status Widget"),
        ("train_ui.curriculum_progress_widget", "Curriculum Progress Widget"),
        ("train_ui.action_loop_widget", "Action Loop Widget"),
        ("train_ui.return_statistics_widget", "Return Statistics Widget"),
        ("train_ui.quick_actions_widget", "Quick Actions Widget"),
        ("train_ui.dashboard_widget", "Dashboard Widget"),
        ("rl.loop_detector", "Loop Detector"),
        ("rl.async_trainer", "Async Trainer"),
        ("rl.env_manager", "Env Manager"),
    ]
    
    for module_name, desc in modules:
        try:
            __import__(module_name)
            result.add_pass(f"{desc} ({module_name})")
        except Exception as e:
            result.add_fail(f"{desc}: {str(e)[:50]}")


def test_protocol(result):
    print_section("TEST 2: Protocol Fields")
    
    try:
        from train_ui import protocol as P
        
        # Test ProgressMsg new fields
        progress = P.ProgressMsg(
            done=1000,
            total=1000000,
            kl=0.03,
            entropy=0.025,
            top_actions={"BUILD_HOUSE": 45.2, "MOVE_RIGHT": 32.1, "PRESERVE": 18.7},
            loop_detected=True,
            loop_action_name="WAIT",
            envs_with_loops=4,
            curriculum_stage_active=1,
            curriculum_next_at_step=500000
        )
        
        data = progress.to_dict()
        
        required_fields = [
            "top_actions", "loop_detected", "loop_action_name",
            "envs_with_loops", "curriculum_stage_active", "curriculum_next_at_step"
        ]
        
        for field in required_fields:
            if field in data:
                result.add_pass(f"ProgressMsg.{field} present")
            else:
                result.add_fail(f"ProgressMsg.{field} missing")
        
        # Test CommandMsg
        cmd = P.CommandMsg(cmd="pause", payload={"paused": True})
        cmd_data = cmd.to_dict()
        
        if "cmd" in cmd_data and "payload" in cmd_data:
            result.add_pass("CommandMsg structure correct")
        else:
            result.add_fail("CommandMsg structure incorrect")
            
    except Exception as e:
        result.add_fail(f"Protocol test failed: {str(e)}")


def test_loop_detector(result):
    print_section("TEST 3: Loop Detector")
    
    try:
        from rl.loop_detector import LoopDetector
        
        # Test basic initialization
        detector = LoopDetector()
        result.add_pass("LoopDetector initialized")
        
        if detector.consecutive_threshold == 3:
            result.add_pass("Default threshold correct (3)")
        else:
            result.add_fail(f"Threshold wrong: {detector.consecutive_threshold}")
        
        # Test with config
        detector2 = LoopDetector({"consecutive_threshold": 5})
        if detector2.consecutive_threshold == 5:
            result.add_pass("Custom threshold works (5)")
        else:
            result.add_fail(f"Custom threshold wrong: {detector2.consecutive_threshold}")
        
        # Test update_batch
        alerts = detector.update_batch([
            {"env_idx": 0, "action": "MOVE_RIGHT"},
            {"env_idx": 0, "action": "MOVE_RIGHT"},
            {"env_idx": 0, "action": "MOVE_RIGHT"},
        ])
        
        if 0 in alerts and alerts[0] == "MOVE_RIGHT":
            result.add_pass("Loop detection working")
        else:
            result.add_fail(f"Alerts not detected: {alerts}")
        
        # Test stats
        stats = detector.get_stats()
        required_keys = ["total_envs", "envs_with_loops", "max_consecutive"]
        
        for key in required_keys:
            if key in stats:
                result.add_pass(f"Stats.{key} available")
            else:
                result.add_fail(f"Stats.{key} missing")
                
    except Exception as e:
        import traceback
        result.add_fail(f"LoopDetector test failed: {str(e)}")


def test_widgets(result):
    print_section("TEST 4: Widget Instantiation")
    
    widgets = [
        ("KLStatusWidget", "train_ui.kl_status_widget"),
        ("CurriculumProgressWidget", "train_ui.curriculum_progress_widget"),
        ("ActionLoopWidget", "train_ui.action_loop_widget"),
        ("ReturnStatisticsWidget", "train_ui.return_statistics_widget"),
        ("QuickActionsWidget", "train_ui.quick_actions_widget"),
    ]
    
    for widget_name, module in widgets:
        try:
            module_obj = __import__(module, fromlist=[widget_name])
            widget_class = getattr(module_obj, widget_name)
            
            # Create instance (parent=None)
            widget = widget_class()
            result.add_pass(f"{widget_name} instantiated")
            
        except Exception as e:
            result.add_fail(f"{widget_name}: {str(e)[:50]}")


def main():
    print("\n" + "#"*60)
    print("# UI Improvements - Verification Test Suite")
    print("#"*60)
    
    result = TestResult()
    
    # Run all tests
    test_imports(result)
    test_protocol(result)
    test_loop_detector(result)
    test_widgets(result)
    
    # Final summary
    success = result.summary()
    
    print("\n" + "="*60)
    if success:
        print("  SUCCESS - Project is ready for use!")
    else:
        print("  FAILURE - Some tests failed")
    print("="*60 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
