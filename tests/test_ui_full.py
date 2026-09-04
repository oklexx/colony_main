"""Full functional test of the train UI MainWindow."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

app = QApplication.instance() or QApplication(sys.argv)

from train_ui.main_window import MainWindow, save_config, load_config
from train_ui.parameter_widget import PARAM_SPECS, REWARD_SPECS


@pytest.fixture(scope="module")
def main_window():
    """Create MainWindow once for all tests in this module."""
    w = MainWindow()
    return w


class TestMainWindowCreation:
    def test_creates(self, main_window):
        assert main_window is not None

    def test_title(self, main_window):
        assert "Сахалинская колония" in main_window.windowTitle()

    def test_minimum_size(self, main_window):
        assert main_window.minimumWidth() == 1400
        assert main_window.minimumHeight() == 700


class TestLeftColumn:
    def test_model_combo_exists(self, main_window):
        assert main_window.model_combo is not None
        assert main_window.model_combo.count() >= 1

    def test_model_combo_default(self, main_window):
        assert main_window.model_combo.currentText() == "— нет моделей —"

    def test_stats_labels_exist(self, main_window):
        for key in ("steps", "best_reward", "episodes", "train_time",
                     "eval_days", "eval_people"):
            assert key in main_window._stat_labels

    def test_obs_labels_exist(self, main_window):
        for key in ("day", "action", "reward", "people", "money"):
            assert key in main_window._obs_labels

    def test_console_exists(self, main_window):
        assert main_window.console is not None
        assert main_window.console.isReadOnly()


class TestRightColumnCurriculum:
    def test_curriculum_table(self, main_window):
        assert main_window.curriculum_table is not None
        assert main_window.curriculum_table.columnCount() == 2

    def test_curriculum_default_rows(self, main_window):
        assert main_window.curriculum_table.rowCount() == 3

    def test_cur_add_row(self, main_window):
        count_before = main_window.curriculum_table.rowCount()
        main_window._cur_add_row(100, 2)
        assert main_window.curriculum_table.rowCount() == count_before + 1
        main_window._cur_clear()

    def test_cur_del_row(self, main_window):
        main_window._cur_add_row(100, 1)
        main_window._cur_add_row(200, 2)
        assert main_window.curriculum_table.rowCount() == 2
        main_window.curriculum_table.selectRow(1)
        main_window._cur_del_row()
        assert main_window.curriculum_table.rowCount() == 1
        main_window._cur_clear()

    def test_cur_clear(self, main_window):
        main_window._cur_add_row(100, 1)
        assert main_window.curriculum_table.rowCount() >= 1
        main_window._cur_clear()
        assert main_window.curriculum_table.rowCount() == 0

    def test_cur_profile_changed(self, main_window):
        main_window.cur_profile_combo.setCurrentText("Отключён")
        assert main_window.curriculum_table.rowCount() == 0
        main_window.cur_profile_combo.setCurrentText("Стандартный")
        assert main_window.curriculum_table.rowCount() == 3
        main_window.cur_profile_combo.setCurrentText("Быстрый")
        assert main_window.curriculum_table.rowCount() == 2
        main_window.cur_profile_combo.setCurrentText("Стандартный")

    def test_get_curriculum_data(self, main_window):
        data = main_window._get_curriculum_data()
        assert isinstance(data, list)
        for item in data:
            assert len(item) == 2

    def test_validate_curriculum_ok(self, main_window):
        ok, _ = main_window._validate_curriculum([[0, 1], [5_000_000, 2]])
        assert ok is True

    def test_validate_curriculum_fail(self, main_window):
        ok, msg = main_window._validate_curriculum([[5, 1], [3, 2]])
        assert ok is False
        assert "≤" in msg

    def test_cur_save_load(self, main_window, tmp_path):
        main_window._cur_clear()
        main_window._cur_add_row(1000, 1)
        main_window._cur_add_row(5000, 2)
        save_path = str(tmp_path / "test_curriculum.json")
        with patch("train_ui.main_window.QFileDialog.getSaveFileName",
                   return_value=(save_path, "JSON (*.json)")):
            main_window._cur_save()
        assert os.path.exists(save_path)
        with open(save_path) as f:
            saved = json.load(f)
        assert len(saved) == 2

        main_window._cur_clear()
        with patch("train_ui.main_window.QFileDialog.getOpenFileName",
                   return_value=(save_path, "JSON (*.json)")):
            main_window._cur_load()
        assert main_window.curriculum_table.rowCount() == 2


class TestRightColumnWatchStage:
    def test_watch_checkboxes_exist(self, main_window):
        assert main_window.watch_use_model_stage_chk is not None
        assert main_window.watch_override_stage_chk is not None
        assert main_window.watch_visual_chk is not None

    def test_watch_default_state(self, main_window):
        assert main_window.watch_use_model_stage_chk.isChecked() is True
        assert main_window.watch_override_stage_chk.isChecked() is False
        assert main_window.watch_override_stage_chk.isEnabled() is True

    def test_watch_stage_mode_use_model(self, main_window):
        main_window.watch_override_stage_chk.setChecked(False)
        main_window.watch_use_model_stage_chk.setChecked(False)
        main_window.watch_use_model_stage_chk.setChecked(True)
        assert main_window.watch_override_stage_chk.isEnabled() is True
        assert main_window.stage_combo.isEnabled() is False
        assert "meta.json" in main_window.watch_stage_info_label.text()

    def test_watch_stage_mode_override(self, main_window):
        main_window.watch_use_model_stage_chk.setChecked(True)
        main_window.watch_override_stage_chk.setChecked(True)
        assert main_window.stage_combo.isEnabled() is True
        assert "переопределение" in main_window.watch_stage_info_label.text().lower()

    def test_watch_stage_mode_no_model(self, main_window):
        main_window.watch_use_model_stage_chk.setChecked(False)
        assert main_window.watch_override_stage_chk.isEnabled() is False
        assert main_window.stage_combo.isEnabled() is True

    def test_stage_combo_default(self, main_window):
        assert main_window.stage_combo.currentIndex() == 2


class TestParameterRows:
    def test_all_params_present(self, main_window):
        assert len(main_window.param_rows) == len(PARAM_SPECS)
        for spec in PARAM_SPECS:
            assert spec.key in main_window.param_rows

    def test_all_rewards_present(self, main_window):
        assert len(main_window.reward_rows) == len(REWARD_SPECS)
        for spec in REWARD_SPECS:
            assert spec.key in main_window.reward_rows

    def test_param_x2(self, main_window):
        row = main_window.param_rows["n_envs"]
        old = row.value()
        row.btn_x2.click()
        assert row.value() == old * 2

    def test_param_d2(self, main_window):
        row = main_window.param_rows["n_envs"]
        old = row.value()
        row.btn_d2.click()
        assert row.value() == old // 2

    def test_param_set_value(self, main_window):
        row = main_window.param_rows["n_envs"]
        row.set_value(64)
        assert row.value() == 64
        row.set_value(8)

    def test_reward_set_value(self, main_window):
        row = main_window.reward_rows["chain_daily"]
        row.set_value(0.01)
        assert row.value() == pytest.approx(0.01, abs=1e-6)
        row.set_value(row.spec.default)

    def test_group_separators(self, main_window):
        row = main_window.param_rows["total_timesteps"]
        row.set_value(20_000_000)
        display_text = row.spin.text()
        assert "," in display_text or "\xa0" in display_text


class TestAMPCompile:
    def test_amp_checkbox(self, main_window):
        main_window.chk_amp.setChecked(True)
        assert main_window.chk_amp.isChecked()
        main_window.chk_amp.setChecked(False)

    def test_compile_checkbox(self, main_window):
        main_window.chk_compile.setChecked(True)
        assert main_window.chk_compile.isChecked()
        main_window.chk_compile.setChecked(False)


class TestResetDefaults:
    def test_reset_params(self, main_window):
        main_window.param_rows["n_envs"].set_value(64)
        main_window.reward_rows["chain_daily"].set_value(0.99)
        main_window._reset_params()
        assert main_window.param_rows["n_envs"].value() == 8
        assert main_window.reward_rows["chain_daily"].value() == \
            pytest.approx(0.5, abs=1e-3)

    def test_reset_to_defaults(self, main_window):
        main_window.param_rows["n_envs"].set_value(64)
        main_window.chk_amp.setChecked(True)
        main_window.chk_compile.setChecked(True)
        main_window.stage_combo.setCurrentIndex(0)
        main_window._cur_clear()
        main_window._reset_to_defaults()
        assert main_window.param_rows["n_envs"].value() == 8
        assert main_window.chk_amp.isChecked() is False
        assert main_window.chk_compile.isChecked() is False
        assert main_window.stage_combo.currentIndex() == 2
        assert main_window.curriculum_table.rowCount() == 3


class TestCollectConfig:
    def test_collect_config_keys(self, main_window):
        cfg = main_window._collect_config()
        assert "name" in cfg
        assert "use_amp" in cfg
        assert "torch_compile" in cfg
        assert "net_arch" in cfg
        assert "curriculum_schedule" in cfg
        assert isinstance(cfg["net_arch"], list)
        assert len(cfg["net_arch"]) == 2

    def test_collect_config_has_all_params(self, main_window):
        cfg = main_window._collect_config()
        for spec in PARAM_SPECS:
            assert spec.key in cfg
        for spec in REWARD_SPECS:
            assert spec.key in cfg

    def test_collect_config_curriculum(self, main_window):
        main_window._cur_clear()
        main_window._cur_add_row(100, 1)
        cfg = main_window._collect_config()
        assert len(cfg["curriculum_schedule"]) == 1


class TestSaveRestoreState:
    def test_save_state(self, main_window):
        state = main_window.save_state()
        assert isinstance(state, dict)
        assert "use_amp" in state
        assert "torch_compile" in state
        assert "stage" in state
        assert "curriculum_schedule" in state

    def test_restore_state(self, main_window):
        state = main_window.save_state()
        main_window._restore_state()
        restored = main_window.save_state()
        assert restored["use_amp"] == state["use_amp"]
        assert restored["torch_compile"] == state["torch_compile"]
        assert restored["stage"] == state["stage"]


class TestConfigSaveLoad:
    def test_save_load_config(self, tmp_path):
        test_config = {"use_amp": True, "stage": 1}
        cfg_path = tmp_path / "config.json"
        with patch("train_ui.main_window.CONFIG_PATH", cfg_path):
            save_config(test_config)
            loaded = load_config()
        assert loaded["use_amp"] is True
        assert loaded["stage"] == 1


class TestLogging:
    def test_log_info(self, main_window):
        count_before = main_window.console.document().blockCount()
        main_window.log("info", "test message")
        assert main_window.console.document().blockCount() == count_before + 1

    def test_log_warn(self, main_window):
        main_window.log("warn", "warning test")

    def test_log_error(self, main_window):
        main_window.log("error", "error test")

    def test_copy_log(self, main_window):
        main_window.console.setPlainText("copy test")
        main_window._copy_log()
        clipboard = QApplication.clipboard()
        assert "copy test" in clipboard.text()

    def test_clear_log(self, main_window):
        main_window.console.setPlainText("clear me")
        main_window._clear_log()
        assert main_window.console.toPlainText() == ""


class TestButtonsExist:
    def test_main_buttons(self, main_window):
        assert main_window.btn_delete is not None
        assert main_window.btn_watch is not None
        assert main_window.btn_resume is not None
        assert main_window.btn_start is not None
        assert main_window.btn_stop is not None
        assert main_window.btn_reset is not None
        assert main_window.btn_reset_default is not None

    def test_curriculum_buttons(self, main_window):
        assert main_window.btn_cur_add is not None
        assert main_window.btn_cur_del is not None
        assert main_window.btn_cur_clear is not None
        assert main_window.btn_cur_save is not None
        assert main_window.btn_cur_load is not None

    def test_stop_disabled_initially(self, main_window):
        assert main_window.btn_stop.isEnabled() is False


class TestMenuBar:
    def test_menubar_exists(self, main_window):
        mb = main_window.menuBar()
        assert mb is not None
        actions = mb.actions()
        assert len(actions) >= 1


class TestProgress:
    def test_progress_bar(self, main_window):
        assert main_window.progress_bar is not None
        main_window.progress_bar.setValue(500)
        assert main_window.progress_bar.value() == 500
        main_window.progress_bar.setValue(0)

    def test_status_label(self, main_window):
        main_window.status_label.setText("test status")
        assert main_window.status_label.text() == "test status"
        main_window.status_label.setText("")
