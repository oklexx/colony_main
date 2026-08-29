from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication

from train_ui.main_window import MainWindow, load_config, save_config


def _load_dark_theme(app: QApplication) -> None:
    qss_path = Path(__file__).resolve().parent / "dark_theme.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Sakhalin Colony Trainer")
    app.setStyle("Fusion")
    _load_dark_theme(app)
    config = load_config()
    win = MainWindow(config=config)
    win.show()
    code = app.exec()
    save_config(win.save_state())
    return code


if __name__ == "__main__":
    sys.exit(main())
