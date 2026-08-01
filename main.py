# -------------------------------------------------
# main.py
#   実行専用エントリポイント。GUI/制御ロジックは module_main_window_JP に分離済み。
#   黒タイル対策の統合設計版。判定ロジックを module_yolo（状態機械＋フレーム保持一本化）
#   に差し替えただけで、その他は main_5goki_JP.py と同一。旧版はそのまま温存してある。
# -------------------------------------------------
import sys

from PySide6.QtWidgets import QApplication

import log_config
from module_main_window_JP import StartupWindow

log = log_config.get_logger("main")

# ==========================================================
# 実行ブロック
# ==========================================================
if __name__ == "__main__":
    # コンソール出力を統一（INFO以上を色付きで表示、DEBUGは logs/app_5goki_*.log へ）
    log_config.setup_logging(line="5goki")
    log.info("DCRsystem 起動")

    app = QApplication(sys.argv)
    window = StartupWindow()
    window.show()
    sys.exit(app.exec())
