"""
果実クラス確定 GUI遷移デモ

カメラ・YOLO・リレー等の実ハードウェアなしに、「1個体分の検出が少しずつ蓄積され、
個体が確定した瞬間に健全/障害の二値判定が下り、履歴テーブル・集計欄が更新される」
という一連の流れを画面で確認するためのツール。

表示・判定ロジックは本番（module_main_window_JP.MainWindow / module_yolo.YoloDetector）
の関数をそのまま束縛して使う。デモ用に再実装していないため、本番の判定・表示ロジックが
変わればこのデモの挙動も自動的に追従する（二重実装によるズレが起きない）。

デモが独自に持つのはハードウェアに依存する部分（カメラ映像・リレー発火・コスト計測等）の
代わりとなる「合成検出データの生成」と「進行状況のログ表示」だけ。

使い方（アプリのルートディレクトリで）:
    uv run python standalone/classification_gui_demo.py

操作:
    「次の検出イベントへ進める」 … 1検出ずつ手動で進める
    「自動再生」                 … 一定間隔で自動的に進める
    「リセット」                 … 履歴・集計をクリアして最初から
"""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QGroupBox,
)
from PySide6.QtCore import Qt, QTimer

import module_gui_JP as gui_mod
import module_relay as r_ctr
import module_main_window_JP as mw_mod
import module_yolo as yolo_mod

# ==========================================================
# 合成シナリオ: (カメラ名, クラス名, 信頼度) の列で1個体分の検出を表す。
#   健全/障害の判定の分かれ目が見えるよう、意図的に代表的なケースを揃えている。
# ==========================================================
SCENARIOS = [
    {
        "desc": "健全 - 2台のカメラで healthy が一致 → 健全確定（高いハードルを満たす）",
        "events": [("cam_top", "healthy", 0.93), ("cam_under", "healthy", 0.88)],
    },
    {
        "desc": "健全(確証不足) - healthyの検出が1台のみ → 健全と断定できず安全側で除去",
        "events": [("cam_inside", "healthy", 0.97)],
    },
    {
        "desc": "障害(即時) - crackが1台で検出 → カメラ台数によらず即座に障害",
        "events": [("cam_top", "healthy", 0.90), ("cam_outside", "crack", 0.85)],
    },
    {
        "desc": "障害(複数クラス混在) - twinとcrackが別カメラで検出 → 信頼度最大の方を表示ラベルに採用",
        "events": [("cam_top", "twin", 0.95), ("cam_under", "crack", 0.82), ("cam_inside", "healthy", 0.90)],
    },
    {
        "desc": "未熟果 - unripeも障害と同じく即時除去対象",
        "events": [("cam_outside", "unripe", 0.93), ("cam_top", "healthy", 0.91)],
    },
    {
        "desc": "健全 - 3台で healthy 一致（余裕を持って健全確定）",
        "events": [("cam_top", "healthy", 0.95), ("cam_under", "healthy", 0.90), ("cam_inside", "healthy", 0.92)],
    },
]


class DemoWindow(QMainWindow):
    """本番の履歴描画・判定チャンネル決定ロジックをそのまま束縛して使うデモウィンドウ。
    束縛したメソッドは module_main_window_JP モジュールのグローバル
    （CLASS_DISPLAY・HISTORY_CAM_COLORS・チップHTML生成関数等）を参照して動くため、
    このファイル側で個別に再実装する必要はない。"""

    _resolve_channel       = mw_mod.MainWindow._resolve_channel
    _append_history_record = mw_mod.MainWindow._append_history_record
    _render_history_table   = mw_mod.MainWindow._render_history_table
    _render_stats_grid     = mw_mod.MainWindow._render_stats_grid
    update_history_display = mw_mod.MainWindow.update_history_display

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GUI遷移デモ（果実クラス確定シミュレータ）")
        self.resize(1400, 900)

        # _append_history_record / _render_history_table / _render_stats_grid が読む状態
        self.history_data: list = []
        self.detection_counts = {cls: 0 for cls in gui_mod.CLASS_DISPLAY}

        self._cycle_id = 0
        self._scenario_idx = 0
        self._pending: list = []          # 現個体で蓄積中の YoloResult
        self._pending_events: list = []   # 現個体の残り検出イベント

        self._build_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(700)
        self._timer.timeout.connect(self._advance_one_event)

        self.update_history_display()
        self._log("デモ開始。「次の検出イベントへ進める」または「自動再生」で進めます。")

    # ------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left = QVBoxLayout()
        root.addLayout(left, 3)

        self.label_history = QLabel("入力待機中...")
        self.label_history.setStyleSheet(gui_mod.LABEL_HISTORY_STYLE)
        self.label_history.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.label_history.setMinimumHeight(360)
        left.addWidget(self.label_history)

        self.label_stats = QLabel("入力待機中...")
        self.label_stats.setStyleSheet(gui_mod.LABEL_STATS_STYLE)
        self.label_stats.setMinimumHeight(260)
        left.addWidget(self.label_stats)

        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setStyleSheet(
            "font-family:Consolas,monospace; font-size:12px; background:#111; color:#ddd;")
        left.addWidget(self._log_view, 1)

        right = QVBoxLayout()
        root.addLayout(right, 1)

        ctrl = QGroupBox("シミュレータ操作")
        cv = QVBoxLayout(ctrl)
        btn_step = QPushButton("次の検出イベントへ進める")
        btn_step.clicked.connect(self._advance_one_event)
        cv.addWidget(btn_step)

        self._btn_auto = QPushButton("自動再生 開始")
        self._btn_auto.setCheckable(True)
        self._btn_auto.toggled.connect(self._toggle_auto)
        cv.addWidget(self._btn_auto)

        btn_reset = QPushButton("リセット")
        btn_reset.clicked.connect(self._reset)
        cv.addWidget(btn_reset)
        right.addWidget(ctrl)

        legend = QGroupBox("判定ロジックの要点")
        lv = QVBoxLayout(legend)
        for text in (
            "◎ = 確定クラスとしてハイライトされる検出",
            "障害判定: healthy以外が1カメラ・1検出でも即成立（低いハードル）",
            f"健全判定: {yolo_mod.HEALTHY_CONFIRM_MIN_CAMS}台以上の異なるカメラで"
            "healthy一致が必要（高いハードル）",
            "健全と表示されても、確証不足なら is_damaged=True で除去されることがある",
        ):
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lv.addWidget(lbl)
        right.addWidget(legend)
        right.addStretch(1)

    # ------------------------------------------------------
    def _toggle_auto(self, checked: bool) -> None:
        if checked:
            self._btn_auto.setText("自動再生 停止")
            self._timer.start()
        else:
            self._btn_auto.setText("自動再生 開始")
            self._timer.stop()

    def _reset(self) -> None:
        self._timer.stop()
        self._btn_auto.setChecked(False)
        self.history_data.clear()
        for k in self.detection_counts:
            self.detection_counts[k] = 0
        self._cycle_id = 0
        self._scenario_idx = 0
        self._pending = []
        self._pending_events = []
        self._log_view.clear()
        self.update_history_display()
        self._log("リセットしました。")

    def _log(self, text: str) -> None:
        self._log_view.append(text)

    # ------------------------------------------------------
    # 検出イベントの進行（1回呼ぶたびに検出1件を「受信」する）
    # ------------------------------------------------------
    def _advance_one_event(self) -> None:
        if not self._pending_events:
            scenario = SCENARIOS[self._scenario_idx % len(SCENARIOS)]
            self._scenario_idx += 1
            self._pending_events = list(scenario["events"])
            self._pending = []
            self._cycle_id += 1
            self._log(f"\n--- 個体 #{self._cycle_id:03d} 検出開始: {scenario['desc']} ---")

        cam_name, label, conf = self._pending_events.pop(0)
        det = yolo_mod.YoloResult(self._cycle_id, label, conf, cam_name)
        self._pending.append(det)
        self._log(f"  [{cam_name}] {label} を信頼度{conf:.0%}で検出（累計{len(self._pending)}件）")

        if not self._pending_events:
            self._finalize_current()

    def _finalize_current(self) -> None:
        # 二値判定は本番と同一関数（module_yolo.YoloDetector._resolve_quality）。
        #   self を使わない純粋なロジックなので None を渡して直接呼び出せる。
        best = yolo_mod.YoloDetector._resolve_quality(None, self._pending)
        self._attach_breakdown(best, self._pending)

        if best.label_name in self.detection_counts:
            self.detection_counts[best.label_name] += 1

        channel, _display_name = self._resolve_channel(best)
        verdict = "障害" if best.is_damaged else "健全"
        action  = "除去（不良ライン）" if channel == r_ctr.RelayChannel.REMOVE else "運搬（健全ライン）"
        self._log(f"  === 確定: label={best.label_name}  is_damaged={best.is_damaged} "
                  f"→ {verdict}判定 → {action} ===")

        self._append_history_record(best, best.id, best.label_name)
        self.update_history_display()

    @staticmethod
    def _attach_breakdown(best, detections: list) -> None:
        """best.class_breakdown / per_cam_breakdown を組み立てる。
        module_yolo.YoloDetector._attach_cycle_stats の該当部分を模したもの。
        このデモはコスト計測・カメラ実測値などハードウェア関連の集計は行わないため、
        履歴表示に必要な「クラス別内訳」「カメラ別内訳」のみ再現している。"""
        by_class: dict[str, float] = {}
        for d in detections:
            if d.label_name not in by_class or d.confidence > by_class[d.label_name]:
                by_class[d.label_name] = d.confidence
        best.class_breakdown = sorted(by_class.items(), key=lambda kv: kv[1], reverse=True)

        cam_class_max: dict[str, dict[str, float]] = {}
        for d in detections:
            m = cam_class_max.setdefault(d.cam_name, {})
            if d.label_name not in m or d.confidence > m[d.label_name]:
                m[d.label_name] = d.confidence
        per_cam = {}
        for cam, cmax in cam_class_max.items():
            top_label = max(cmax, key=cmax.get)
            per_cam[cam] = {"top": (top_label, cmax[top_label]), "final_conf": cmax.get(best.label_name)}
        best.per_cam_breakdown = per_cam


def main() -> int:
    app = QApplication(sys.argv)
    win = DemoWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
