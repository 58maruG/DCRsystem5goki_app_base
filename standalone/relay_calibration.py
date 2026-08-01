# -------------------------------------------------
# relay_calibration.py  （GUIスライダー方式・新仕様）
#   リレー（排出弁）の開弁タイミングを、スライダーで角度をリアルタイムに動かしながら
#   実機に合わせ込むキャリブレーションツール。
#
#   方式（旧CLI版からの刷新）:
#     - カメラ＋YOLO＋トラッカーを本番(main.py)と同条件で回す。
#     - GUIに「健全果(運搬弁)」「被害果(除去弁)」の角度スライダーを置く。
#     - 果実が確定（カメラ外へ通過）した瞬間を起点に、スライダーの角度から計算した
#         待機時間 = sec * (角度 / 360)
#       だけ待って、実際にリレー弁を開く。
#     - 果実が弁の真下に来た瞬間と開弁が合うようにスライダーを動かして詰めていく。
#     - 合った角度を relay_config.json に保存。本番(module_relay)が次回起動時に
#       自動で読み込み、wait = sec * (角度/360) で適用する。
#
#   起動:
#     uv run python standalone/relay_calibration.py
#
#   注意:
#     - キャリブ中は実際に弁を開く。周囲の安全を確認してから「開始」すること。
#     - 健全果を流せば運搬弁、被害果を流せば除去弁が、それぞれの角度で発火する。
#     - Arduino が非常停止(ESTOP)を送ってきたら回転・発火を止める。
# -------------------------------------------------
import sys
import os
import json
import datetime

# このスクリプトはプロジェクト直下のモジュールを参照するため、親ディレクトリを import パスに追加
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QSlider, QSpinBox,
    QPushButton, QGridLayout, QVBoxLayout, QHBoxLayout, QGroupBox,
)
from PySide6.QtCore import Qt, QRunnable, QThreadPool, Signal, Slot, QObject
from PySide6.QtGui import QImage, QPixmap

# 本番と同じモジュール群（定数も流用して齟齬を防ぐ）
import module_relay as r_ctr
import module_motor_serial as motor_ctr
import module_cameras_5goki as cam_ctr
import module_yolo as yolo_ctr

# Arduino のシリアルポート。None で自動検出。うまくいかない場合は "COM3" 等を直接指定する
SERIAL_PORT = None

# main.py の update_camera_delays と同じ値（確定タイミング再現のため合わせる）
CAMERA_DELAYS = {"cam_under": 1.922, "cam_inside": 2.015}

# 角度スライダーの範囲。QSlider は整数のみなので「0.5°単位（=度×2）」で扱う。
#   読み出し時に2で割って角度[度]へ戻す（_deg を参照）。
ANGLE_UNIT = 2          # 1目盛 = 1/ANGLE_UNIT 度（=0.5°）
ANGLE_MIN  = 0
ANGLE_MAX  = 180 * ANGLE_UNIT

# 2×2 表示のカメラ並び（main のタイルと同順）
CAM_ORDER = ["cam_inside", "cam_outside", "cam_under", "cam_top"]


def sec_per_rotation(speed: int) -> float:
    """module_relay と同じ計算式で1回転相当の秒(sec)を返す。"""
    delay       = r_ctr.SPEED_MAP[speed]
    t_one_pulse = delay * 2
    return t_one_pulse * r_ctr.PULSE_PER_ROTATION * 2  # ギア比2


# ==========================================================
# リレー発火ワーカー: 確定時にライブ角度の待機時間だけ待って弁を開く。
#   GUIスレッドを止めないようバックグラウンドで実行する。
# ==========================================================
class RelayFireWorker(QRunnable):
    def __init__(self, relay, channel, wait_sec):
        super().__init__()
        self.relay    = relay
        self.channel  = channel
        self.wait_sec = wait_sec

    def run(self):
        try:
            self.relay.fire(self.channel, self.wait_sec)
        except Exception as e:
            print(f"!! リレー発火でエラー: {e}")


# ==========================================================
# Arduino の非常停止(ESTOP)をGUIスレッドへ渡すシグナル
# ==========================================================
class MotorSignals(QObject):
    estop = Signal()
    estop_cleared = Signal()


# ==========================================================
# キャリブレーション画面
# ==========================================================
class CalibrationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("リレー角度キャリブレーション")

        self.running   = False     # 検知＋発火を行うか
        self.relay_ok  = False
        self.motor_ok  = False
        self.pool      = QThreadPool()

        self._build_ui()
        self._init_hardware()

        self._refresh_wait_labels()

    # ------------------------------------------------------
    # UI 構築
    # ------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # --- 左: 2×2 カメラ映像 ---
        cam_box = QGridLayout()
        self.cam_labels = {}
        positions = {"cam_inside": (0, 0), "cam_outside": (0, 1),
                     "cam_under": (1, 0), "cam_top": (1, 1)}
        for name, (r, c) in positions.items():
            lbl = QLabel(name)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(420, 320)
            lbl.setStyleSheet("background:#202020; color:#AAAAAA; border:1px solid #444;")
            self.cam_labels[name] = lbl
            cam_box.addWidget(lbl, r, c)
        root.addLayout(cam_box, stretch=3)

        # --- 右: 操作パネル ---
        panel = QVBoxLayout()

        # 速度
        sp_box = QHBoxLayout()
        sp_box.addWidget(QLabel("速度(1-10):"))
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 10)
        self.speed_spin.setValue(6)
        self.speed_spin.valueChanged.connect(self._on_speed_changed)
        sp_box.addWidget(self.speed_spin)
        sp_box.addStretch(1)
        panel.addLayout(sp_box)

        # relay_config.json に保存済みの角度を初期値にする。0.5°単位に変換。
        cal = r_ctr.load_relay_calibration()
        init_remove    = int(round(cal["remove"]["angle"] * ANGLE_UNIT))
        init_transport = int(round(cal["transport"]["angle"] * ANGLE_UNIT))

        # 除去弁（被害果）スライダー
        self.remove_slider, self.remove_wait_lbl = self._make_angle_group(
            panel, "被害果 → 除去弁", init_remove)
        # 運搬弁（健全果）スライダー
        self.transport_slider, self.transport_wait_lbl = self._make_angle_group(
            panel, "健全果 → 運搬弁", init_transport)

        # 開始/停止
        self.btn_start = QPushButton("開始")
        self.btn_start.setStyleSheet("font-size:20px; padding:10px;")
        self.btn_start.clicked.connect(self._toggle_running)
        panel.addWidget(self.btn_start)

        # 保存
        self.btn_save = QPushButton("この角度を保存")
        self.btn_save.setStyleSheet("font-size:18px; padding:8px;")
        self.btn_save.clicked.connect(self._save)
        panel.addWidget(self.btn_save)

        # ステータス
        self.status_lbl = QLabel("初期化中...")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("font-size:15px; color:#222; padding:6px;")
        panel.addWidget(self.status_lbl)

        panel.addStretch(1)
        root.addLayout(panel, stretch=1)

    def _make_angle_group(self, parent_layout, title, default_angle):
        box = QGroupBox(title)
        v = QVBoxLayout(box)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(ANGLE_MIN, ANGLE_MAX)
        slider.setValue(default_angle)
        slider.valueChanged.connect(self._refresh_wait_labels)
        v.addWidget(slider)

        wait_lbl = QLabel()
        wait_lbl.setStyleSheet("font-size:16px; font-weight:bold;")
        v.addWidget(wait_lbl)

        parent_layout.addWidget(box)
        return slider, wait_lbl

    @staticmethod
    def _deg(slider) -> float:
        """スライダー値(0.5°単位) → 角度[度]に変換する。"""
        return slider.value() / ANGLE_UNIT

    # ------------------------------------------------------
    # ハードウェア初期化（カメラ必須 / リレー・モーターは任意）
    # ------------------------------------------------------
    def _init_hardware(self):
        # カメラ
        self.manager = cam_ctr.CameraManager()
        if not self.manager.init_cameras():
            self._status("!! カメラの初期化に失敗しました。接続を確認してください。")
            self.controllers = []
        else:
            self.controllers = self.manager.controllers
            for c in self.controllers:
                c.delay_seconds = CAMERA_DELAYS.get(c.name, 0.0)
            self.manager.start_all_get_frame()

        # YOLO（dcr=None でログは汚さない）
        try:
            self.detector = yolo_ctr.YoloDetector(dcr=None, cameras=self.controllers)
        except Exception as e:
            self.detector = None
            self._status(f"!! YOLO の初期化に失敗しました: {e}")

        # 検知ループは本番(MainWindow)と同じくシグナル駆動。
        #   カメラ別ワーカースレッドが frame_ready(表示用)・final_ready(確定結果)を emit する。
        #   running の切替は self.detector.set_running() で行い、ここではポーリングしない。
        if self.detector is not None:
            self.detector.signals.frame_ready.connect(self._on_frame_ready)
            self.detector.signals.final_ready.connect(self._on_finalized)
            self.detector.start_workers()
            self.detector.set_running(False)  # 「開始」が押されるまで推論はしない

        # リレー
        self.relay = r_ctr.RelayController()
        try:
            self.relay_ok = self.relay.init()
        except Exception as e:
            self.relay_ok = False
            print(f"!! リレー初期化でエラー: {e}")
        if not self.relay_ok:
            self._status("!! リレー未接続。弁は開かず角度の確認のみになります。")

        # モーター（非常停止シグナル経由でGUIスレッドへ）
        self.motor_signals = MotorSignals()
        self.motor_signals.estop.connect(self._on_estop)
        self.motor_signals.estop_cleared.connect(self._on_estop_cleared)
        self.motor = motor_ctr.MotorSerial(
            port=SERIAL_PORT,
            on_estop=self.motor_signals.estop.emit,
            on_estop_cleared=self.motor_signals.estop_cleared.emit,
        )
        try:
            self.motor_ok = self.motor.init()
        except Exception as e:
            self.motor_ok = False
            print(f"!! モーター初期化でエラー: {e}")
        if not self.motor_ok:
            self._status("!! モーター未接続。手動で果実を流してください。")

        if self.controllers and self.detector is not None:
            self._status("準備完了。安全を確認して『開始』を押してください。")

    # ------------------------------------------------------
    # 検知ループ（YoloDetector のカメラ別ワーカースレッドからのシグナルで駆動）
    # ------------------------------------------------------
    @Slot(str, object)
    def _on_frame_ready(self, cam_name, rgb_image):
        """frame_ready: ワーカーが作った表示用フレーム（RGB・640角・帯線描画済み）を表示する。
        cvtColor・帯線描画はワーカー側で済んでいるので、ここは QImage 化と setPixmap のみ
        （module_main_window_JP.MainWindow._on_frame_ready と同じ処理）。"""
        lbl = self.cam_labels.get(cam_name)
        if lbl is None:
            return
        h, w, ch = rgb_image.shape
        img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        lbl.setPixmap(pix)

    @Slot(object)
    def _on_finalized(self, result):
        """確定＝本番の move 起点。ライブ角度の待機時間で対応する弁を発火する。"""
        label = result.label_name
        if label == "healthy":
            channel, angle, name = r_ctr.RelayChannel.TRANSPORT, self._deg(self.transport_slider), "運搬弁"
        else:
            channel, angle, name = r_ctr.RelayChannel.REMOVE, self._deg(self.remove_slider), "除去弁"

        wait = sec_per_rotation(self.speed_spin.value()) * (angle / 360.0)
        self._status(f"確定 id={result.id} [{label}] → {name} 角度{angle:.1f}° 待機{wait:.3f}s")

        if self.relay_ok:
            self.pool.start(RelayFireWorker(self.relay, channel, wait))

    # ------------------------------------------------------
    # 操作系
    # ------------------------------------------------------
    def _toggle_running(self):
        if not self.running:
            if not self.controllers or self.detector is None:
                self._status("!! カメラ/YOLO が未初期化のため開始できません。")
                return
            self.running = True
            self.btn_start.setText("停止")
            self.detector.set_running(True)
            if self.motor_ok:
                try:
                    self.motor.set_speed(self.speed_spin.value())
                    self.motor.rotate()
                except Exception as e:
                    print(f"!! 回転開始でエラー: {e}")
            self._status("計測中。果実を流してください（健全果=運搬弁 / 被害果=除去弁）。")
        else:
            self._stop_running()

    def _stop_running(self):
        self.running = False
        self.btn_start.setText("開始")
        if self.detector is not None:
            self.detector.set_running(False)
        if self.motor_ok:
            try:
                self.motor.stop()
            except Exception as e:
                print(f"!! 回転停止でエラー: {e}")
        if self.relay_ok:
            try:
                self.relay.stop()
            except Exception:
                pass
        self._status("停止しました。")

    def _on_speed_changed(self, _val):
        self._refresh_wait_labels()
        # 回転中なら新しい速度を即時反映
        if self.running and self.motor_ok:
            try:
                self.motor.set_speed(self.speed_spin.value())
                self.motor.rotate()
            except Exception as e:
                print(f"!! 速度反映でエラー: {e}")

    def _refresh_wait_labels(self):
        sec = sec_per_rotation(self.speed_spin.value())
        rw = sec * (self._deg(self.remove_slider) / 360.0)
        tw = sec * (self._deg(self.transport_slider) / 360.0)
        self.remove_wait_lbl.setText(f"角度 {self._deg(self.remove_slider):5.1f}°  →  待機 {rw:.3f} s")
        self.transport_wait_lbl.setText(f"角度 {self._deg(self.transport_slider):5.1f}°  →  待機 {tw:.3f} s")

    def _save(self):
        out = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "speed_at_calibration": self.speed_spin.value(),
            "remove":    {"angle": self._deg(self.remove_slider)},
            "transport": {"angle": self._deg(self.transport_slider)},
        }
        try:
            with open(r_ctr.RELAY_CAL_PATH, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            self._status(f"保存しました: 除去={out['remove']['angle']:.1f}° / "
                         f"運搬={out['transport']['angle']:.1f}°  本番が次回起動時に適用します。")
        except Exception as e:
            self._status(f"!! 保存に失敗: {e}")

    # --- 非常停止 ---
    def _on_estop(self):
        self._status("!! 非常停止(ESTOP)を検知。回転・発火を停止します。")
        self._stop_running()

    def _on_estop_cleared(self):
        self._status("非常停止が解除されました。『開始』で再開できます。")

    def _status(self, msg):
        print(msg)
        if hasattr(self, "status_lbl"):
            self.status_lbl.setText(msg)

    # ------------------------------------------------------
    # 終了処理
    # ------------------------------------------------------
    def closeEvent(self, event):
        try:
            if self.motor_ok:
                self.motor.stop()
                self.motor.close()
        except Exception:
            pass
        try:
            if self.relay_ok:
                self.relay.stop()
                self.relay.close()
        except Exception:
            pass
        try:
            if self.detector is not None:
                self.detector.close()
        except Exception:
            pass
        try:
            if self.controllers:
                self.manager.stop_all_get_frame()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = CalibrationWindow()
    win.showMaximized()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
