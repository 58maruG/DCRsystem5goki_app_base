# -------------------------------------------------
# module_main_window_JP.py
#   main.py から分離したウィンドウ／制御ロジック一式。
#   main側は「起動するだけ」にするため、GUIクラス・バックグラウンド処理・
#   イベントハンドラはすべてこちらに置く。中身は分離前の main_5goki_JP_v3.py と同一。
# -------------------------------------------------
import os
import json
import time
from datetime import datetime

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Slot, Qt, QRunnable, QThreadPool, QTimer, Signal, QObject
from PySide6.QtGui import QImage, QPixmap

# ロギング（コンソール統一）
import log_config
log = log_config.get_logger("main")

# データログ（cycle / health / events / detections）
from dcr_logger import DCRLogger, Telemetry
import telemetry_sources

# GUIモジュール
import module_gui_JP

# 制御モジュール
import module_cameras_5goki as cam_ctr
import module_relay as r_ctr
import module_patlite as p_ctr
import module_yolo as yolo_ctr
import module_motor_serial as motor_ctr

# Arduinoのシリアルポート。None で自動検出。うまくいかない場合は "COM3" 等を直接指定する
SERIAL_PORT = None


# クラス色チップ用ユーティリティ。文字色・枠線とも全チップ共通で黒に統一する（視認性を揃えるため）。
#   引数 bg_hex は将来の配色調整用に残す（現状は未使用）。
def _chip_colors(bg_hex: str):
    return "#000000", "border:1px solid #000000;"


# 色付きチップ＋信頼度を1つ組み立てる（内側テーブル）。conf_color で信頼度の文字色を指定。
def _chip_inner_html(label: str, conf: float, conf_color: str) -> str:
    info = module_gui_JP.CLASS_DISPLAY.get(label, {"jp": str(label), "color": "#CCCCCC"})
    bg   = info["color"]
    pct  = int(round(conf * 100))
    text_color, border = _chip_colors(bg)
    return (
        f'<table cellspacing="0" cellpadding="3"><tr>'
        f'<td bgcolor="{bg}" align="center" '
        f'style="color:{text_color}; {border} font-size:18px; font-weight:bold;">'
        f'&nbsp;{info["jp"]}&nbsp;</td>'
        f'<td style="color:{conf_color}; font-size:18px; font-weight:bold;">：{pct}%</td>'
        f'</tr></table>'
    )


# 確定クラス用チップ（クラス名の矩形の外・左に◎マーク＋信頼度）。
#   カメラ列で確定クラスを検出したセルに使う。◎は枠外に独立配置する。
def _chip_confirmed_html(label: str, conf: float) -> str:
    info = module_gui_JP.CLASS_DISPLAY.get(label, {"jp": str(label), "color": "#CCCCCC"})
    bg   = info["color"]
    pct  = int(round(conf * 100))
    text_color, border = _chip_colors(bg)
    return (
        f'<table cellspacing="0" cellpadding="3"><tr>'
        f'<td style="color:#000000; font-size:22px; font-weight:bold; padding-right:3px;">◎</td>'
        f'<td bgcolor="{bg}" align="center" '
        f'style="color:{text_color}; {border} font-size:18px; font-weight:bold;">'
        f'&nbsp;{info["jp"]}&nbsp;</td>'
        f'<td style="color:#000000; font-size:18px; font-weight:bold;">：{pct}%</td>'
        f'</tr></table>'
    )


# GUI履歴表のカメラ別列。左から (内部キー, 表示ヘッダ) の順。ユーザー指定の in→out→top→under。
HISTORY_CAM_COLUMNS = [
    ("cam_inside",  "inside"),
    ("cam_outside", "outside"),
    ("cam_top",     "top"),
    ("cam_under",   "under"),
]

# ==========================================================
# 汎用バックグラウンドタスク用クラス
# ==========================================================
class TaskWorker(QRunnable):
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.func(*self.args, **self.kwargs)
        except Exception as e:
            log.error("バックグラウンドタスク例外: %s", e)

# ==========================================================
# Arduinoの非常停止通知を受け取るためのシグナル
#   受信スレッド(別スレッド)から emit し、GUIスレッドのスロットで処理する。
#   （Qtのシグナルはスレッド間でキューイングされるため安全）
# ==========================================================
class MotorSignals(QObject):
    estop = Signal()           # 非常停止 作動
    estop_cleared = Signal()   # 非常停止 解除
    standalone = Signal()      # 単体モード 移行
    pc_mode = Signal()         # PC連携モード 復帰

# ==========================================================
# スタートアップウィンドウ
# ==========================================================
class StartupWindow(module_gui_JP.StartupWindowUI):
    def __init__(self):
        super().__init__()
        self.button_start.clicked.connect(self.launch_main)

    def launch_main(self):
        self.main_window = MainWindow()
        self.main_window.showFullScreen()
        self.close()

# ==========================================================
# サブウィンドウ
# ==========================================================
class SubWindow(module_gui_JP.SubWindowUI):
    def __init__(self, parent_window, initial_speed):
        super().__init__()
        self.button_up_speed.clicked.connect(self.on_up_speed)
        self.button_down_speed.clicked.connect(self.on_down_speed)
        self.button_back.clicked.connect(self.go_back)

        self.parent_window = parent_window
        self.current_speed = initial_speed
        self.update_speed_ui()

    def update_speed_ui(self):
        self.label_current_speed.setText(str(self.current_speed))

        if self.current_speed >= 10:
            self.button_up_speed.set_locked(True)
        else:
            self.button_up_speed.set_locked(False)

        if self.current_speed <= 1:
            self.button_down_speed.set_locked(True)
        else:
            self.button_down_speed.set_locked(False)

    @Slot()
    def on_up_speed(self):
        if self.current_speed < 10:
            self.current_speed += 1
            self.update_speed_ui()

    @Slot()
    def on_down_speed(self):
        if self.current_speed > 1:
            self.current_speed -= 1
            self.update_speed_ui()

    @Slot()
    def go_back(self):
        self.parent_window.saved_speed = self.current_speed
        # モード表示パネルのパルス速度を即時反映
        self.parent_window.refresh_mode_panel()
        self.close()

# ==========================================================
# カメラエラーウィンドウ
# ==========================================================
class CameraErrorWindow(module_gui_JP.CameraErrorWindowUI):
    # 復旧処理（ブロッキング）の完了をGUIスレッドへ返すシグナル。
    #   別スレッドの worker から emit し、GUIスレッドのスロットで後処理する。
    #   ok=成否 / info=失敗時に表示するメッセージ（成功時は None）。
    recovery_finished = Signal(bool, object)

    def __init__(self, parent_window, lost_cam_name):
        super().__init__(lost_cam_name)
        self.parent_window = parent_window
        self.button_continue.clicked.connect(self.attempt_recovery)
        self.recovery_finished.connect(self._on_recovery_finished)

    def attempt_recovery(self):
        log.info("復旧プロセス開始...")
        self.button_continue.setEnabled(False)
        self.label_cams.setText("復旧中...")
        # カメラ再初期化は数秒〜十数秒ブロックするため、GUIスレッドで直接実行すると
        #   タッチパネルが完全に無反応になる。別スレッドへ逃がし、結果は
        #   recovery_finished シグナルでGUIスレッドへ返す（UIは反応を維持する）。
        self.parent_window.run_in_background(self._recovery_worker)

    def _recovery_worker(self):
        # ※ここはバックグラウンドスレッド。Qtウィジェットには一切触れないこと。
        #   ブロッキング処理を try/except で必ず包み、例外時も結果を emit して
        #   「続行」ボタンが無効のまま固定される事故を防ぐ。
        try:
            self.parent_window.cameras.stop_all_get_frame()
            time.sleep(1.0)

            if not self.parent_window.cameras.init_cameras():
                self.recovery_finished.emit(
                    False, "カメラを開けませんでした。\nUSBケーブルを確認してください。")
                return

            connected_names = [c.name for c in self.parent_window.cameras.controllers]
            required_names = [name for _, name in cam_ctr.TARGET_SERIALS]
            missing = set(required_names) - set(connected_names)

            if missing or len(connected_names) != 4:
                msg = f"不完全: {len(connected_names)}/4 台のカメラ\n"
                if missing:
                    msg += f"未接続: {', '.join(missing)}"
                self.recovery_finished.emit(False, msg)
                return

            self.recovery_finished.emit(True, None)
        except Exception as e:
            log.error("復旧処理中に例外が発生しました: %s", e)
            self.recovery_finished.emit(
                False, f"復旧中にエラーが発生しました。\n再試行してください（{type(e).__name__}）")

    @Slot(bool, object)
    def _on_recovery_finished(self, ok, info):
        # ※ここはGUIスレッド。ウィジェット操作・カメラ再開・タイマー再開はここで行う。
        if not ok:
            self.label_cams.setText(info if isinstance(info, str) else "復旧に失敗しました。")
            self.button_continue.setEnabled(True)
            return

        log.info("4台全てのカメラが正常に再オープンされました。")
        pw = self.parent_window

        # 次回のカメラエラーを再び処理できるようフラグをリセット
        pw._camera_error_handled = False

        for controller in pw.cameras.controllers:
            controller.signals.connection_lost.connect(pw.handle_camera_error)

        pw.cameras.start_all_get_frame()
        # 復旧で作り直された新カメラ obj にワーカーを束ね直す（旧 obj を掴んだままだと表示が固まる）
        if hasattr(pw, "detector") and pw.detector is not None:
            pw.detector.restart_workers()
            pw.detector.set_running(pw.toggle_switch.isChecked())
        pw.run_in_background(pw.relay.stop)

        # 復旧完了 → エラー固定を解除し、現在の状態に合わせてGUI/パトライトを復元する
        pw._error_active = False
        if pw._estop_active:
            # 緊急停止中のまま復旧した場合はestop表示を優先する
            pw.update_status_display("estop")
            if not pw._standalone_active:
                pw.run_in_background(pw.patlite.set_system_state, p_ctr.SystemState.ESTOP)
        elif pw._standalone_active:
            pw.run_in_background(pw.patlite.set_system_state, p_ctr.SystemState.STANDALONE)
        else:
            pw.update_status_display("stop")
            pw.run_in_background(pw.patlite.set_system_state, p_ctr.SystemState.STANDBY)

        pw.timer.start(50)
        self.close()

# ==========================================================
# メインウィンドウ
# ==========================================================
class MainWindow(module_gui_JP.MainWindowUI):
    def __init__(self):
        super().__init__()
        self.thread_pool = QThreadPool()

        self._init_logging()
        self._init_operation_lock_flags()
        self._init_devices()
        self._init_yolo_detector()
        self._start_capture()
        self._wire_detector_signals()
        self._wire_gui_events()
        self._init_timers()
        self._init_startup_display()

    # --- データログ・テレメトリキャッシュの初期化 ---
    def _init_logging(self):
        # データログ開始（書き込みスレッドのみ先に起動。startup イベントは
        #   モデルの精度などが揃う YOLO 初期化後に記録する）
        self.dcr = DCRLogger(base_dir="logs", line="5goki")
        self.dcr.start(log_startup=False)

        # HW環境スナップショット用キャッシュ。health(1Hz) が最新値を入れ、cycle が読む。
        #   ホットパス（cycle）で nvml を直接叩かないための仕組み（最大約1秒前の値）。
        self.tele = Telemetry()

        self.history_data = []
        self.current_id = 1

    # --- 操作系ロックの要因フラグ（非常停止・単体モード・カメラエラー）初期化 ---
    def _init_operation_lock_flags(self):
        # どちらかが立っていれば運転トグル・設定ボタンを無効化する。
        self._estop_active = False
        self._standalone_active = False
        # カメラエラー中フラグ。True の間はパトライトとGUIステータスをerrorで固定する。
        self._error_active = False
        # エラーを起こしたカメラ名。単体モード中に発生した場合はPC復帰時に使用。
        self._error_cam_name = None
        # カメラエラー処理済みフラグ（複数台が同時に切断した場合の重複処理防止）
        self._camera_error_handled = False

    # --- デバイス接続（パトライト・リレー・Arduino・カメラ）---
    def _init_devices(self):
        # --- デバイス接続 (接続中はYELLOW) ---
        self.patlite = p_ctr.PatliteController()
        if not self.patlite.init():
            log.error("パトライトの接続に失敗しました")
            self.close()

        # 初期化中を表示
        self.patlite.set_system_state(p_ctr.SystemState.INITIALIZING)

        self.relay = r_ctr.RelayController()
        if not self.relay.init():
            log.error("リレーボードの接続に失敗しました")
            self.close()

        # Arduino(ターンテーブル)接続。失敗してもGUIは起動を続ける
        #   非常停止通知はシグナル経由でGUIスレッドへ渡す
        self.motor_signals = MotorSignals()
        self.motor_signals.estop.connect(self.handle_estop)
        self.motor_signals.estop_cleared.connect(self.handle_estop_cleared)
        self.motor_signals.standalone.connect(self.handle_standalone)
        self.motor_signals.pc_mode.connect(self.handle_pc_mode)
        self.motor = motor_ctr.MotorSerial(
            port=SERIAL_PORT,
            on_estop=self.motor_signals.estop.emit,
            on_estop_cleared=self.motor_signals.estop_cleared.emit,
            on_standalone=self.motor_signals.standalone.emit,
            on_pc_mode=self.motor_signals.pc_mode.emit,
            dcr=self.dcr,
        )
        if not self.motor.init():
            log.warning("Arduino(モータ)の接続に失敗しました（GUIは起動を継続します）")

        self.cameras = cam_ctr.CameraManager()
        if not self.cameras.init_cameras():
            log.error("カメラの接続に失敗しました")
            self.close()

        # カメラエラーハンドラの設定
        for controller in self.cameras.controllers:
            controller.signals.connection_lost.connect(self.handle_camera_error)

        # カウント用辞書の初期化
        self.detection_counts = {
            "healthy": 0, "twin": 0, "unripe": 0, "mold": 0, "stemcrack": 0, "birddamage": 0,
            "malformation": 0, "crack": 0, "wilt": 0, "suturecrack": 0, "brownrot": 0, "blacktwin": 0,
            "insect": 0, "kasure": 0
        }
        self.update_stats_display()

    # --- YOLO検出器の初期化とstartupイベント記録 ---
    def _init_yolo_detector(self):
        # YOLO初期化（モデルパスは module_yolo_csv3.py の MODEL_PATH で一元管理）
        #   dcr を渡し、detections ログを新ロガーへ一本化する
        self.detector = yolo_ctr.YoloDetector(dcr=self.dcr, cameras=self.cameras.controllers)

        # startup イベント（比較の前提を固定する静的情報）。
        #   HWダウングレード判断のため GPU 名・総VRAM・モデル構成を残す。
        self.dcr.log_startup(
            model=os.path.basename(yolo_ctr.MODEL_PATH),
            imgsz=yolo_ctr.YOLO_IMG_SIZE,
            precision=self.detector.model_precision(),
            batch=1,
            **telemetry_sources.gpu_static_info(),
        )
        # メモリリーク局所化のための tracemalloc 定期スナップショット（約60秒ごと）
        self.dcr.start_mem_snapshot(60.0)

    # --- 初期スピード・カメラ表示遅延の適用とキャプチャ開始 ---
    def _start_capture(self):
        # スピード初期値設定とカメラ遅延の初期適用
        self.saved_speed = 6  # デフォルト速度
        self.update_camera_delays()

        self.cameras.start_all_get_frame()

    # --- 検出器シグナル → GUIスレッドへの配線とワーカー起動 ---
    def _wire_detector_signals(self):
        # 検出器シグナル → GUIスレッドへのマーシャリング（Qtの既定でキュー接続＝GUIスレッド実行）。
        #   frame_ready: カメラ別ワーカーが作った表示フレームを setPixmap する。
        #   final_ready: 個体確定結果を process_final_result（リレー制御・履歴表示）へ渡す。
        self.detector.signals.frame_ready.connect(self._on_frame_ready)
        self.detector.signals.final_ready.connect(self.process_final_result)
        # カメラ起動後にカメラ別処理ワーカーを起動する（運転は set_running で切替）
        self.detector.start_workers()

    # --- GUIイベント（トグル・設定・電源ボタン）の配線 ---
    def _wire_gui_events(self):
        self.toggle_switch.toggled.connect(self.on_main_toggled)
        self.button_setting.clicked.connect(self.on_setting_button)
        self.button_power.clicked.connect(self.on_power_bottom)

    # --- タイマー初期化（表示互換用・ヘルスログ）---
    def _init_timers(self):
        # 表示更新はカメラ別ワーカーの frame_ready シグナルが駆動するため、
        #   旧 QTimer 駆動の update_video_feeds は使わない。timer オブジェクトは
        #   エラー/終了処理の self.timer.stop() 互換のため生成だけしておく（未開始）。
        self.timer = QTimer(self)

        # ヘルスlog（≈1Hz）。システムの健全性スナップショットを残す
        self.health_timer = QTimer(self)
        self.health_timer.timeout.connect(self.update_health)
        self.health_timer.start(1000)

    # --- 起動時のGUI/パトライト初期表示と非常停止状態の問い合わせ ---
    def _init_startup_display(self):
        # 全デバイス接続完了 → 待機中 (RED)
        self.patlite.set_system_state(p_ctr.SystemState.STANDBY)

        # 起動時は PCモード として表示
        self.label_mode.setText("PCモード")
        self.label_mode.setStyleSheet(module_gui_JP.LABEL_MODE_STYLE_PC)

        # モード表示パネルの初期化（パルス速度・モデル名）。
        #   モデル名は読込中の重みファイル名を表示（将来モード切替の値に差し替え予定）。
        self.refresh_mode_panel()
        self.label_model.setText(os.path.basename(yolo_ctr.MODEL_PATH))

        # ステータスバー初期状態（停止中）。
        #   error状態の赤点滅用タイマーを先に用意してから初期表示する。
        self._error_blink_on = True
        self._status_blink_timer = QTimer(self)
        self._status_blink_timer.timeout.connect(self._toggle_status_blink)
        self.update_status_display("stop")

        # 起動時点の非常停止状態を問い合わせる。
        #   応答(ESTOP/ESTOP_CLEARED)は受信スレッド経由で handle_estop /
        #   handle_estop_cleared を呼び、トグル有効/無効や表示を同期する。
        #   ※この後にイベントループへ入ってから処理されるため、上のSTANDBY
        #     設定より後に反映され、非常停止中ならESTOP表示が優先される。
        self.motor.query_estop()

    # --- カメラ表示遅延をスピードに合わせて更新する関数 ---
    def update_camera_delays(self):
        config_path = os.path.join(os.path.dirname(__file__), "json", "delay_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        delays = {"cam_under": float(loaded["cam_under"]), "cam_inside": float(loaded["cam_inside"])}
        log.info("遅延設定を読み込みました: %s (更新日時: %s)",
                 config_path, loaded.get("updated_at", "不明"))

        log.info("表示遅延を設定  cam_under=%.3fs / cam_inside=%.3fs",
                 delays["cam_under"], delays["cam_inside"])

        for controller in self.cameras.controllers:
            if controller.name in delays:
                controller.delay_seconds = delays[controller.name]

    # --- カメラ別ワーカーからの表示フレームをGUIに反映する（frame_ready スロット）---
    #   cvtColor・帯線描画はワーカー側で済んでいるので、ここは QImage 化と setPixmap のみ。
    #   rgb_image は emit されたRGB配列（このスロット実行中は生存）。QPixmap.fromImage で複製される。
    @Slot(str, object)
    def _on_frame_ready(self, cam_name, rgb_image):
        target_label = {
            "cam_inside":  self.cam_in,
            "cam_outside": self.cam_out,
            "cam_under":   self.cam_under,
            "cam_top":     self.cam_top,
        }.get(cam_name)
        if target_label is None:
            return
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            target_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        target_label.setPixmap(scaled_pixmap)

    # --- バックグラウンドで渡された関数を実行するヘルパー関数 ---
    def run_in_background(self, func, *args, **kwargs):
        worker = TaskWorker(func, *args, **kwargs)
        self.thread_pool.start(worker)

    # --- リレーを動作させ、排出タイムスタンプ・結果を cycle ログへ後追い記録する ---
    def _relay_and_log(self, channel, speed, cycle_id):
        result = self.relay.move(channel, speed)

        def _fmt(t):
            return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if t is not None else ""

        self.dcr.cycle(
            cycle_id=cycle_id,
            planned_eject_ts=_fmt(result.get("planned_eject_ts")),
            outcome_flag=1 if result.get("ok") else 0,
            **{"eject_delay[ms]": result.get("eject_delay_ms", "")},
        )

    # --- ヘルスログ更新（≈1Hz）---
    #   QTimer コールバック（GUI スレッド）から即座に返し、重いI/O処理はバックグラウンドへ逃がす。
    #   cpu_temp_c() が localhost:8085 への HTTP 接続(timeout=0.5s)を試みるため、
    #   LHM 未起動時に GUI スレッドで直接呼ぶと 500ms のフリーズが毎秒発生する。
    def update_health(self):
        self.run_in_background(self._do_health_check)

    def _do_health_check(self):
        # ※ここはバックグラウンドスレッド。Qtウィジェットには触れないこと。
        #   Telemetry.update/snapshot はロック済み、dcr.health は put_nowait のため安全。
        try:
            s = telemetry_sources.sys_stats()
            g = telemetry_sources.gpu_stats()           # GPU温度/使用率/クロック + VRAM/電力
            pm = telemetry_sources.proc_mem()           # 自プロセスRSS（リーク検出）
            tv = telemetry_sources.torch_vram()         # torch VRAM 内訳（リーク検出）

            # cycle 軸へ写すHW負荷系をキャッシュ更新（gpu_* は g、cpu_util は s 由来）。
            #   None は Telemetry.update が弾くので、取得できた値だけが残る。
            self.tele.update(**g, **{"cpu_util[%]": s.get("cpu_util[%]")})

            # フレームバッファ滞留数（表示遅延キューの合計）。詰まりの目安
            try:
                queue_depth = sum(len(c.frame_queue) for c in self.cameras.controllers)
            except Exception:
                queue_depth = ""

            # 累積サイクル数（その時点の最新 cycle_id）。「MB/サイクル」算出の正規化に使う
            cycles_total = getattr(self.detector, "current_cherry_id", "") if hasattr(self, "detector") else ""

            self.dcr.health(
                **{"cpu_temp": telemetry_sources.cpu_temp_c(),
                   "queue_depth[n]": queue_depth,
                   "cycles_total[n]": cycles_total},
                **s, **g, **pm, **tv,
            )
        except Exception as e:
            log.debug("health 取得に失敗: %s", e)

    # --- 設定ボタン押下イベント ---
    @Slot()
    def on_setting_button(self):
        self.settings_window = SubWindow(parent_window=self, initial_speed=self.saved_speed)
        self.settings_window.show()

    # --- カメラエラー発生時の処理 ---
    @Slot(str)
    def handle_camera_error(self, cam_name):
        # 複数台が同時に切断した場合、2台目以降はログだけ残して返る
        if self._camera_error_handled:
            last_id = getattr(self.detector, "current_cherry_id", "") if hasattr(self, "detector") else ""
            self.dcr.error("camera_error_additional", cam=cam_name, last_cycle_id=last_id,
                           msg=f"[追加カメラ切断] カメラ '{cam_name}' も接続が切れました。")
            return
        self._camera_error_handled = True

        # イベント化（コンソール + events.jsonl 両方）
        last_id = getattr(self.detector, "current_cherry_id", "") if hasattr(self, "detector") else ""
        self.dcr.error("camera_error", cam=cam_name, last_cycle_id=last_id,
                       msg=f"[緊急停止] カメラ '{cam_name}' の接続が切れました。")

        # GUIのトグルをOFFにする（→ detector.set_running(False)。カメラ別ワーカーは
        #   推論・状態更新を止め、生映像の表示のみになる。カメラ停止後は新フレームが来ず
        #   表示は最後のフレームで固まる）
        self.toggle_switch.setChecked(False)

        # 旧 QTimer は未使用（生成のみ）。stop() は互換のため残す no-op。
        self.timer.stop()

        # カメラ停止はGUIスレッドをブロックしないようバックグラウンドへ逃がす。
        #   ワーカーは停止中カメラの get_next_frame がタイムアウトしてアイドルになるだけで安全。
        self.run_in_background(self.cameras.stop_all_get_frame)

        # エラーフラグとカメラ名を記録（単体モード中でも必ず保持する）
        self._error_active = True
        self._error_cam_name = cam_name

        if self._standalone_active:
            # 単体モード中はGUI/パトライトを変更しない。PCモード復帰時に反映する。
            return

        # PCモード中: RED点滅 + エラーウィンドウを即時表示する
        self.update_status_display("error")
        self.run_in_background(self.patlite.set_system_state, p_ctr.SystemState.ERROR)

        self.error_win = CameraErrorWindow(self, cam_name)
        self.error_win.show()

    # --- 操作系(運転トグル・設定ボタン)のロック状態を反映する関数 ---
    #   非常停止中 or 単体モード中はロック。それ以外は運転状態に応じて戻す。
    #   複数の要因(estop/standalone)を一元的に判定し、解除時の取りこぼしを防ぐ。
    def _refresh_operation_lock(self):
        locked = self._estop_active or self._standalone_active or self._error_active
        # 運転トグル
        self.toggle_switch.set_locked(locked)
        # 設定ボタン: ロック中、または運転中(トグルON)は触らせない
        self.button_setting.set_locked(locked or self.toggle_switch.isChecked())
        # 電源ボタン: 運転中(トグルON)は誤操作防止のためロック
        self.button_power.set_locked(self.toggle_switch.isChecked())

    # --- 非常停止 作動時の処理 (Arduinoから "ESTOP" を受信) ---
    #   人がスイッチを押した＝監視中のためブザーは鳴らさない(赤点滅)。
    #   モータはArduino側で物理遮断・停止済み。PC側はGUIと検査を止める。
    @Slot()
    def handle_estop(self):
        self.dcr.error("estop", source="switch", trigger="NC_open",
                       msg="[非常停止] スイッチが押されました。")

        # on_main_toggled(False) が呼ばれなくなったので、必要な停止処理を手動で行う
        self.toggle_switch.setChecked(False)
        self.motor.stop()
        self.run_in_background(self.relay.stop)

        # 非常停止中はトグルを無効化し、押しても反応しないようにする
        # (NC接点が切れている＝危険状態のまま起動操作を受け付けないため)
        self._estop_active = True
        self._refresh_operation_lock()
        self.update_status_display("estop")

        # 非常停止中表示: 紫点灯・ブザーなし（単体モード中またはカメラエラー中は変更しない）
        if not self._standalone_active and not self._error_active:
            self.run_in_background(self.patlite.set_system_state, p_ctr.SystemState.ESTOP)

    # --- 非常停止 解除時の処理 (Arduinoから "ESTOP_CLEARED" を受信) ---
    #   PCモードでは自動再開せず、待機中に戻すだけ。
    #   再開はオペレータが改めてトグルONする(=指令送信)。
    @Slot()
    def handle_estop_cleared(self):
        self.dcr.info("estop_cleared", msg="[非常停止] 解除されました。待機中に戻ります。")

        # トグルを再び有効化し、操作を受け付けられるようにする
        # (ただし単体モード中ならロックは据え置かれる)
        self._estop_active = False
        self._refresh_operation_lock()
        self.update_status_display("stop")

        # カメラエラー中はパトライトを変更しない。それ以外は単体モード/待機中に戻す。
        if not self._error_active:
            if self._standalone_active:
                self.run_in_background(self.patlite.set_system_state, p_ctr.SystemState.STANDALONE)
            else:
                self.run_in_background(self.patlite.set_system_state, p_ctr.SystemState.STANDBY)

    # --- 単体モード 移行時の処理 (Arduinoから "STANDALONE" を受信) ---
    #   Arduinoが単独でターンテーブルを動かすモード。PCの操作は受け付けない。
    #   PC側は推論・リレーを止めて画面をロックするが、モータへ停止(S)は送らない
    #   （Arduinoの単体動作を妨げないため）。
    @Slot()
    def handle_standalone(self):
        self.dcr.info("standalone", msg="[単体モード] Arduinoが単体モードに移行しました。PC操作をロックします。")

        # 先にフラグを立てる。これで on_main_toggled(False) がモータへSを送らない。
        self._standalone_active = True

        # 運転中なら停止処理(推論停止・リレー停止)を走らせる。
        # 既にOFFなら toggled シグナルは出ないため、リレー停止は明示的にも行う。
        self.toggle_switch.setChecked(False)
        self.run_in_background(self.relay.stop)

        # 状態ラベルに単体モードを表示
        self.label_toggle_status.setText("単体モード中")
        self.label_toggle_status.setStyleSheet("""
            font-family: "Meiryo"; font-size: 30px; font-weight: bold;
            color: #FF8C00; qproperty-alignment: 'AlignCenter';
        """)

        # モードラベルを「単体モード」に更新
        self.label_mode.setText("単体モード")
        self.label_mode.setStyleSheet(module_gui_JP.LABEL_MODE_STYLE_STANDALONE)

        # ブロッキングオーバーレイでGUI全体をロック（電源ボタンは前面に維持）
        self.blocking_overlay.show()
        self.blocking_overlay.raise_()
        self.button_power.raise_()

        # 単体モード中はパトライトを消灯（カメラエラー中でも単体モードを優先する）
        self.run_in_background(self.patlite.set_system_state, p_ctr.SystemState.STANDALONE)

        # カメラエラーウィンドウが開いていれば閉じる（PCモード復帰時に再表示する）
        if hasattr(self, 'error_win') and self.error_win.isVisible():
            self.error_win.close()

    # --- PC連携モード 復帰時の処理 (Arduinoから "PC_MODE" を受信) ---
    #   単体モードを抜けたらロックを解除して待機中に戻す。
    #   自動再開はせず、オペレータが改めてトグルONする。
    @Slot()
    def handle_pc_mode(self):
        self.dcr.info("pc_mode", msg="[単体モード] 解除されました。PC連携モードに戻ります。")

        self._standalone_active = False

        # 状態ラベルを停止中表示に戻す
        self.label_toggle_status.setText("停止中")
        self.label_toggle_status.setStyleSheet("""
            font-family: "Meiryo"; font-size: 30px; font-weight: bold;
            color: #888888; qproperty-alignment: 'AlignCenter';
        """)

        # モードラベルを「PCモード」に戻す
        self.label_mode.setText("PCモード")
        self.label_mode.setStyleSheet(module_gui_JP.LABEL_MODE_STYLE_PC)

        # ブロッキングオーバーレイを非表示にしてGUI操作を再開
        self.blocking_overlay.hide()

        # 操作系のロックを解除（非常停止が生きていればそちらの判定で据え置き）
        self._refresh_operation_lock()

        if self._error_active:
            # 単体モード中にカメラエラーが発生していた場合、ここで反映する
            self.update_status_display("error")
            self.run_in_background(self.patlite.set_system_state, p_ctr.SystemState.ERROR)
            self.error_win = CameraErrorWindow(self, self._error_cam_name or "")
            self.error_win.show()
        elif self._estop_active:
            # 非常停止が生きていればESTOP（紫）を点灯する
            self.run_in_background(self.patlite.set_system_state, p_ctr.SystemState.ESTOP)
        else:
            self.run_in_background(self.patlite.set_system_state, p_ctr.SystemState.STANDBY)

    # --- 電源ボタン押下イベント ---
    @Slot()
    def on_power_bottom(self):
        log.info("電源ボタンが押されました。終了します。")
        self.timer.stop()
        if hasattr(self, 'health_timer'):
            self.health_timer.stop()
        if hasattr(self, '_status_blink_timer'):
            self._status_blink_timer.stop()

        self.patlite.close()    # 消灯してから切断
        self.relay.close()
        self.cameras.stop_all_get_frame()
        self.motor.cleanup()   # 停止・初期化
        self.motor.close()     # シリアルを閉じる
        if hasattr(self, 'detector') and self.detector is not None:
            self.detector.close()

        # データログを flush して shutdown イベントを残す
        if hasattr(self, 'dcr') and self.dcr is not None:
            self.dcr.stop()

        self.close()

    # --- 1個体あたりの処理コスト内訳を cycle ログ用の列に整形する ---
    @staticmethod
    def _cost_columns(result_obj):
        """「検出→前処理→推論→後処理」を4カメラ・全フレーム分合計した値を列に落とす。
        1枚あたりの平均（infer_latency[ms] 等）とは別軸なので列を分けている。
        未計測（None）は空欄にして、集計時に0と区別できるようにする。"""
        def v(name):
            x = getattr(result_obj, name, None)
            return "" if x is None else x

        return {
            "capture_sum[ms]":     v("capture_sum_ms"),
            "capture_frames[n]":   v("capture_frames"),
            "hsv_sum[ms]":         v("hsv_sum_ms"),
            "hsv_frames[n]":       v("hsv_frames"),
            "visible_frames[n]":   v("visible_frames"),
            "preproc_sum[ms]":     v("preproc_sum_ms"),
            "infer_sum[ms]":       v("infer_sum_ms"),
            "infer_count[n]":      v("infer_count"),
            "postproc_sum[ms]":    v("postproc_sum_ms"),
            "total_per_fruit[ms]": v("total_per_fruit_ms"),
            "visible_dur[s]":      v("visible_dur_s"),
            "cycle_dur[s]":        v("cycle_dur_s"),
        }

    # --- HSV通過・YOLO無検出の cycle ログだけを残す（リレーは動かさない）---
    def _log_no_detection_cycle(self, result_obj):
        _hw = self.tele.snapshot()
        self.dcr.cycle(
            cycle_id=result_obj.id,
            hsv_mask_ratio=result_obj.hsv_mask_ratio if result_obj.hsv_mask_ratio is not None else "",
            yolo_no_det_flag=1,
            **{"capture_latency[ms]": result_obj.capture_latency_ms if result_obj.capture_latency_ms is not None else "",
               "frame_dropped[n]":    result_obj.frame_dropped if result_obj.frame_dropped is not None else "",
               "hsv_flag":            result_obj.hsv_pass if result_obj.hsv_pass is not None else ""},
            **self._cost_columns(result_obj),
            **_hw,
        )

    # --- リレー先を判定する（健全果のみ運搬、それ以外は除去）---
    #   健全/障害の判定は result_obj.is_damaged（module_yolo._resolve_quality が確定）で行う。
    #   label_name の "healthy" 一致では判定しないこと（複数カメラでの健全確証が無い場合、
    #   label_name="healthy" でも is_damaged=True になりうるため）。
    #   表示名は module_gui_JP.CLASS_DISPLAY に一元化している。
    def _resolve_channel(self, result_obj):
        disease_name = result_obj.label_name
        info = module_gui_JP.CLASS_DISPLAY.get(disease_name)
        if info is None:
            # CLASS_DISPLAY 未登録クラス（モデルに新クラスが増えた等）。
            #   無言で排出されず・IDだけ欠番になる(=ID飛び)のを防ぐため、
            #   不良として除去し、英語ラベルのまま表示・警告ログを残す。
            log.warning("未登録クラス '%s' を検出。不良として除去します。CLASS_DISPLAYへの登録を推奨。",
                        disease_name)
            return r_ctr.RelayChannel.REMOVE, disease_name

        # is_damaged が未確定(None)の場合も安全側でREMOVE扱いにする（通常は到達しない）。
        channel = r_ctr.RelayChannel.TRANSPORT if result_obj.is_damaged is False else r_ctr.RelayChannel.REMOVE
        return channel, info["jp"]

    # --- cycle ログ1行目（確定直後に書けるデータ）を記録する ---
    #   eject_decision: 除去(REMOVE)=1 / 健全運搬(TRANSPORT)=0
    #   capture_latency_ms / frame_dropped / preproc_ms / postproc_ms は
    #   _attach_cycle_stats が result_obj に付与した集計値
    def _log_cycle_row(self, result_obj, obj_id, channel):
        # HW負荷系スナップショット（health 1Hz の最新キャッシュ値）。
        #   この果実を推論した瞬間のGPU/CPU状態を per-cycle 相関用に写す。
        _hw = self.tele.snapshot()
        self.dcr.cycle(
            cycle_id=obj_id,
            hsv_mask_ratio=result_obj.hsv_mask_ratio if result_obj.hsv_mask_ratio is not None else "",
            **{"capture_latency[ms]": result_obj.capture_latency_ms if result_obj.capture_latency_ms is not None else "",
               "frame_dropped[n]":    result_obj.frame_dropped if result_obj.frame_dropped is not None else "",
               "hsv_flag":            result_obj.hsv_pass if result_obj.hsv_pass is not None else "",
               "infer_latency[ms]":   result_obj.infer_avg_ms if result_obj.infer_avg_ms is not None else "",
               "preproc[ms]":         result_obj.preproc_ms if result_obj.preproc_ms is not None else "",
               "postproc[ms]":        result_obj.postproc_ms if result_obj.postproc_ms is not None else "",
               "eject_flag":          1 if channel == r_ctr.RelayChannel.REMOVE else 0,
               "yolo_no_det_flag":    0},
            **self._cost_columns(result_obj),
            **_hw,
        )

    # --- 履歴表示用レコードを組み立てて history_data に積む ---
    def _append_history_record(self, result_obj, obj_id, disease_name):
        # クラス別の最大信頼度（信頼度降順）。空なら確定クラス単体で代替する。
        breakdown = list(result_obj.class_breakdown) or [(disease_name, result_obj.confidence)]
        # 表示順は「確定クラスを左端 → 残りは信頼度降順」にする。
        #   確定クラス(disease_name)は優先ロジックで決まり最大信頼度とは限らないため、
        #   信頼度降順の breakdown から確定クラスを抜き出して先頭へ移す。
        confirmed = [bd for bd in breakdown if bd[0] == disease_name]
        others    = [bd for bd in breakdown if bd[0] != disease_name]
        if confirmed:
            breakdown = confirmed + others
        else:
            # 万一 breakdown に確定クラスが無ければ確定クラス単体を先頭に補う
            breakdown = [(disease_name, result_obj.confidence)] + others
        record = {
            "id": obj_id,
            "final": disease_name,      # 確定クラス（英語ラベル）
            "breakdown": breakdown,     # [(英語ラベル, 信頼度0〜1), ...] 確定クラス先頭・残り降順
            # カメラ別の最高信頼度クラス {cam_name: (英語ラベル, 信頼度0〜1)}。
            #   GUIのカメラ別列(inside/outside/top/under)が読む。検出無しのカメラはキー無し。
            "per_cam": dict(getattr(result_obj, "per_cam_breakdown", {}) or {}),
        }
        self.history_data.append(record)
        if len(self.history_data) > 5:
            self.history_data.pop(0)

    # --- 確定した推論結果をGUIとリレーに反映する関数 ---
    def process_final_result(self, result_obj):
        if not self.toggle_switch.isChecked():
            return

        # HSV通過・YOLO無検出: リレーは動かさず cycle ログだけ残して終了
        if getattr(result_obj, 'yolo_no_det_flag', 0) == 1:
            self._log_no_detection_cycle(result_obj)
            return

        disease_name = result_obj.label_name
        confidence_percent = int(result_obj.confidence * 100)
        obj_id = result_obj.id

        if disease_name in self.detection_counts:
            self.detection_counts[disease_name] += 1

        channel, display_name = self._resolve_channel(result_obj)

        if channel is not None:
            # リレー制御 + 排出ログ（バックグラウンドで実行し、完了後に cycle 補完行を書く）
            self.run_in_background(self._relay_and_log, channel, self.saved_speed, obj_id)

            self._log_cycle_row(result_obj, obj_id, channel)
            self._append_history_record(result_obj, obj_id, disease_name)

            log.info("判定確定 | ID:%03d | 結果:%s | 信頼度:%d%%", obj_id, display_name, confidence_percent)
            self.update_history_display()

    # --- 統計情報を更新する関数 ---
    def update_stats_display(self):
        if hasattr(self, 'label_stats'):
            self.label_stats.setText("入力待機中...")

    # --- モード表示パネルの更新（パルス速度）---
    def refresh_mode_panel(self):
        if hasattr(self, 'label_pulse_speed'):
            self.label_pulse_speed.setText(str(self.saved_speed))

    # --- ステータスバー表示の更新 ---
    #   active に該当する状態を強調（色付き背景）、それ以外は半透明グレーで表示。
    #   配色はパトライトの状態色に合わせる（run=緑 / stop=赤 / estop=紫 / error=赤点滅）。
    def update_status_display(self, active):
        # エラー解消前は他の状態への遷移を無視する
        if self._error_active and active != "error":
            return
        self._current_status = active
        # error はパトライトの RED_BLINK に合わせて点滅させる。それ以外は常時点灯。
        if active == "error":
            self._error_blink_on = True
            if not self._status_blink_timer.isActive():
                self._status_blink_timer.start(500)   # 0.5秒間隔で点滅
        else:
            self._status_blink_timer.stop()
            self._error_blink_on = True
        self._render_status_bar()

    # --- error状態の赤点滅トグル（QTimerから呼ばれる）---
    def _toggle_status_blink(self):
        self._error_blink_on = not self._error_blink_on
        self._render_status_bar()

    # --- 現在のステータスからバーのHTMLを組み立てて表示する ---
    def _render_status_bar(self):
        # 各状態の強調色（背景）。パトライトと同じ配色。
        colors = {
            "run":   "#32CD32",   # 緑（点灯）
            "stop":  "#FF3030",   # 赤（点灯）
            "estop": "#800080",   # 紫（点灯）
            "error": "#FF3030",   # 赤（点滅）
        }
        cells = ""
        for key, text in (("run", "RUN"), ("stop", "STOP"), ("estop", "ESTOP"), ("error", "ERROR")):
            if key == self._current_status:
                if key == "error" and not self._error_blink_on:
                    # 点滅の消灯フェーズ: 背景を消して赤文字のみ（バー地色に戻す）
                    cells += (
                        f'<td align="center" width="25%" '
                        f'style="color:#FF3030; font-size:26px; font-weight:bold;">&nbsp;{text}&nbsp;</td>'
                    )
                else:
                    cells += (
                        f'<td align="center" width="25%" '
                        f'style="background-color:{colors[key]}; color:#FFFFFF; '
                        f'font-size:26px; font-weight:bold;">&nbsp;{text}&nbsp;</td>'
                    )
            else:
                cells += (
                    f'<td align="center" width="25%" '
                    f'style="color:#5A5A5A; font-size:22px;">{text}</td>'
                )
        html = (
            '<table width="100%" cellspacing="6" style="font-family:\'Meiryo\';">'
            f'<tr>{cells}</tr></table>'
        )
        self.label_status.setText(html)

    # --- 履歴表示を更新する関数 (HTMLテーブル版) ---
    def update_history_display(self):
        self._render_history_table()
        self._render_stats_grid()

    # --- 判定履歴テーブル（ID + カメラ別列）の描画 ---
    def _render_history_table(self):
        # 列構成（固定）: ID / inside / outside / top / under。
        #   確定クラス専用列は廃止。各カメラ列に「そのカメラの検出クラス＋信頼度」を表示し、
        #   確定クラス(4カメラ集約で決めた1クラス)を検出した列は「◎確定クラス」を赤枠で優先表示する
        #   （そのカメラの最高信頼度クラスでなくても確定クラスを出す）。
        #   確定クラスを検出していないカメラは自分の最高信頼度クラスを表示。検出ゼロは "-"。
        n_cam = len(HISTORY_CAM_COLUMNS)                 # =4
        cam_w = max(8, round((100 - 10) / n_cam))        # カメラ1列の幅(%)

        rows_html = ""
        for item in self.history_data:
            id_txt  = f"{item['id']:03}"
            final   = item.get("final", "")
            per_cam = item.get("per_cam", {})

            # カメラ別列（確定クラス検出時は◎優先、それ以外は最高信頼度クラス、未検出は "-"）
            cam_cells = ""
            for cam_key, _hdr in HISTORY_CAM_COLUMNS:
                info = per_cam.get(cam_key)
                if info is None:
                    cell_html = '<span style="font-size:22px; font-weight:bold; color:#000000;">-</span>'
                elif info.get("final_conf") is not None:
                    # このカメラは確定クラスを検出 → ◎付き・赤枠で確定クラスを優先表示
                    cell_html = _chip_confirmed_html(final, info["final_conf"])
                else:
                    # 確定クラス未検出 → このカメラの最高信頼度クラスを表示
                    label, conf = info["top"]
                    cell_html = _chip_inner_html(label, conf, "#000000")
                cam_cells += (
                    f'<td width="{cam_w}%" align="center" valign="middle" '
                    f'style="border-right:1px solid #000000; border-bottom:1px solid #000000; padding:8px 6px;">'
                    f'{cell_html}</td>'
                )

            rows_html += (
                f'<tr><td width="10%" align="center" '
                f'style="border-right:1px solid #000000; border-bottom:1px solid #000000; '
                f'font-size:26px; font-weight:bold; color:#000000;">{id_txt}</td>'
                f'{cam_cells}</tr>'
            )

        # ヘッダのカメラ列
        header_cam_cols = ""
        for _key, hdr in HISTORY_CAM_COLUMNS:
            header_cam_cols += (
                f'<th width="{cam_w}%" align="center" '
                f'style="font-size:22px; color:#000000; '
                f'border-right:1px solid #000000; border-bottom:2px solid #000000;">{hdr}</th>'
            )

        full_html = f"""
        <html>
        <body style="background-color:#FFFFFF;">
            <table width="100%" height="100%" cellspacing="0" cellpadding="6"
                   style="border:1px solid #000000; font-family:'Meiryo';">
                <tr>
                    <th width="10%" align="center"
                        style="font-size:22px; color:#000000;
                               border-right:1px solid #000000; border-bottom:2px solid #000000;">ID</th>
                    {header_cam_cols}
                </tr>
                {rows_html}
            </table>
        </body>
        </html>
        """
        self.label_history.setText(full_html)

    # --- 個数スタック欄（クラス名=結果表示と同じ色付き矩形、個数=黒太字）の描画 ---
    def _render_stats_grid(self):
        c = self.detection_counts

        def _stat_cell(label_key, jp, value):
            # 結果表示のチップと同じ配色でクラス名を矩形に囲う（色は CLASS_DISPLAY 準拠）。
            info = module_gui_JP.CLASS_DISPLAY.get(label_key, {"jp": jp, "color": "#CCCCCC"})
            bg   = info["color"]
            text_color, border = _chip_colors(bg)
            return (
                f'<td style="padding:5px 12px;">'
                f'<table cellspacing="0" cellpadding="3"><tr>'
                f'<td bgcolor="{bg}" align="center" '
                f'style="color:{text_color}; {border} font-size:20px; font-weight:bold;">'
                f'&nbsp;{jp}&nbsp;</td>'
                f'<td style="color:#000000; font-size:20px; font-weight:bold;">：{value}</td>'
                f'</tr></table>'
                f'</td>'
            )

        # クラスをグループごとにまとめて配置する。
        #   1段目: 健全果・未熟果（正常/未熟）
        #   左列:  傷系（裂果・果梗裂果・縫合線裂果）→ 異形系（双子果・奇形果・黒双子）
        #   右列:  食害系（鳥害・虫害）→ 病気系（カビ・灰星病）→ その他（萎凋果・擦れ果）
        #   → 各グループが同一列で縦に連続するように並べる。
        stat_rows = [
            (("healthy", "健全果"),        ("unripe", "未熟果")),      # 1段目: 正常・未熟
            (("crack", "裂果"),             ("birddamage", "鳥害")),    # 左:傷系  / 右:食害系
            (("stemcrack", "果梗裂果"),     ("insect", "虫害")),        # 左:傷系  / 右:食害系
            (("suturecrack", "縫合線裂果"), ("mold", "カビ")),          # 左:傷系  / 右:病気系
            (("twin", "双子果"),            ("brownrot", "灰星病")),    # 左:異形系 / 右:病気系
            (("malformation", "奇形果"),    ("wilt", "萎凋果")),        # 左:異形系 / 右:その他
            (("blacktwin", "黒双子"),       ("kasure", "擦れ果")),      # 左:異形系 / 右:その他
        ]
        stats_rows_html = ""
        for left, right in stat_rows:
            left_html  = _stat_cell(left[0], left[1], c[left[0]])
            right_html = _stat_cell(right[0], right[1], c[right[0]]) if right else "<td></td>"
            stats_rows_html += f"<tr>{left_html}{right_html}</tr>"

        stats_html = f"""
        <html>
        <body style="background-color:#FFFFFF; color:#000000; font-family:'Meiryo';">
            <table width="100%" height="100%" cellspacing="0" style="border: none;">
                {stats_rows_html}
            </table>
        </body>
        </html>
        """
        self.label_stats.setText(stats_html)

    # --- トグルスイッチ状態変更イベント ---
    @Slot(bool)
    def on_main_toggled(self, checked):
        # 設定ボタンのロックは運転状態・非常停止・単体モードを一元判定して反映する
        self._refresh_operation_lock()
        # カメラ別ワーカーの運転フラグを切り替える（OFF中は生映像のみ表示・推論/状態更新なし）
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.set_running(checked)
        if checked:
            self.update_camera_delays()
            self.label_toggle_status.setText("動作中")
            self.label_toggle_status.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #32CD32; qproperty-alignment: 'AlignCenter';
            """)
            log.info("スピード設定をメインに保存: %d", self.saved_speed)
            self.refresh_mode_panel()
            self.update_status_display("run")
            self.dcr.info("gui_state", state="run", speed=self.saved_speed, msg="トグルスイッチ ON: 動作開始")
            # シリアル書き込みは数バイト・1ms未満のため GUIスレッドから直接呼んで順序を保証する
            self.motor.set_speed(self.saved_speed)
            self.motor.rotate()
            # 正常運転中: GREEN、ブザーなし（カメラエラー中は変更しない）
            if not self._error_active:
                self.run_in_background(self.patlite.set_system_state, p_ctr.SystemState.RUNNING)
        else:
            self.label_toggle_status.setText("停止中")
            self.label_toggle_status.setStyleSheet("""
                font-family: "Meiryo"; font-size: 30px; font-weight: bold;
                color: #888888; qproperty-alignment: 'AlignCenter';
            """)
            # 非常停止中は estop 表示を維持する（停止に上書きしない）
            if not self._estop_active:
                self.update_status_display("stop")
            self.dcr.info("gui_state", state="stop", msg="トグルスイッチ OFF: 停止")
            # stop も同期呼び出しにして rotate より後に届くことを保証する。
            # ただし単体モード中はArduinoの単体動作を妨げないため停止(S)を送らない。
            if not self._standalone_active:
                self.motor.stop()
            self.run_in_background(self.relay.stop)
            # 単体モード中またはカメラエラー中はパトライトをここで変えない
            if not self._standalone_active and not self._error_active:
                self.run_in_background(self.patlite.set_system_state, p_ctr.SystemState.STANDBY)
