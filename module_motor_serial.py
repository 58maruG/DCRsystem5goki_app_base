# -------------------------------------------------
# module_motor_serial.py
#   ArduinoとUSBシリアルで通信し、ターンテーブルを制御するモジュール。
#
#   GUI側は rotate() / stop() / set_speed() / cleanup() を呼ぶだけでよい。
#   送信するシリアルコマンドは次のとおり (Arduinoスケッチと一致させること):
#     R        : 回転開始
#     S        : 停止
#     C        : 停止・初期化 (システムクリーンアップ)
#     V{速度}  : 速度設定 (例 V5)
#     Q        : 状態問い合わせ (現在の非常停止状態を ESTOP/ESTOP_CLEARED で返す)
#
#   Arduinoが非同期に送ってくる通知 (受信スレッドが処理):
#     ESTOP / ESTOP_CLEARED   : 非常停止の作動 / 解除
#     STANDALONE / PC_MODE     : 単体モードへ移行 / PC連携モードへ復帰
#       ※単体モードはArduino単独でターンテーブルを動かす運転モード。
#         移行・復帰のたびにArduinoがこの行を送ること。起動時点で既に
#         単体モードなら、起動完了直後にも STANDALONE を送ると状態が同期できる。
#
#   依存: pyserial   (uv add pyserial / pip install pyserial)
# -------------------------------------------------
from __future__ import annotations
import threading
import time

import serial
import serial.tools.list_ports

import log_config
log = log_config.get_logger("motor")

# ================================================
# シリアル通信定数
# ================================================
# Arduinoが起動完了時に送ってくる合図 (スケッチの setup() と一致させる)
READY_TOKEN = "READY"
# 自動検出に使うUSBシリアル変換チップの識別子(ポート説明文に含まれる文字列)
_KNOWN_KEYWORDS = ("arduino", "ch340", "ch341", "usb-serial", "wch", "ftdi", "cp210")

# ================================================
# 速度テーブル（スケッチの SPEED_DELAY_US と一致させること）
# pulse_width_ms = _SPEED_DELAY_US[level-1] / 1000  (HIGH/LOW それぞれの幅)
# ================================================
_SPEED_DELAY_US = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]


# ================================================
# モータ制御(シリアル)クラス
# ================================================
class MotorSerial:
    def __init__(self, port: str | None = None, baudrate: int = 115200,
                 timeout: float = 2.0,
                 on_estop=None, on_estop_cleared=None,
                 on_standalone=None, on_pc_mode=None,
                 dcr=None, heartbeat_interval_s: float = 30.0) -> None:

        # --- シリアル接続設定 ---
        self.port     = port            # None のとき自動検出
        self.baudrate = baudrate
        self.timeout  = timeout
        self.ser      = None
        self._lock    = threading.Lock()  # 複数スレッドからの同時送信を防ぐ

        # --- 非同期通知コールバック ---
        # （別スレッドから呼ばれるため、GUI更新はQtシグナル等で受けること）
        self.on_estop         = on_estop          # "ESTOP" 受信時
        self.on_estop_cleared = on_estop_cleared  # "ESTOP_CLEARED" 受信時
        self.on_standalone    = on_standalone     # "STANDALONE" 受信時 (単体モード移行)
        self.on_pc_mode       = on_pc_mode        # "PC_MODE" 受信時 (PC連携モード復帰)

        # --- 受信スレッド ---
        self._reader_thread = None
        self._reading       = False   # 受信スレッドの稼働フラグ

        # --- events ログ・ハートビート ---
        self._dcr          = dcr
        self._hb_interval_s = heartbeat_interval_s
        self._hb_lock      = threading.Lock()
        self._hb_ping_ts   = None   # ping送信時刻（None=飛行中なし）
        self._hb_stop      = threading.Event()
        self._hb_thread    = None

    # --- 接続するポートを自動検出する ---
    def _detect_port(self) -> str | None:
        for p in serial.tools.list_ports.comports():
            desc = (p.description or "").lower()
            if any(k in desc for k in _KNOWN_KEYWORDS):
                return p.device
        return None

    # --- 接続・初期化 (成功でTrue) ---
    def init(self) -> bool:
        try:
            port = self.port or self._detect_port()
            if port is None:
                log.error("Arduinoが見つかりません。SERIAL_PORT を手動で指定してください(例 'COM3')。")
                return False

            # write_timeout も明示する。未設定(None)だとOS側の送信バッファが詰まった際に
            #   write() が無期限にブロックし、GUIスレッドから同期的に呼ばれる send系
            #   （query_estop 等）でアプリ全体がフリーズする事故につながるため。
            self.ser = serial.Serial(port, self.baudrate,
                                      timeout=self.timeout, write_timeout=self.timeout)
            # ポートを開くとArduinoが自動リセットされる。起動完了を待つ
            time.sleep(2.0)
            self._wait_ready()
            # 受信を専用スレッドに一本化して開始する
            # （以降 readline はこのスレッドだけが行う）
            self._reading = True
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()
            log.info("接続しました (%s)", port)
            if self._dcr:
                self._dcr.info("arduino_connect", port=port, baudrate=self.baudrate)
                self._hb_stop.clear()
                self._hb_thread = threading.Thread(
                    target=self._heartbeat_loop, name="dcr-arduino-hb", daemon=True
                )
                self._hb_thread.start()
            return True
        except serial.SerialException as e:
            log.error("接続エラー: %s", e)
            self.ser = None
            return False

    # --- 起動完了(READY)を待つ。来なくても致命ではないので警告のみ ---
    def _wait_ready(self) -> None:
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            line = self.ser.readline().decode(errors="ignore").strip()
            if line == READY_TOKEN:
                return
        log.warning("READY応答がありません(処理は継続します)。")

    # --- 非同期通知コールバックの呼び出し (内部用) ---
    #   コールバック内の例外を1つ握りつぶすだけで受信スレッド全体を落とさないための共通処理。
    def _dispatch_callback(self, callback, label: str) -> None:
        if callback is None:
            return
        try:
            callback()
        except Exception as e:
            log.error("%sコールバック例外: %s", label, e)

    # --- 受信スレッド本体 (Arduinoからの行を常時読み続ける) ---
    #   コマンド応答(OK:.. / ERR:..)はログ表示のみ。
    #   非同期通知(ESTOP / ESTOP_CLEARED)はコールバックへ渡す。
    def _reader_loop(self) -> None:
        while self._reading:
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
            except serial.SerialException as e:
                if self._reading:
                    log.error("受信エラー: %s", e)
                    if self._dcr:
                        self._dcr.error("arduino_disconnect", reason=str(e))
                break
            except Exception:
                break

            if not line:
                continue  # timeoutで空行になっただけ。ループ継続

            if line == "ESTOP":
                log.debug("[Serial Recv] ESTOP (非常停止 作動)")
                self._dispatch_callback(self.on_estop, "on_estop")
            elif line == "ESTOP_CLEARED":
                log.debug("[Serial Recv] ESTOP_CLEARED (非常停止 解除)")
                self._dispatch_callback(self.on_estop_cleared, "on_estop_cleared")
            elif line == "STANDALONE":
                log.debug("[Serial Recv] STANDALONE (単体モード 移行)")
                self._dispatch_callback(self.on_standalone, "on_standalone")
            elif line == "PC_MODE":
                log.debug("[Serial Recv] PC_MODE (PC連携モード 復帰)")
                self._dispatch_callback(self.on_pc_mode, "on_pc_mode")
            elif line == "OK:PONG":
                t_recv     = time.time()
                latency_ms = None
                with self._hb_lock:
                    if self._hb_ping_ts is not None:
                        latency_ms      = round((t_recv - self._hb_ping_ts) * 1000, 1)
                        self._hb_ping_ts = None
                if self._dcr and latency_ms is not None:
                    self._dcr.info("arduino_heartbeat", status="ok", latency_ms=latency_ms)
                log.debug("[Serial Recv] OK:PONG")
            else:
                log.debug("[Serial Recv] %s", line)

    # --- ハートビートループ (arduino_heartbeat イベントを定期記録) ---
    def _heartbeat_loop(self) -> None:
        while self._reading:
            stopped = self._hb_stop.wait(self._hb_interval_s)
            if stopped or not self._reading:
                break
            t_send = time.time()
            with self._hb_lock:
                self._hb_ping_ts = t_send
            if not self._send_command("P"):
                with self._hb_lock:
                    self._hb_ping_ts = None
                continue
            # pong を待つ（最大 self.timeout 秒）
            deadline = t_send + self.timeout
            while time.time() < deadline and self._reading:
                with self._hb_lock:
                    if self._hb_ping_ts is None:
                        break
                time.sleep(0.05)
            with self._hb_lock:
                if self._hb_ping_ts is not None:
                    self._hb_ping_ts = None
                    if self._dcr:
                        self._dcr.warn("arduino_heartbeat", status="timeout")

    # --- 低レベル送信 (シリアルコマンド文字列をそのまま送る) ---
    #   応答(OK:../ERR:..)の受信は _reader_loop が一括で行う
    def _send_command(self, command: str) -> bool:
        if self.ser is None or not self.ser.is_open:
            log.warning("未接続のため送信できません: %s", command)
            return False

        try:
            with self._lock:
                log.debug("[Serial Send] %s", command)
                self.ser.write((command + "\n").encode())
            return True
        except serial.SerialException as e:
            log.error("送信エラー: %s", e)
            return False

    # --- 高レベルコマンド (GUIから呼ぶ) ---
    def rotate(self) -> bool:
        """回転開始"""
        return self._send_command("R")

    def stop(self) -> bool:
        """停止"""
        return self._send_command("S")

    def cleanup(self) -> bool:
        """停止・初期化 (システムクリーンアップ)"""
        return self._send_command("C")

    def set_speed(self, speed: int) -> bool:
        """速度設定 (1〜10)。設定時に arduino_params イベントを記録する。"""
        ok = self._send_command(f"V{speed}")
        if ok and self._dcr and 1 <= speed <= 10:
            pw_ms = _SPEED_DELAY_US[speed - 1] / 1000.0
            self._dcr.info("arduino_params", speed_level=speed,
                           pulse_width_ms=pw_ms, period_ms=round(pw_ms * 2, 3))
        return ok

    def query_estop(self) -> bool:
        """現在の非常停止状態を問い合わせる (起動時の状態同期用)。

        Arduinoは現在の状態に応じて 'ESTOP' か 'ESTOP_CLEARED' を返す。
        応答は通常の非同期通知と同じく _reader_loop が受け取り、
        on_estop / on_estop_cleared コールバックを呼ぶ。
        """
        return self._send_command("Q")

    # --- 切断 (念のため停止してから閉じる) ---
    def close(self) -> None:
        # 受信スレッドとハートビートスレッドを止める
        self._reading = False
        self._hb_stop.set()
        if self.ser is not None and self.ser.is_open:
            try:
                with self._lock:
                    self.ser.write(b"S\n")
                    time.sleep(0.05)
                    self.ser.close()
                log.info("切断しました")
            except serial.SerialException as e:
                log.error("切断エラー: %s", e)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=self.timeout + 0.5)
            self._reader_thread = None
        if self._hb_thread is not None:
            self._hb_thread.join(timeout=self.timeout + 0.5)
            self._hb_thread = None
        self.ser = None
