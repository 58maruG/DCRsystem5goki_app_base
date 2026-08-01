# -------------------------------------------------
# リレーボードをYOLOの結果により制御するプログラムmodule
# -------------------------------------------------
from __future__ import annotations
import platform
import ctypes
import time
import os
import json
from enum import IntEnum

import log_config
log = log_config.get_logger("relay")

# ================================================
# DLL 接続定数
# ================================================
YDCI_RESULT_SUCCESS = 0   # 正常終了
YDCI_OPEN_NORMAL    = 0   # YdciOpen 通常オープン

# ================================================
# リレー動作定数
# ================================================
RELAY_OPEN_TIME    = 0.15    # リレーの開閉時間（秒）
PULSE_PER_ROTATION = 6400    # TB6600のパルス数（1回転あたり）

# ================================================
# リレータイミング補正設定（角度ベース）
#   standalone/relay_calibration.py（GUIツール）が合わせ込んだ「検知位置→各弁
#   までの実効回転角度[度]」を書き出すファイル。
#   待機時間 = sec * (角度 / 360) で実機に合わせる。
# ================================================
RELAY_CAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json", "relay_config.json")


def load_relay_calibration() -> dict:
    """リレー待機角度の補正（relay_config.json）を読み込む。"""
    with open(RELAY_CAL_PATH, "r", encoding="utf-8-sig") as f:  # BOM付きでも読めるよう sig
        cfg = json.load(f)
    cal = {
        "remove":    {"angle": float(cfg["remove"]["angle"])},
        "transport": {"angle": float(cfg["transport"]["angle"])},
    }
    log.info("リレー補正(角度)を読込: %s", cal)
    return cal

# ================================================
# 速度テーブル（speed → HIGH/LOW幅 [秒]）
# ================================================
SPEED_MAP = {
    1: 0.0010,  # 回転遅い
    2: 0.0009,
    3: 0.0008,
    4: 0.0007,
    5: 0.0006,
    6: 0.0005,  # 基準 (デフォルト)
    7: 0.0004,
    8: 0.0003,
    9: 0.0002,
    10: 0.0001, # 回転速い
}


# ================================================
# 列挙型定義
# ================================================
class RelayState(IntEnum):
    """リレーの状態定義"""
    CLOSE = 0  # 回路を閉じる
    OPEN  = 1  # 回路を開く


class RelayChannel(IntEnum):
    """チャンネル定義"""
    TRANSPORT = 0  # 健全果運搬用
    REMOVE    = 1  # 被害果除去用


# ================================================
# リレーコントローラ
# ================================================
class RelayController:
    def __init__(self) -> None:
        self.ydci         = None
        self.board_id     = ctypes.c_ushort()
        self.is_connected = False
        # 起動時にリレータイミング補正を読み込む（無ければ無補正）
        self.calibration  = load_relay_calibration()

    # --- 初期化・接続 ---
    def init(self) -> bool:
        # すでに接続されている場合は、正常終了を返す（二重オープンエラー防止）
        if self.is_connected and self.ydci is not None:
            log.info("既に接続されています。")
            return True

        # DLLのロード
        pf = platform.system()
        if pf == 'Windows':
            try:
                self.ydci = ctypes.windll.Ydci
            except OSError:
                log.error("Ydci.dll が見つかりません。")
                return False
        else:
            log.error("サポートされていないOSです: %s", pf)
            return False

        result_board = self.ydci.YdciOpen(
            self.board_id, b'RLY-P4/2/0B-UBT', ctypes.byref(self.board_id), YDCI_OPEN_NORMAL
        )
        if result_board != YDCI_RESULT_SUCCESS:
            # すでに開いている場合のエラーコード（例：-838860781等）でも接続済みならTrue
            if result_board == -838860781:
                log.warning("既に他のプロセスまたは以前の処理で開かれています。")
                self.is_connected = True
                return True
            log.error("オープンできません。エラーコード: %s", result_board)
            return False

        self.is_connected = True
        log.info("リレーボード(%s)接続成功", self.board_id.value)
        self.stop()  # 初期状態として全OFF
        return True

    # --- リレー状態設定（内部用） ---
    def _set_state(self, channel: int, state: int) -> bool:
        """
        [ydci.YdciRlyOutput()]
        board_id -> リレー制御ボードを識別するための変数。ctypes.c_ushort 型で定義し、YdciRlyOpen が正常に実行されると、この変数にボードIDが格納されます。
        ctypes.byref(output_data) -> relay_ON(0),relay_OFF(1)
        start_channel -> 操作を開始するチャネルの番号
        num_channel -> 操作するチャネルの総数
        """
        if self.ydci is None:
            log.error("初期化されていません。")
            return False
        output_data = ctypes.c_ubyte(state)
        result = self.ydci.YdciRlyOutput(self.board_id, ctypes.byref(output_data), channel, 1)
        if result != YDCI_RESULT_SUCCESS:
            log.error("Ch%s の状態設定に失敗しました。エラーコード: %s", channel, result)
            return False
        return True

    # --- 待機時間の算出（内部用） ---
    def _set_wait_time(self, speed: int) -> tuple[float, float]:
        if not self.is_connected:
            log.warning("ボード未接続のためパルス動作をスキップします。")
            return 0.0, 0.0

        delay       = SPEED_MAP[speed]
        t_one_pulse = delay * 2
        sec         = t_one_pulse * PULSE_PER_ROTATION * 2  # ギア比が2なので

        # キャリブレーションで合わせ込んだ角度[度]を使う。
        #   待機時間 = sec * (角度 / 360)
        remove_angle    = self.calibration["remove"]["angle"]
        transport_angle = self.calibration["transport"]["angle"]
        remove_channel_wait    = sec * (remove_angle    / 360)
        transport_channel_wait = sec * (transport_angle / 360)

        return remove_channel_wait, transport_channel_wait

    # --- 待機後に指定チャンネルを RELAY_OPEN_TIME 秒だけ開弁する（move/fire共通） ---
    def _wait_then_pulse(self, channel: int, wait_sec: float) -> tuple[bool, float, float]:
        """戻り値: (開閉とも成功したか, 開弁予定epoch時刻, 実際に開弁したepoch時刻)"""
        called_at  = time.time()
        planned_ts = called_at + wait_sec
        time.sleep(wait_sec)

        t_open   = time.time()
        ok_open  = self._set_state(channel, RelayState.OPEN)
        time.sleep(RELAY_OPEN_TIME)
        ok_close = self._set_state(channel, RelayState.CLOSE)

        return ok_open and ok_close, planned_ts, t_open

    # --- リレー動作（指定チャンネル） ---
    def move(self, channel: int, speed: int) -> dict:
        """リレーを動作させる。タイミング情報と成否を含む辞書を返す。
        ok: 成功したか / planned_eject_ts: 開弁予定 epoch 時刻 / valve_opened_ts: 実際の開弁 epoch 時刻
        eject_delay_ms: 予定と実際の差(ms)。リレーが未接続の場合は ok=False で他は None。"""
        remove_wait_sec, transport_wait_sec = self._set_wait_time(speed)

        if channel == RelayChannel.REMOVE:
            wait_sec = remove_wait_sec
        elif channel == RelayChannel.TRANSPORT:
            wait_sec = transport_wait_sec
        else:
            log.error("不正なチャンネルが指定されました。")
            return {"ok": False, "planned_eject_ts": None,
                    "valve_opened_ts": None, "eject_delay_ms": None}

        ok, planned_ts, t_open = self._wait_then_pulse(channel, wait_sec)
        return {
            "ok":               ok,
            "planned_eject_ts": planned_ts,
            "valve_opened_ts":  t_open,
            "eject_delay_ms":   round((t_open - planned_ts) * 1000, 2),
        }

    # --- 待機時間を外部指定して弁を開閉（キャリブレーション用） ---
    def fire(self, channel: int, wait_sec: float) -> dict:
        """move() と違い角度計算をせず、与えられた wait_sec だけ待ってから開弁する。
        GUIスライダーで決めたライブ角度の待機時間をそのまま試打するために使う。"""
        if channel not in (RelayChannel.REMOVE, RelayChannel.TRANSPORT):
            log.error("不正なチャンネルが指定されました。")
            return {"ok": False, "planned_eject_ts": None, "valve_opened_ts": None}

        ok, planned_ts, t_open = self._wait_then_pulse(channel, max(0.0, wait_sec))
        return {"ok": ok, "planned_eject_ts": planned_ts, "valve_opened_ts": t_open}

    # --- 全リレー停止 ---
    def stop(self) -> None:
        if not self.is_connected or self.ydci is None:
            log.warning("ボード未接続のため停止動作をスキップします。")
            return
        # 接続を解除せず、状態だけを「安全（OFF）」にする
        self._set_state(RelayChannel.REMOVE,    RelayState.CLOSE)
        self._set_state(RelayChannel.TRANSPORT, RelayState.CLOSE)
        log.info("全リレーをOFF（停止）にしました。")

    # --- 接続終了 ---
    def close(self) -> None:
        if self.ydci is not None and self.is_connected:
            self._set_state(RelayChannel.REMOVE,    RelayState.CLOSE)
            self._set_state(RelayChannel.TRANSPORT, RelayState.CLOSE)
            self.ydci.YdciClose(self.board_id)
            self.ydci         = None
            self.is_connected = False
        log.info("切断完了")

    def __del__(self) -> None:
        self.close()
