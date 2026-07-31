# -------------------------------------------------
# module_patlite.py
# パトライト制御モジュール (v2: システム状態ベース)
# -------------------------------------------------
from __future__ import annotations
import hid
import time

import log_config
log = log_config.get_logger("patlite")

# ================================================
# USB 接続設定
# ================================================
VENDER_ID  = 0x191a
PRODUCT_ID = 0x6001

# ================================================
# ブザー設定
# ================================================
BUZZER_VOLUME = 0  # 0=消音, 1〜9=音量, 10=最大音量


# ================================================
# LEDパターン定義
# ================================================
class LedPattern:
    """
    LEDの制御値とデバッグ表示名を管理するクラス。
    構成: (Byte5の16進数値, デバッグ用表示名)

    NE-USB 通信仕様より、Byte5は 上位ニブル=色 / 下位ニブル=LEDパターン:
      色      : 0消灯 1赤 2緑 3黄 4青 5紫 6水 7白 (15=前回保持)
      パターン: 0消灯 1点灯 2〜7=点滅パターン1〜6 (15=前回保持)
    """
    OFF       = (0x00, "消灯")
    RED       = (0x11, "赤")
    GREEN     = (0x21, "緑")
    YELLOW    = (0x31, "黄")
    BLUE      = (0x41, "青")
    VIOLET    = (0x51, "紫")
    SKY       = (0x61, "水")
    WHITE     = (0x71, "白")
    # 赤(1) + 点滅パターン1(2) = 0x12。別の点滅速度にしたいときは下位ニブルを
    # 0x13〜0x17(パターン2〜6)に変える。
    RED_BLINK = (0x12, "赤点滅")


# ================================================
# システム状態定義
# ================================================
class SystemState:
    """
    システム状態とパトライト設定の対応クラス。
    構成: (LedPattern, ブザーON/OFF, 状態名)
    """
    INITIALIZING = (LedPattern.YELLOW,    False, "初期化中")
    STANDBY      = (LedPattern.RED,       False, "待機中")
    RUNNING      = (LedPattern.GREEN,     False, "正常運転中")
    ESTOP        = (LedPattern.VIOLET,    False, "非常停止中")  # 人が押す=監視中のためブザーなし(紫点灯)
    ERROR        = (LedPattern.RED_BLINK, False, "エラー停止")  # 想定外異常: ブザーなし・赤点滅で通知
    STANDALONE   = (LedPattern.OFF,       False, "単体モード中") # ArduinoがPCなしで動作中: 消灯・ブザーなし


# ================================================
# パトライトコントローラ
# ================================================
class PatliteController:
    def __init__(self) -> None:
        self.device = None

    # --- 初期化・接続 ---
    def init(self) -> bool:
        try:
            if self.device:
                return True
            self.device = hid.device()
            self.device.open(VENDER_ID, PRODUCT_ID)
            log.info("接続成功")
            time.sleep(1.0)
            self.set_color(LedPattern.OFF)
            return True
        except Exception as e:
            log.error("接続エラー: %s", e)
            self.device = None
            return False

    # --- 制御コマンド送信（内部用） ---
    def _send_command(self, data: list[int]) -> bool:
        if self.device is None:
            log.error("初期化されていません。")
            return False
        try:
            self.device.write(data)
            return True
        except Exception as e:
            log.error("書き込み失敗: %s", e)
        return False

    # --- LED色・ブザー同時制御 ---
    def set_color(self, pattern: tuple = LedPattern.OFF,
                  buzzer: bool = False) -> tuple[bool, str]:
        led_byte, color_name = pattern
        data = [0] * 9
        data[1] = 0x00
        data[2] = 0x00
        data[3] = 0x01 if buzzer else 0x00  # ブザーパターン: 1=連続吹鳴, 0=停止
        data[4] = BUZZER_VOLUME if buzzer else 0x00  # ブザー音量: 0=消音, 1〜9, 10=最大
        data[5] = led_byte
        data[6] = 0x00
        data[7] = 0x00
        data[8] = 0x00
        return self._send_command(data), color_name

    # --- システム状態設定（SystemState クラスの定数を使用） ---
    def set_system_state(self, state: tuple) -> tuple[bool, str]:
        pattern, buzzer, state_name = state
        log.info("状態遷移: %s", state_name)
        return self.set_color(pattern, buzzer)

    # --- 接続終了 ---
    def close(self) -> None:
        if self.device:
            self.set_color(LedPattern.OFF)
            self.device.close()
            self.device = None
            log.info("切断完了")
