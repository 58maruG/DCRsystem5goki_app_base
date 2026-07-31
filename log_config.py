# log_config.py  —  コンソール出力（System B）の統一設定
#   アプリ全体の print をこのロガーに寄せ、書式・レベル・モジュール名・色を統一する。
#     コンソール: 既定 INFO 以上を色付きで表示
#     ファイル  : 既定 DEBUG 以上を logs/app_<line>_<YYYYMMDD>.log へ全部残す
#
#   使い方:
#     import log_config
#     log_config.setup_logging(line="5goki")     # main の最初で1回だけ
#     log = log_config.get_logger("motor")        # 各モジュールで
#     log.info("接続しました (COM3)")
#     log.warning("READY応答がありません")
#     log.error("接続エラー: ...")
#     log.debug("[Serial Send] ROTATE")           # コンソールには出ずファイルのみ
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# ログ書式: [時刻] [レベル] [モジュール] メッセージ
_FMT = "[%(asctime)s] [%(levelname)-5s] [%(name)-8s] %(message)s"
_DATEFMT = "%H:%M:%S"

# レベル別の ANSI 色（コンソール用）
_COLORS = {
    logging.DEBUG: "\033[90m",     # 灰
    logging.INFO: "\033[0m",       # 無色
    logging.WARNING: "\033[33m",   # 黄
    logging.ERROR: "\033[31m",     # 赤
    logging.CRITICAL: "\033[1;41m",  # 赤背景
}
_RESET = "\033[0m"

def _enable_windows_ansi():
    """Windows 10+ のコンソールで ANSI エスケープを有効化する。失敗しても無視。"""
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # STDOUT(-11) の仮想ターミナル処理を有効化
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    except Exception:
        pass


class _ColorFormatter(logging.Formatter):
    """コンソール用: 行頭に色を付け、行末でリセットする。"""
    def __init__(self, use_color=True):
        super().__init__(_FMT, datefmt=_DATEFMT)
        self.use_color = use_color

    def format(self, record):
        text = super().format(record)
        if self.use_color:
            color = _COLORS.get(record.levelno, "")
            if color:
                return f"{color}{text}{_RESET}"
        return text


def setup_logging(line="line", log_dir="logs",
                  console_level=logging.INFO, file_level=logging.DEBUG,
                  color=True, to_file=True):
    """アプリ全体のロギングを初期化する。main の最初で1回だけ呼ぶ。

    console_level: コンソールに出す最小レベル（既定 INFO）
    file_level   : ファイルに残す最小レベル（既定 DEBUG）
    color        : コンソールを色付けするか
    to_file      : ファイルへ残すか
    """
    # レベル名を 5 文字以内の短縮表記に揃える（[%(levelname)-5s] の桁を保つ）
    logging.addLevelName(logging.WARNING, "WARN")
    logging.addLevelName(logging.CRITICAL, "CRIT")

    root = logging.getLogger()
    root.setLevel(min(console_level, file_level if to_file else console_level))

    # 二重設定を避けるため既存ハンドラを除去（再呼び出しにも耐える）
    for h in list(root.handlers):
        root.removeHandler(h)

    if color:
        _enable_windows_ansi()
        use_color = sys.stdout is not None and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    else:
        use_color = False

    # コンソールハンドラ
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(_ColorFormatter(use_color=use_color))
    root.addHandler(ch)

    # ファイルハンドラ（日付つき・追記）
    if to_file:
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            d = datetime.now().strftime("%Y%m%d")
            fh = logging.FileHandler(Path(log_dir) / f"app_{line}_{d}.log", encoding="utf-8")
            fh.setLevel(file_level)
            fh.setFormatter(logging.Formatter(_FMT, datefmt="%Y-%m-%d %H:%M:%S"))
            root.addHandler(fh)
        except Exception as e:
            # ファイルを開けなくてもコンソールは生かす
            root.warning("ログファイルを開けませんでした: %s", e)

    return root


def get_logger(name):
    """モジュール用ロガーを返す。setup_logging 前でも安全（後から効く）。"""
    return logging.getLogger(name)
