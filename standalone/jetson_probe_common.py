"""Jetson購入判断のための検証スクリプトが共有するヘルパー。

  run_as_jetson.py       本番(main.py)を制約下で起動し、実運転のログを取る
  jetson_live_probe.py   run_as_jetson.py から使う計測プローブ（CPU・クロック・HSV時間）
  analyze_cycle_logs.py  本番が書いた cycle ログから1個体あたりの実コストを集計・比較する
  project_jetson.py      実測したスケール係数を Jetson 実機の処理時間へ換算する

──────────────────────────────────────────────────────────────
【設計方針】HSV検出パイプラインをここで再実装しない
──────────────────────────────────────────────────────────────
縮小率の切り替えは module_yolo.HSV_DETECT_SCALE の書き換え（detect_scale）で行う。
get_target_info は呼び出しのたびにモジュールグローバルを読むため、この書き換えだけで
「本番と1ビットも違わない処理を、任意の縮小率で」実行できる。

──────────────────────────────────────────────────────────────
【依存】psutil（CPU計測・affinity）、cv2/numpy は必須。他は使う機能に応じて遅延 import する。
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager

import cv2

# standalone/ から実行しても親ディレクトリのモジュールを解決できるようにする
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 出力先の既定。analysis/ は .gitignore 済みなので、計測結果でリポジトリを汚さない。
DEFAULT_OUT_DIR = os.path.join("analysis", "jetson_validation")


def repo_root() -> str:
    return _REPO_ROOT


def ensure_repo_cwd() -> None:
    """カレントディレクトリをリポジトリルートへ移す。

    module_yolo._load_hsv_config が相対パス "json/hsv_config_{cam}.json" を開くため、
    standalone/ から実行してもルートに居る必要がある。"""
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        os.chdir(_REPO_ROOT)


def out_dir(sub: str | None = None, base: str | None = None) -> str:
    """出力ディレクトリを作って返す（相対パスはリポジトリルート基準）。"""
    d = base or DEFAULT_OUT_DIR
    if not os.path.isabs(d):
        d = os.path.join(_REPO_ROOT, d)
    if sub:
        d = os.path.join(d, sub)
    os.makedirs(d, exist_ok=True)
    return d


def write_json(path: str, obj: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


# ==========================================================
# 本番パイプラインの呼び出し
# ==========================================================
_module_yolo = None


def load_module_yolo():
    """本番の module_yolo を遅延 import して返す。

    import すると ultralytics / torch / PySide6 が読み込まれる（数秒かかる）。
    ultralytics の import 自体が cv2.getNumThreads() を1へ変える副作用を持つため、
    スレッド数を制御したいスクリプトは必ずこの import の「後」に cv2.setNumThreads()
    を呼ぶこと（apply_cpu_limits はそれを守るために内部で本関数を先に呼ぶ）。"""
    global _module_yolo
    if _module_yolo is None:
        ensure_repo_cwd()
        import module_yolo  # noqa: E402  遅延 import（重い依存を持つため）
        _module_yolo = module_yolo
    return _module_yolo


@contextmanager
def detect_scale(scale: float):
    """module_yolo.HSV_DETECT_SCALE を一時的に差し替える。

    run_as_jetson.py はこれで包んで main.py を起動する（起動中ずっと差し替わる）。"""
    my = load_module_yolo()
    old = my.HSV_DETECT_SCALE
    my.HSV_DETECT_SCALE = float(scale)
    try:
        yield
    finally:
        my.HSV_DETECT_SCALE = old


def production_scale() -> float:
    """現在 module_yolo が採用している縮小率（既定 0.5）。"""
    return float(load_module_yolo().HSV_DETECT_SCALE)


# ==========================================================
# 実行条件の固定（コア数 / スレッド数）
# ==========================================================
def apply_cpu_limits(cores: int = 0, cv_threads: int = -1,
                     torch_threads: int = -1) -> dict:
    """Jetson相当の実行条件を再現する。適用後の実効値を返す。

      cores        : 使用論理コア数（0 = 制限しない）。CPU affinity で絞る。
      cv_threads   : cv2.setNumThreads の値（-1 = 触らない / 0 or 1 = 実質シングル）。
      torch_threads: torch.set_num_threads の値（-1 =触らない）。

    【重要】本関数は cv2.setNumThreads より先に module_yolo を import する。
    ultralytics の import が cv2 スレッド数を1に書き換える副作用があり、
    順序を逆にすると指定した値が上書きされて計測条件がずれるため。"""
    load_module_yolo()   # 副作用を先に済ませてから、こちらの指定で上書きする

    applied = {"requested_cores": cores, "requested_cv_threads": cv_threads}

    if cores and cores > 0:
        try:
            import psutil
            psutil.Process().cpu_affinity(list(range(cores)))
            applied["affinity"] = psutil.Process().cpu_affinity()
        except ImportError:
            applied["affinity_error"] = "psutil が無いため affinity を設定できません"
        except (AttributeError, OSError) as e:
            # macOS など cpu_affinity 非対応プラットフォーム
            applied["affinity_error"] = f"{type(e).__name__}: {e}"
    else:
        applied["affinity"] = "unlimited"

    if cv_threads >= 0:
        cv2.setNumThreads(cv_threads)
    cv2.setUseOptimized(True)
    applied["cv2_num_threads"] = cv2.getNumThreads()
    applied["cv2_use_optimized"] = cv2.useOptimized()

    if torch_threads >= 0:
        try:
            import torch
            torch.set_num_threads(torch_threads)
            applied["torch_num_threads"] = torch.get_num_threads()
        except ImportError:
            applied["torch_error"] = "torch なし"

    return applied


# ==========================================================
# 環境情報の記録
# ==========================================================
_BUILD_KEYS = [
    "Parallel framework", "Use IPP", "NEON", "Use OpenCL", "CUDA",
    "Use Intel IPP", "cpu baseline", "requires", "dispatch",
]


def opencv_build_summary() -> dict:
    """cv2.getBuildInformation() から、PC↔Jetson の比較で効く行だけ抜き出す。

    JetPack 同梱の OpenCV は CUDA 無効ビルドで、並列フレームワーク（TBB/OpenMP/pthreads）や
    SIMD ディスパッチも PC と異なる。両環境でこの出力を保存し、差分を報告に載せること。"""
    info = cv2.getBuildInformation()
    picked = {}
    for line in info.splitlines():
        s = line.strip()
        low = s.lower()
        if any(k.lower() in low for k in _BUILD_KEYS):
            if ":" in s:
                k, v = s.split(":", 1)
                picked[k.strip()] = v.strip()
    return picked


def cpu_model() -> str:
    """CPU の型番。platform.processor() は Linux で空になることが多いので補う。"""
    name = platform.processor() or ""
    if not name and sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.lower().startswith(("model name", "hardware")):
                        name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    return name or platform.machine()


def jetson_power_mode() -> str | None:
    """Jetson 実機なら nvpmodel の電力モードを返す。PC では None。"""
    if not os.path.exists("/etc/nvpmodel.conf"):
        return None
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True,
                             timeout=5.0)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def env_info() -> dict:
    """計測結果と一緒に保存する環境スナップショット。
    これが無いと後から数値の比較ができない（PC/Jetson・スレッド数・ビルド差）。"""
    info = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu_model(),
        "logical_cores": os.cpu_count(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "cv2_num_threads": cv2.getNumThreads(),
        "opencv_build": opencv_build_summary(),
        "env": {k: os.environ.get(k) for k in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "OPENCV_FOR_THREADS_NUM")
                if os.environ.get(k)},
    }
    mode = jetson_power_mode()
    if mode:
        info["nvpmodel"] = mode
    try:
        import psutil
        freq = psutil.cpu_freq()
        if freq:
            info["cpu_freq_mhz"] = {"current": round(freq.current, 1),
                                    "max": round(freq.max, 1)}
    except (ImportError, OSError, AttributeError):
        pass
    return info


# ==========================================================
# 統計
# ==========================================================
def stats_ms(samples) -> dict:
    """計測列[ms]から代表値をまとめる。平均は外れ値に弱いので中央値も必ず併記する。"""
    n = len(samples)
    if n == 0:
        return {"n": 0}
    s = sorted(samples)

    def pct(p: float) -> float:
        return s[min(n - 1, int(n * p))]

    return {
        "n": n,
        "mean_ms": round(sum(s) / n, 3),
        "p50_ms": round(pct(0.50), 3),
        "p95_ms": round(pct(0.95), 3),
        "p99_ms": round(pct(0.99), 3),
        "min_ms": round(s[0], 3),
        "max_ms": round(s[-1], 3),
    }
