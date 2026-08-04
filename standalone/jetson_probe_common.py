"""Jetson購入判断のための事前検証スクリプト群が共有するヘルパー。

検証プラン（jetson_purchase_validation_plan.md）の Task 1〜3 で使う以下から import される。

  profile_cpu_usage.py       Task 1  実効並列度・OpenCVスレッド構成の計測
  bench_hsv_jetson.py        Task 2  制約条件下でのCPUスケール係数の実測
  project_jetson.py          Task 2-5 1個体あたり処理時間への換算と GO/NO GO 判定
  verify_hsv_scale_video.py  Task 3  半解像度化による検出劣化の検証（動画・実機ライブ）

──────────────────────────────────────────────────────────────
【設計方針】HSV検出パイプラインをここで再実装しない
──────────────────────────────────────────────────────────────
本リポジトリには既に同一アルゴリズムの実装が2つある。

  1. 本番      module_yolo.ImageProcessor.get_target_info
  2. 校正ツール standalone/hsv_calibration.analyze（1の鏡像。各段のマスクも返す）

ここに3つ目を書くと「ベンチが速いのは実装が違うから」という結論の出せない
状態になる。よって本モジュールは **本番の get_target_info をそのまま呼ぶ**。
縮小率だけ module_yolo.HSV_DETECT_SCALE の一時差し替え（detect_scale）で制御する。

検証プラン §2-1 のサンプルコードは単帯 inRange + 5x5 OPEN + countNonZero だが、
実装は 2帯 inRange → OPEN(5x5)x2 → CLOSE(5x5)x2 → connectedComponentsWithStats
→ 虚像除去 → 果柄除去(半径20) → 左右端接触の棄却 であり、別物である。
プランのコードをそのまま使うと k を過小評価する（軽い処理を測ってしまう）。

──────────────────────────────────────────────────────────────
【依存】cv2 / numpy は必須。以下は使う機能に応じて遅延 import する。
  module_yolo（本番パイプライン） … ultralytics/torch/PySide6 を芋づるで読む
  psutil（CPU計測・affinity）
  pypylon（ライブ計測）
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import subprocess
import sys
import time

import cv2
import numpy as np

# standalone/ から実行しても親ディレクトリのモジュールを解決できるようにする
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ==========================================================
# カメラ定義（module_cameras_5goki / hsv_calibration と揃える）
# ==========================================================
CAM_NAMES = ["cam_top", "cam_under", "cam_inside", "cam_outside"]

# pfs の実解像度。cam_pfs/*.pfs の Width/Height と一致していること。
#   hsv_calibration.NATIVE_SIZE と同じ値（正方形なので1辺だけ持つ）。
NATIVE_SIZE = {
    "cam_top": 640, "cam_under": 640, "cam_inside": 560, "cam_outside": 500,
}

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


def imread_ja(path: str) -> np.ndarray | None:
    """日本語・スペースを含むパスでも読めるよう imdecode 経由で読む
    （hsv_calibration.imread_ja と同じ）。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ==========================================================
# 本番パイプラインの呼び出し
# ==========================================================
_module_yolo = None


def load_module_yolo():
    """本番の module_yolo を遅延 import して返す。

    import すると ultralytics / torch / PySide6 が読み込まれる（数秒かかる）。
    JETSON_MIGRATION_NOTES.md §14.1 の通り **ultralytics の import 自体が
    cv2.getNumThreads() を1へ変える副作用**を持つため、スレッド数を制御したい
    スクリプトは必ずこの import の「後」に cv2.setNumThreads() を呼ぶこと
    （apply_cpu_limits はそれを守るために内部で本関数を先に呼ぶ）。"""
    global _module_yolo
    if _module_yolo is None:
        ensure_repo_cwd()
        import module_yolo  # noqa: E402  遅延 import（重い依存を持つため）
        _module_yolo = module_yolo
    return _module_yolo


@contextlib.contextmanager
def detect_scale(scale: float):
    """module_yolo.HSV_DETECT_SCALE を一時的に差し替える。

    get_target_info は呼び出しのたびにモジュールグローバルを読むので、
    この差し替えだけで「本番と1ビットも違わない処理を、任意の縮小率で」計測できる。
    """
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


_reference_module = None


def load_reference_module(scale: float = 1.0):
    """本番とは独立した module_yolo のコピーを読み込み、縮小率を固定して返す。

    【なぜコピーが要るのか】
    detect_scale() のグローバル差し替えはプロセス全体に効く。本番実行中は4本の
    カメラワーカーが同時に get_target_info を呼んでいるため、基準側の計測のために
    グローバルを 1.0 にすると、その瞬間に走っている**他カメラの本番処理まで**
    1.0 で動いてしまう（データ競合。しかも結果が静かに壊れるので気づきにくい）。

    独立コピーなら HSV_DETECT_SCALE も設定キャッシュも別物なので競合しない。
    ultralytics 等の重い依存は sys.modules にキャッシュ済みのため再実行されず、
    コストは module_yolo 自身のトップレベル文だけ（モデルのロードは含まない）。"""
    global _reference_module
    if _reference_module is None:
        import importlib.util
        ensure_repo_cwd()
        load_module_yolo()            # 本番側を先に確定させる（設定ログの重複を避ける）
        path = os.path.join(_REPO_ROOT, "module_yolo.py")
        spec = importlib.util.spec_from_file_location("module_yolo__hsv_reference", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _reference_module = mod
    _reference_module.HSV_DETECT_SCALE = float(scale)
    return _reference_module


def detect(frame: np.ndarray, cam_name: str) -> dict | None:
    """本番の HSV 存在検出をそのまま実行する。縮小率は detect_scale() で制御する。"""
    return load_module_yolo().ImageProcessor.get_target_info(frame, cam_name)


def infer_window_px(cam_name: str) -> int:
    """推論する中心窓の半幅[px]（module_yolo.infer_window_px そのもの）。"""
    return load_module_yolo().infer_window_px(cam_name)


def in_infer_window(mx: int, width: int, cam_name: str) -> bool:
    """重心が中心窓の内側にあるか。エジェクタ射出タイミングに直結する判定。
    module_yolo._process_frame の帯ゲートと同じ式。"""
    return abs(mx - width / 2.0) <= infer_window_px(cam_name)


def load_hsv_config(cam_name: str) -> dict:
    """カメラ別HSV設定を本番と同じ経路で読む（キャッシュも本番と共有）。"""
    return load_module_yolo()._load_hsv_config(cam_name)


def analyze_status(frame: np.ndarray, cam_name: str, scale: float) -> str:
    """不一致フレームの原因診断用。校正ツールの analyze を使って
    'ok' / 'no_blob' / 'too_small' / 'edge' のどれで落ちたかを返す。

    get_target_info は落ちた理由に関わらず None を返すため、原因の切り分けには
    こちらを使う。表示用マスクをネイティブ解像度へ戻す処理が入っており本番より
    重いので、**不一致が出たフレームだけ**に呼ぶこと（時間計測には使わない）。"""
    ensure_repo_cwd()
    sys.path.insert(0, os.path.join(_REPO_ROOT, "standalone"))
    try:
        import hsv_calibration  # noqa: E402  pypylon/PySide6 を読む（GUIは起動しない）
    except ImportError as e:
        # pypylon 等が無い環境。診断は諦めるが、本体の集計は続行させる
        # （長時間の動画処理の最後で落ちるのが最悪のため）。
        return f"診断不可({type(e).__name__}: {e})"
    finally:
        sys.path.pop(0)
    raw = hsv_calibration.load_raw_config(cam_name)
    flat = hsv_calibration.flat_from_raw(raw, cam_name)
    return hsv_calibration.analyze(frame, flat, scale)["status"]


# ==========================================================
# 実行条件の固定（コア数 / スレッド数）
# ==========================================================
def apply_cpu_limits(cores: int = 0, cv_threads: int = -1,
                     torch_threads: int = -1) -> dict:
    """Jetson相当の実行条件を再現する。適用後の実効値を返す。

      cores        : 使用論理コア数（0 = 制限しない）。CPU affinity で絞る。
      cv_threads   : cv2.setNumThreads の値（-1 = 触らない / 0 or 1 = 実質シングル）。
      torch_threads: torch.set_num_threads の値（-1 = 触らない）。

    【重要】本関数は cv2.setNumThreads より先に module_yolo を import する。
    ultralytics の import が cv2 スレッド数を1に書き換える副作用（notes §14.1）があり、
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
    SIMD ディスパッチも PC と異なる。両環境でこの出力を保存し、差分を報告に載せること
    （プラン §2 注意点）。"""
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
    """Jetson 実機なら nvpmodel の電力モードを返す（プラン Appendix B: 電力モードの記録）。
    PC では None。"""
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
# 入力フレーム
# ==========================================================
def make_synthetic_frame(cam_name: str, cx_ratio: float = 0.5,
                         radius: int | None = None) -> np.ndarray:
    """実画像が無い環境でも走らせるための擬似フレーム（赤い果実＋果柄）。

    既存の bench/bench_*.py が使っている「赤い円」と同じ考え方だが、本番の
    2段目（虚像除去・果柄除去）まで到達させるため以下を満たすように作る。

      - 面積が min_blob_area（例 15000px²）を超える半径
      - 彩度が高い（虚像除去で消えない）
      - 上方向へ細い果柄を伸ばす（remove_stem が実際に仕事をする）

    cx_ratio: 果実中心のx座標（画像幅に対する比）。0.0〜1.0。
              Task 3 の擬似シーケンス生成で果実を左から右へ流すのに使う。"""
    n = NATIVE_SIZE.get(cam_name, 640)
    cfg = load_hsv_config(cam_name)
    if radius is None:
        # 面積 πr² が min_blob_area の約2倍になる半径（閾値ぎりぎりを避ける）
        radius = int(round(np.sqrt(2.0 * float(cfg["min_blob_area"]) / np.pi)))
        radius = max(8, min(radius, n // 2 - 4))

    frame = np.full((n, n, 3), 30, dtype=np.uint8)       # 暗い背景
    cx = int(round(cx_ratio * n))
    cy = n // 2
    # 果柄（細い突起）: 果柄除去の開処理が実際に走るよう、半径20px想定より細くする
    stem_w = max(2, int(round(float(cfg["stem_open"]) * 0.6)))
    cv2.rectangle(frame, (cx - stem_w // 2, max(0, cy - radius - 40)),
                  (cx + stem_w // 2, cy), (40, 40, 200), -1)
    cv2.circle(frame, (cx, cy), radius, (40, 40, 210), -1)   # BGR: 濃い赤
    # 光沢（高明度・低彩度）を1点入れて虚像除去の分岐も通す
    cv2.circle(frame, (cx - radius // 3, cy - radius // 3), max(3, radius // 6),
               (235, 235, 245), -1)
    return frame


def load_frames(cam_name: str, image: str | None = None,
                image_dir: str | None = None, limit: int = 0) -> tuple[list, str]:
    """ベンチ入力フレームを用意する。実画像優先・無ければ擬似フレームへフォールバック。

    戻り値: (フレーム列, 入力の説明文字列)

    学習データセットは全カメラ共通で 640x640 保存のため、カメラのネイティブ解像度と
    異なる場合はリサイズしてから返す（hsv_calibration の静止画モードと同じ扱い）。
    ここを揃えないと解像度が違うまま測ってしまい、係数が意味を失う。"""
    paths = []
    if image:
        paths = [image]
    elif image_dir:
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        for root, _dirs, files in os.walk(image_dir):
            for fn in sorted(files):
                if fn.lower().endswith(exts):
                    paths.append(os.path.join(root, fn))
        if limit > 0:
            paths = paths[:limit]

    frames = []
    for p in paths:
        img = imread_ja(p)
        if img is None:
            print(f"[warn] 画像を読めません（スキップ）: {p}")
            continue
        frames.append(fit_native(img, cam_name))

    if frames:
        src = f"real:{image or image_dir} ({len(frames)}枚)"
        return frames, src

    if paths:
        print("[warn] 指定パスから1枚も読めませんでした。擬似フレームへフォールバックします。")
    return [make_synthetic_frame(cam_name)], "synthetic(赤い果実＋果柄)"


def fit_native(img: np.ndarray, cam_name: str) -> np.ndarray:
    """画像をカメラのネイティブ解像度へ合わせる（既に一致していれば何もしない）。"""
    n = NATIVE_SIZE.get(cam_name)
    if n is None or (img.shape[0] == n and img.shape[1] == n):
        return img
    return cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)


# ==========================================================
# 統計
# ==========================================================
def stats_ms(samples) -> dict:
    """計測列[ms]から代表値をまとめる。平均は外れ値に弱いので中央値も必ず併記する。"""
    a = np.asarray(samples, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean_ms": round(float(a.mean()), 3),
        "p50_ms": round(float(np.percentile(a, 50)), 3),
        "p95_ms": round(float(np.percentile(a, 95)), 3),
        "p99_ms": round(float(np.percentile(a, 99)), 3),
        "min_ms": round(float(a.min()), 3),
        "max_ms": round(float(a.max()), 3),
        "std_ms": round(float(a.std()), 3),
    }
