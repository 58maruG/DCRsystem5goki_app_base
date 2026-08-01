"""
果実検出チューナー（HSV存在検出＋推論ゲート）

4カメラのライブ映像を見ながら、果実検出に関わる閾値をすべてスライダーで調整し、
カメラ別JSON（json/hsv_config_{cam_name}.json）へ保存する。

本ツールの画像処理は module_yolo.ImageProcessor.get_target_info() と
同一手順を再現している（マスク生成・虚像除去・果柄除去は本番と同じ
hsv_mask_utils を直接呼ぶ）。したがって、ここで
「検出あり／推論ON」と表示される状態が、そのまま本番の挙動になる。

調整できる閾値:
  [HSVマスク]   H1/H2 の色相範囲、S・V の範囲（1段目のマスク生成）
  [虚像除去]    reflect_sat_lo/hi, reflect_v_lo/hi
                「淡い かつ 明るい」画素をアクリル板の反射とみなして落とす。
                範囲が全域(0〜255)なら無効。
  [果柄除去]    stem_open … 開処理の半径。これより細い突起（果柄）を切り離す。0で無効。
  [最小面積]    min_blob_area … これ未満のブロブは果実とみなさない。
  [推論ゲート]  fruit_speed_px … 果実の搬送速度[px/フレーム]。
                推論する中心窓の半幅 = 速度 × 枚数 ÷ 2 で自動計算される。
                果実の重心がこの窓に入っている間だけ、1カメラあたり
                INFER_FRAMES_PER_CAM 枚だけYOLO推論が走る。
                枚数そのものは module_yolo.INFER_FRAMES_PER_CAM で変更する。

表示モード:
  検出結果      … 元画像＋採用ブロブの外接矩形＋ゲート2本線
  1段目マスク   … HSV inRange + 開閉処理の直後
  虚像除去後    … remove_reflection_sat 適用後
  果柄除去後    … remove_stem 適用後（＝最終的に面積判定されるマスク）
  ※ 2段目（虚像除去・果柄除去）は候補ブロブ周辺のみを処理するため、
     切り出し範囲外は黒で表示する（本番でも範囲外は捨てられるため挙動は同じ）。
"""

import sys
import json
import os
import threading
import cv2
import numpy as np
from pypylon import pylon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QSpinBox,
    QGroupBox, QGridLayout, QTabWidget, QComboBox,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

# standalone/ から実行しても親ディレクトリのモジュールを解決できるようにする
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hsv_mask_utils import (  # noqa: E402
    mask_from_hsv, remove_stem, remove_reflection_sat,
)

# ==========================================================
# 設定
# ==========================================================
TARGET_SERIALS = [
    ("25453227", "cam_top"),
    ("25453229", "cam_under"),
    ("25308967", "cam_inside"),
    ("25308968", "cam_outside"),
]

PREVIEW_W, PREVIEW_H = 380, 380   # プレビュー表示サイズ（4画面一覧用）

# グリッド配置: (cam_name, row, col)  ← ここを変えるだけで並びを変更できる
GRID_LAYOUT = [
    ("cam_inside",  0, 0),   # 左上
    ("cam_outside", 0, 1),   # 右上
    ("cam_under",   1, 0),   # 左下
    ("cam_top",     1, 1),   # 右下
]

HSV_JSON_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "json")

# カメラ別JSONがまだ無い状態でツールを開いたときのスライダー初期値。
# 「虚像除去・果柄除去とも無効」＝旧実装と同一マスクになる値。
# 本番(module_yolo)はJSON必須で既定値を持たないため、ここはツール専用の初期値。
DEFAULT_CFG = {
    "lower1": [0, 100, 100],   "upper1": [32, 255, 255],
    "lower2": [160, 100, 100], "upper2": [180, 255, 255],
    "reflect_sat_lo": 0, "reflect_sat_hi": 255,
    "reflect_v_lo":   0, "reflect_v_hi":   255,
    "stem_open":      0,
    "min_blob_area":  500,
}

# JSONに fruit_speed_px が無い場合のみ使うツール専用の初期値
#   （module_yolo.INFER_FRAMES_PER_CAM とは値を必ず揃えること）。
DEFAULT_FRUIT_SPEED_PX = {
    "cam_top": 87, "cam_under": 87, "cam_inside": 91, "cam_outside": 87,
}
FALLBACK_FRUIT_SPEED_PX = 87
INFER_FRAMES_PER_CAM = 5   # module_yolo 側と必ず揃えること


def window_half_px(speed: int) -> int:
    """推論する中心窓の半幅[px]。module_yolo.infer_window_px と同じ式。"""
    return max(1, int(round(speed * INFER_FRAMES_PER_CAM / 2.0)))

# スライダー定義: グループ名 → [(表示名, 内部キー, 最小値, 最大値, 説明)]
GROUP_DEFS = [
    ("HSVマスク（1段目）", [
        ("H1 最小", "h1_lo", 0, 180, "赤色範囲1の色相 下限"),
        ("H1 最大", "h1_hi", 0, 180, "赤色範囲1の色相 上限"),
        ("H2 最小", "h2_lo", 0, 180, "赤色範囲2の色相 下限"),
        ("H2 最大", "h2_hi", 0, 180, "赤色範囲2の色相 上限"),
        ("S  最小", "s_lo",  0, 255, "彩度の下限（H1/H2共通）。上げると淡い色を拾わなくなる"),
        ("S  最大", "s_hi",  0, 255, "彩度の上限（H1/H2共通）"),
        ("V  最小", "v_lo",  0, 255, "明度の下限（H1/H2共通）。上げると暗部を拾わなくなる"),
        ("V  最大", "v_hi",  0, 255, "明度の上限（H1/H2共通）"),
    ]),
    ("虚像除去（アクリル反射）", [
        ("S 下限", "reflect_sat_lo", 0, 255, "虚像とみなす彩度の下限"),
        ("S 上限", "reflect_sat_hi", 0, 255, "虚像とみなす彩度の上限。255のままだと彩度条件は無効"),
        ("V 下限", "reflect_v_lo",   0, 255, "虚像とみなす明度の下限。0のままだと明度条件は無効"),
        ("V 上限", "reflect_v_hi",   0, 255, "虚像とみなす明度の上限"),
    ]),
    ("果柄除去・面積", [
        ("果柄 半径", "stem_open",     0,  40, "開処理の半径。これより細い突起を切り離す。0で無効"),
        ("最小面積",  "min_blob_area", 0, 50000, "これ未満のブロブは果実とみなさない"),
    ]),
    ("推論ゲート", [
        ("搬送速度", "fruit_speed_px", 1, 300,
         "果実が1フレームで進む距離[px]。推論する中心窓の半幅＝速度×枚数÷2 で決まる"),
    ]),
]

# 全スライダーキーを平坦化（順序保持）
ALL_KEYS = [d[1] for _, defs in GROUP_DEFS for d in defs]

VIEW_MODES = ["検出結果", "1段目マスク", "虚像除去後", "果柄除去後"]

# 端接触の判定マージン[px]（get_target_info と同値）
EDGE_MARGIN = 5


# ==========================================================
# JSON 入出力
# ==========================================================
def config_path(cam_name: str) -> str:
    return os.path.join(HSV_JSON_DIR, f"hsv_config_{cam_name}.json")


def load_raw_config(cam_name: str) -> dict:
    """カメラ別JSONを読む。読めない場合は既定値のみを返す（ツールは起動を続ける）。"""
    path = config_path(cam_name)
    if not os.path.exists(path):
        print(f"[{cam_name}] 設定ファイルが見つかりません（既定値で開始）: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as e:
        print(f"[{cam_name}] 設定の読込に失敗しました（既定値で開始）: {path} - {e}")
        return {}


def save_config(cam_name: str, raw: dict) -> str:
    """既存JSONの未知キー（_comment など）を保ったまま上書き保存する。"""
    path = config_path(cam_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=4, ensure_ascii=False)
    return path


def flat_from_raw(raw: dict, cam_name: str) -> dict:
    """JSON構造 → スライダー用の平坦な辞書。欠けているキーは既定値で補う。"""
    cfg = dict(DEFAULT_CFG)
    cfg.update({k: v for k, v in raw.items() if k in DEFAULT_CFG})
    speed = raw.get("fruit_speed_px",
                    DEFAULT_FRUIT_SPEED_PX.get(cam_name, FALLBACK_FRUIT_SPEED_PX))
    return {
        "h1_lo": cfg["lower1"][0], "h1_hi": cfg["upper1"][0],
        "h2_lo": cfg["lower2"][0], "h2_hi": cfg["upper2"][0],
        "s_lo":  cfg["lower1"][1], "s_hi":  cfg["upper1"][1],
        "v_lo":  cfg["lower1"][2], "v_hi":  cfg["upper1"][2],
        "reflect_sat_lo": int(cfg["reflect_sat_lo"]),
        "reflect_sat_hi": int(cfg["reflect_sat_hi"]),
        "reflect_v_lo":   int(cfg["reflect_v_lo"]),
        "reflect_v_hi":   int(cfg["reflect_v_hi"]),
        "stem_open":      int(cfg["stem_open"]),
        "min_blob_area":  int(cfg["min_blob_area"]),
        "fruit_speed_px": int(speed),
    }


def raw_from_flat(flat: dict, base: dict, roi_width: int | None) -> dict:
    """スライダー値 → JSON構造。base の未知キー（_comment 等）は残す。"""
    raw = dict(base)
    raw.update({
        "lower1": [flat["h1_lo"], flat["s_lo"], flat["v_lo"]],
        "upper1": [flat["h1_hi"], flat["s_hi"], flat["v_hi"]],
        "lower2": [flat["h2_lo"], flat["s_lo"], flat["v_lo"]],
        "upper2": [flat["h2_hi"], flat["s_hi"], flat["v_hi"]],
        "reflect_sat_lo": flat["reflect_sat_lo"],
        "reflect_sat_hi": flat["reflect_sat_hi"],
        "reflect_v_lo":   flat["reflect_v_lo"],
        "reflect_v_hi":   flat["reflect_v_hi"],
        "stem_open":      flat["stem_open"],
        "min_blob_area":  flat["min_blob_area"],
        "fruit_speed_px": flat["fruit_speed_px"],
    })
    if roi_width:
        raw["roi_width_at_tuning"] = int(roi_width)
    return raw


def mask_params(flat: dict) -> dict:
    """mask_from_hsv に渡す形式へ変換する。"""
    return {
        "lower1": [flat["h1_lo"], flat["s_lo"], flat["v_lo"]],
        "upper1": [flat["h1_hi"], flat["s_hi"], flat["v_hi"]],
        "lower2": [flat["h2_lo"], flat["s_lo"], flat["v_lo"]],
        "upper2": [flat["h2_hi"], flat["s_hi"], flat["v_hi"]],
    }


# ==========================================================
# 画像処理（get_target_info と同一手順）
# ==========================================================
def analyze(frame: np.ndarray, flat: dict) -> dict:
    """1フレームを本番と同じ手順で処理し、各段階のマスクと判定結果を返す。

    status: 'ok'（検出あり） / 'no_blob'（候補なし） / 'too_small'（面積不足）
            / 'edge'（画面端に接触して棄却）
    """
    h, w = frame.shape[:2]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    res = {
        "status": "no_blob", "area": 0, "stat": None,
        "mask1": None, "mask_reflect": None, "mask_final": None,
        "labels": None, "max_index": 0, "in_band": False,
    }

    # ── 1段目: 果実検出用HSVマスク ─────────────────────────────
    mask1 = mask_from_hsv(hsv, mask_params(flat))
    res["mask1"] = res["mask_reflect"] = res["mask_final"] = mask1

    n1, labels1, stats1, cents1 = cv2.connectedComponentsWithStats(mask1)
    if n1 <= 1:
        return res

    idx = int(np.argmax(stats1[1:, cv2.CC_STAT_AREA])) + 1
    min_area = int(flat["min_blob_area"])
    if stats1[idx, cv2.CC_STAT_AREA] < min_area:
        res["status"] = "too_small"
        res["area"]   = int(stats1[idx, cv2.CC_STAT_AREA])
        res["stat"]   = stats1[idx]
        return res

    # 2段目（整形）が必要かの判定は _load_hsv_config の _refine と同じ規則
    s_active = not (flat["reflect_sat_lo"] <= 0 and flat["reflect_sat_hi"] >= 255)
    v_active = not (flat["reflect_v_lo"]   <= 0 and flat["reflect_v_hi"]   >= 255)
    refine   = bool(flat["stem_open"] > 0 or s_active or v_active)

    if not refine:
        s = stats1[idx]
        res["area"] = int(s[cv2.CC_STAT_AREA])
        res["stat"] = s
        if _touches_edge(s, w, h):
            res["status"] = "edge"
            return res
        res.update(status="ok", labels=labels1, max_index=idx,
                   mx=int(cents1[idx][0]), my=int(cents1[idx][1]))
        res["in_band"] = _check_band(res["mx"], w, window_half_px(flat["fruit_speed_px"]))
        return res

    # ── 2段目: 候補ブロブ周辺だけを切り出して整形する ──────────
    pad = max(int(flat["stem_open"]), 4) + 1
    bx, by = int(stats1[idx, cv2.CC_STAT_LEFT]),  int(stats1[idx, cv2.CC_STAT_TOP])
    bw, bh = int(stats1[idx, cv2.CC_STAT_WIDTH]), int(stats1[idx, cv2.CC_STAT_HEIGHT])
    x0, y0 = max(0, bx - pad),      max(0, by - pad)
    x1, y1 = min(w, bx + bw + pad), min(h, by + bh + pad)

    sub = mask1[y0:y1, x0:x1]
    sub_r = remove_reflection_sat(
        sub, hsv[y0:y1, x0:x1, 1], hsv[y0:y1, x0:x1, 2],
        flat["reflect_sat_lo"], flat["reflect_sat_hi"],
        flat["reflect_v_lo"],   flat["reflect_v_hi"])
    sub_s = remove_stem(sub_r, int(flat["stem_open"]))

    # 表示用: 切り出し範囲外は黒（本番でも範囲外のブロブは捨てられる）
    res["mask_reflect"] = _paste(sub_r, h, w, x0, y0)
    res["mask_final"]   = _paste(sub_s, h, w, x0, y0)

    n2, lab2, st2, ce2 = cv2.connectedComponentsWithStats(sub_s)
    if n2 <= 1:
        return res

    j  = int(np.argmax(st2[1:, cv2.CC_STAT_AREA])) + 1
    s2 = st2[j].copy()
    res["area"] = int(s2[cv2.CC_STAT_AREA])
    s2[cv2.CC_STAT_LEFT] += x0
    s2[cv2.CC_STAT_TOP]  += y0
    res["stat"] = s2

    if res["area"] < min_area:
        res["status"] = "too_small"
        return res
    if _touches_edge(s2, w, h):
        res["status"] = "edge"
        return res

    labels_full = np.zeros((h, w), dtype=np.int32)
    labels_full[y0:y1, x0:x1] = (lab2 == j)
    res.update(status="ok", labels=labels_full, max_index=1,
               mx=int(ce2[j][0]) + x0, my=int(ce2[j][1]) + y0)
    res["in_band"] = _check_band(res["mx"], w, window_half_px(flat["fruit_speed_px"]))
    return res


def _paste(sub: np.ndarray, h: int, w: int, x0: int, y0: int) -> np.ndarray:
    full = np.zeros((h, w), dtype=np.uint8)
    full[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] = sub
    return full


def _touches_edge(stat, w: int, h: int) -> bool:
    return bool(stat[0] <= EDGE_MARGIN or stat[1] <= EDGE_MARGIN
                or (stat[0] + stat[2]) >= (w - EDGE_MARGIN)
                or (stat[1] + stat[3]) >= (h - EDGE_MARGIN))


def band_lines(w: int, half: int) -> tuple[int, int]:
    cx = w / 2.0
    xl = min(max(int(round(cx - half)), 0), w - 1)
    xr = min(max(int(round(cx + half)), 0), w - 1)
    return xl, xr


def _check_band(mx: int, w: int, half: int) -> bool:
    """果実の重心が中心窓の内側にあるか（evaluate_frame と同一判定）。
    本番はこれに加えて「1カメラあたりの推論枚数の上限」でも打ち切るため、
    本ツールの表示は『窓の内側にいる＝推論の対象になりうる』を意味する。"""
    return abs(mx - w / 2.0) <= half


def render(frame: np.ndarray, res: dict, flat: dict, mode: str) -> np.ndarray:
    """表示モードに応じた描画済みBGR画像を返す。"""
    if mode == "1段目マスク":
        base = cv2.cvtColor(res["mask1"], cv2.COLOR_GRAY2BGR)
    elif mode == "虚像除去後":
        base = cv2.cvtColor(res["mask_reflect"], cv2.COLOR_GRAY2BGR)
    elif mode == "果柄除去後":
        base = cv2.cvtColor(res["mask_final"], cv2.COLOR_GRAY2BGR)
    else:
        base = frame.copy()

    h, w = base.shape[:2]

    # 推論する中心窓の2本線（重心が内側なら緑＝推論対象、外なら橙）
    xl, xr = band_lines(w, window_half_px(flat["fruit_speed_px"]))
    band_color = (0, 255, 0) if res["in_band"] else (0, 165, 255)
    cv2.line(base, (xl, 0), (xl, h - 1), band_color, 2)
    cv2.line(base, (xr, 0), (xr, h - 1), band_color, 2)

    stat = res["stat"]
    if stat is not None:
        color = {"ok": (0, 255, 0), "too_small": (0, 0, 255),
                 "edge": (0, 255, 255)}.get(res["status"], (0, 0, 255))
        x, y, bw, bh = int(stat[0]), int(stat[1]), int(stat[2]), int(stat[3])
        cv2.rectangle(base, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(base, f"{res['area']:,}", (x, max(y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 判定に使う重心を明示する（窓の内外はこの点で決まる）
    if res.get("mx") is not None:
        cv2.drawMarker(base, (int(res["mx"]), int(res["my"])), band_color,
                       cv2.MARKER_CROSS, 26, 2)
    return base


def to_pixmap(frame: np.ndarray, w: int, h: int) -> QPixmap:
    rgb = np.ascontiguousarray(
        cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2RGB))
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())


# ==========================================================
# カメラキャプチャスレッド
# ==========================================================
class CameraThread:
    def __init__(self, serial: str, cam_name: str):
        self.serial = serial
        self.cam_name = cam_name
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop_flag = False
        self.error: str | None = None

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._stop_flag = True

    def _run(self):
        cam = None
        try:
            tl = pylon.TlFactory.GetInstance()
            device = next(
                (d for d in tl.EnumerateDevices() if d.GetSerialNumber() == self.serial),
                None,
            )
            if device is None:
                self.error = f"カメラが見つかりません (serial={self.serial})"
                print(f"[{self.cam_name}] {self.error}")
                return

            cam = pylon.InstantCamera(tl.CreateDevice(device))
            cam.Open()

            pfs_path = os.path.join(
                os.path.dirname(HSV_JSON_DIR), "cam_pfs",
                f"{self.cam_name}_{self.serial}.pfs")
            if os.path.exists(pfs_path):
                try:
                    pylon.FeaturePersistence.Load(pfs_path, cam.GetNodeMap(), True)
                except Exception as e:
                    # 4台同時起動時は帯域がカメラ間で分け合われ、pfs保存値(163MB/s)通りに
                    #   ならないことがある(verify不一致で例外)。module_cameras_5goki.py の
                    #   load_pfs_custom と同様、ここで継続してよい（帯域は下で明示設定する）。
                    print(f"[{self.cam_name}] pfs読込エラー（現設定で継続）: {e}")
            else:
                print(f"[{self.cam_name}] pfsが見つかりません（カメラ現設定で継続）: {pfs_path}")

            # 帯域上限はpfsの値によらずここで明示固定する（module_cameras_5goki.py と同じ対策）。
            #   先に設定しないとpfs側の値(163MB/s等)で上書きされ得るため、pfs読み込み後に行う。
            if hasattr(cam, 'DeviceLinkThroughputLimit'):
                cam.DeviceLinkThroughputLimitMode.Value = "On"
                cam.DeviceLinkThroughputLimit.Value = 80000000  # 80MB/s

            conv = pylon.ImageFormatConverter()
            conv.OutputPixelFormat = pylon.PixelType_BGR8packed
            conv.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            while not self._stop_flag and cam.IsGrabbing():
                result = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if result.GrabSucceeded():
                    with self._lock:
                        self._frame = conv.Convert(result).GetArray().copy()
                result.Release()
        except Exception as e:
            self.error = str(e)
            print(f"[{self.cam_name}] エラー: {e}")
        finally:
            try:
                if cam is not None:
                    if cam.IsGrabbing():
                        cam.StopGrabbing()
                    if cam.IsOpen():
                        cam.Close()
            except Exception as e:
                print(f"[{self.cam_name}] 終了処理でエラー: {e}")


# ==========================================================
# 1カメラ分のスライダーパネル
# ==========================================================
class ParamPanel(QWidget):
    def __init__(self, cam_name: str):
        super().__init__()
        self.cam_name = cam_name
        self._raw = load_raw_config(cam_name)
        self._flat = flat_from_raw(self._raw, cam_name)
        self._roi_width: int | None = None
        self._sliders: dict[str, QSlider] = {}
        self._spins: dict[str, QSpinBox] = {}
        self._build_ui()
        self._apply_flat_to_widgets()

    # --------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        for group_name, defs in GROUP_DEFS:
            grp = QGroupBox(group_name)
            grid = QGridLayout(grp)
            grid.setContentsMargins(6, 4, 6, 4)
            grid.setSpacing(3)
            grid.setColumnStretch(1, 1)

            for row, (name, key, lo, hi, tip) in enumerate(defs):
                lbl = QLabel(name)
                lbl.setFixedWidth(72)
                lbl.setToolTip(tip)

                slider = QSlider(Qt.Horizontal)
                slider.setRange(lo, hi)
                slider.setFixedHeight(18)
                slider.setToolTip(tip)

                spin = QSpinBox()
                spin.setRange(lo, hi)
                spin.setFixedWidth(62)
                spin.setButtonSymbols(QSpinBox.NoButtons)
                spin.setToolTip(tip)

                # 双方向同期（同値なら valueChanged が再発火しないため無限ループにならない）
                slider.valueChanged.connect(spin.setValue)
                spin.valueChanged.connect(slider.setValue)
                slider.valueChanged.connect(self._on_change)

                self._sliders[key] = slider
                self._spins[key] = spin

                grid.addWidget(lbl,    row, 0)
                grid.addWidget(slider, row, 1)
                grid.addWidget(spin,   row, 2)

            layout.addWidget(grp)

        btn_row = QHBoxLayout()
        btn_save = QPushButton("保存")
        btn_save.setFixedHeight(30)
        btn_save.clicked.connect(self.save)
        btn_reload = QPushButton("JSON再読込")
        btn_reload.setFixedHeight(30)
        btn_reload.clicked.connect(self.reload)
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_reload)
        layout.addLayout(btn_row)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        layout.addStretch()

    # --------------------------------------------------
    def _apply_flat_to_widgets(self):
        for key in ALL_KEYS:
            val = int(self._flat[key])
            slider, spin = self._sliders[key], self._spins[key]
            val = max(slider.minimum(), min(slider.maximum(), val))
            slider.blockSignals(True)
            spin.blockSignals(True)
            slider.setValue(val)
            spin.setValue(val)
            slider.blockSignals(False)
            spin.blockSignals(False)
        self._on_change()

    def _on_change(self):
        self._flat = {key: self._sliders[key].value() for key in ALL_KEYS}

    # --------------------------------------------------
    def set_roi_width(self, width: int):
        """ROI幅が判明した時点で、搬送速度スライダーの上限を「窓がROI幅を超えない値」に合わせる。"""
        if self._roi_width == width:
            return
        self._roi_width = width
        # 窓の半幅 = 速度×枚数/2 が ROI幅/2 を超えないようにする
        upper = max(1, int(width / INFER_FRAMES_PER_CAM))
        for w in (self._sliders["fruit_speed_px"], self._spins["fruit_speed_px"]):
            w.blockSignals(True)
            w.setMaximum(upper)
            w.blockSignals(False)
        self._sliders["fruit_speed_px"].setValue(
            min(self._sliders["fruit_speed_px"].value(), upper))
        self._on_change()

    def get_flat(self) -> dict:
        return self._flat

    def save(self):
        try:
            raw = raw_from_flat(self._flat, self._raw, self._roi_width)
            path = save_config(self.cam_name, raw)
            self._raw = raw
            self._status.setText(f"保存しました: {os.path.basename(path)}")
            self._status.setStyleSheet("color:#00aa33;")
            return True
        except Exception as e:
            self._status.setText(f"保存に失敗しました: {e}")
            self._status.setStyleSheet("color:#cc0000;")
            print(f"[{self.cam_name}] 保存に失敗しました: {e}")
            return False

    def reload(self):
        self._raw = load_raw_config(self.cam_name)
        self._flat = flat_from_raw(self._raw, self.cam_name)
        self._apply_flat_to_widgets()
        self._status.setText("JSONから再読込しました")
        self._status.setStyleSheet("color:#555555;")


# ==========================================================
# メインウィンドウ
# ==========================================================
class TunerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("果実検出チューナー（HSV存在検出＋推論ゲート）")
        self.resize(1500, 900)

        self._threads: dict[str, CameraThread] = {}
        self._previews: dict[str, QLabel] = {}
        self._infos: dict[str, QLabel] = {}
        self._panels: dict[str, ParamPanel] = {}

        self._build_ui()
        self._start_cameras()

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)  # 20 fps

    # --------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # ── 左: 4カメラのプレビュー ─────────────────────────
        left = QVBoxLayout()
        left.setSpacing(6)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("表示モード:"))
        self._mode = QComboBox()
        self._mode.addItems(VIEW_MODES)
        self._mode.setFixedWidth(160)
        mode_row.addWidget(self._mode)
        mode_row.addStretch()
        left.addLayout(mode_row)

        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setSpacing(8)
        for cam_name, row, col in GRID_LAYOUT:
            grid.addWidget(self._make_camera_cell(cam_name), row, col)
        left.addWidget(grid_widget)
        left.addStretch()
        root.addLayout(left)

        # ── 右: カメラ別スライダー（タブ切替）───────────────
        right = QVBoxLayout()
        right.setSpacing(6)

        self._tabs = QTabWidget()
        self._tabs.setFixedWidth(340)
        for cam_name, _, _ in GRID_LAYOUT:
            panel = ParamPanel(cam_name)
            self._panels[cam_name] = panel
            self._tabs.addTab(panel, cam_name.replace("cam_", ""))
        right.addWidget(self._tabs)

        btn_all = QPushButton("▼ 全カメラを一括保存")
        btn_all.setFixedHeight(36)
        btn_all.clicked.connect(self._save_all)
        right.addWidget(btn_all)
        root.addLayout(right)

        self._sb = self.statusBar()
        self._sb.showMessage("カメラ接続中...")

    def _make_camera_cell(self, cam_name: str) -> QGroupBox:
        grp = QGroupBox(cam_name)
        v = QVBoxLayout(grp)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(3)

        preview = QLabel()
        preview.setFixedSize(PREVIEW_W, PREVIEW_H)
        preview.setAlignment(Qt.AlignCenter)
        preview.setStyleSheet("background:#1a1a1a;")
        v.addWidget(preview)

        info = QLabel("接続待ち...")
        info.setAlignment(Qt.AlignCenter)
        info.setFixedHeight(36)
        info.setStyleSheet("font-size:12px;")
        v.addWidget(info)

        self._previews[cam_name] = preview
        self._infos[cam_name] = info
        return grp

    # --------------------------------------------------
    def _start_cameras(self):
        for serial, cam_name in TARGET_SERIALS:
            t = CameraThread(serial, cam_name)
            t.start()
            self._threads[cam_name] = t
        self._sb.showMessage("カメラ起動中（映像が表示されるまで少しお待ちください）")

    # --------------------------------------------------
    def _tick(self):
        mode = self._mode.currentText()
        for cam_name, _, _ in GRID_LAYOUT:
            thread = self._threads.get(cam_name)
            info = self._infos[cam_name]
            if thread is None:
                continue

            frame = thread.get_frame()
            if frame is None:
                if thread.error:
                    info.setText(f"接続エラー\n{thread.error}")
                    info.setStyleSheet("color:#cc4444; font-size:11px;")
                continue

            panel = self._panels[cam_name]
            panel.set_roi_width(frame.shape[1])
            try:
                res = analyze(frame, panel.get_flat())
                view = render(frame, res, panel.get_flat(), mode)
            except Exception as e:
                # 1台の処理失敗で全体を止めない
                info.setText(f"処理エラー: {e}")
                info.setStyleSheet("color:#cc4444; font-size:11px;")
                continue

            self._previews[cam_name].setPixmap(to_pixmap(view, PREVIEW_W, PREVIEW_H))
            self._update_info(info, res, panel.get_flat())

    @staticmethod
    def _update_info(info: QLabel, res: dict, flat: dict):
        area = res["area"]
        area_txt = f"{area:,}" if area else "--"
        half = window_half_px(flat["fruit_speed_px"])
        if res["status"] == "ok":
            band = (f"推論窓の内側 ✓（±{half}px / {INFER_FRAMES_PER_CAM}枚）"
                    if res["in_band"] else f"推論窓の外（±{half}px）")
            info.setText(f"面積: {area_txt} ／ 検出: あり ✓\n{band}")
            color = "#00aa33" if res["in_band"] else "#cc8800"
            info.setStyleSheet(f"color:{color}; font-size:12px; font-weight:bold;")
            return

        reason = {
            "no_blob":   "候補なし（HSV範囲を広げる）",
            "too_small": "面積不足（最小面積を下げる／マスクを広げる）",
            "edge":      "画面端に接触（対象外）",
        }.get(res["status"], res["status"])
        info.setText(f"面積: {area_txt} ／ 検出: なし\n{reason}")
        info.setStyleSheet("color:#cc4444; font-size:12px;")

    # --------------------------------------------------
    def _save_all(self):
        ok, ng = [], []
        for cam_name, _, _ in GRID_LAYOUT:
            (ok if self._panels[cam_name].save() else ng).append(cam_name)
        msg = f"保存完了: {', '.join(ok)}" if ok else ""
        if ng:
            msg += f" ／ 失敗: {', '.join(ng)}"
        self._sb.showMessage(msg)

    def closeEvent(self, event):
        self._timer.stop()
        for t in self._threads.values():
            t.stop()
        event.accept()


# ==========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = TunerWindow()
    win.show()
    sys.exit(app.exec())
