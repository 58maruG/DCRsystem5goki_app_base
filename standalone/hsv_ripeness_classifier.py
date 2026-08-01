"""
果実熟度分類器（赤色占有率スコアによるhealthy/unripe振り分け）

2種類のHSVマスク（果実全体・赤色のみ）を使って赤色占有率を計算し、
healthy と unripe に振り分けて保存するGUI。

  赤色占有率 = 果実内の赤色面積 ÷ 果実全体面積
  ・占有率 ≧ 閾値 → healthy（緑枠）
  ・占有率 ＜ 閾値 かつ果実検出あり → unripe（橙枠）
  ・果実マスクで未検出 → 除外（赤枠）

使い方:
  1. 「果実マスク」スライダーで黄〜橙〜赤を包括するHSVを設定する。
  2. 「赤色マスク」スライダーで赤色のみを検出するHSVを設定する。
  3. 「赤色閾値(%)」で healthy / unripe の境界を調整する。
  4. 緑枠=healthy, 橙枠=unripe, 赤枠=未検出 を確認して設定を保存する。
  5. 保存ボタンで画像を振り分け出力する。

保存先:
  healthy : train用_v4_ripeness_healthy/{class_name}/
  unripe  : train用_v4_ripeness_unripe/{class_name}/
"""

import sys
import os
import json
import shutil
import cv2
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton,
    QScrollArea, QListWidget, QListWidgetItem,
    QGridLayout, QGroupBox,
    QProgressDialog, QMessageBox,
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont


# ==========================================================
# 設定
# ==========================================================
TRAIN_DIR   = r"C:\Users\kotan\gohara\cherry_yolo\model作成用imageset\all\overripe"
THUMB_SIZE  = 112
GRID_COLS   = 7
DEBOUNCE_MS = 300

_TRAIN_PATH = Path(TRAIN_DIR)
HEALTHY_DIR = _TRAIN_PATH.parent / f"{_TRAIN_PATH.name}_ripeness_healthy"
UNRIPE_DIR  = _TRAIN_PATH.parent / f"{_TRAIN_PATH.name}_ripeness_unripe"

_JSON_DIR    = Path(__file__).parent.parent / "json"
_CONFIG_PATH = _JSON_DIR / "_hsv_ripeness_config.json"

# 果実全体マスクのデフォルト（黄〜橙〜赤を広くカバー）
DEFAULT_FRUIT_HSV = {
    "lower1": [0,   100, 100],
    "upper1": [32,  255, 255],
    "lower2": [160, 100, 100],
    "upper2": [180, 255, 255],
}

# 赤色マスクのデフォルト（H が低い赤 + H が高い赤）
DEFAULT_RED_HSV = {
    "lower1": [0,   60, 100],
    "upper1": [11,  255, 255],
    "lower2": [166, 60, 100],
    "upper2": [180, 255, 255],
}

DEFAULT_THRESHOLD = 25   # 赤色占有率の初期閾値（%）
DEFAULT_MIN_AREA  = 300  # 果実マスクの最小面積（サムネイル換算px²）

SLIDER_DEFS = [
    ("H1 最小", "lower1", 0, 180),
    ("H1 最大", "upper1", 0, 180),
    ("H2 最小", "lower2", 0, 180),
    ("H2 最大", "upper2", 0, 180),
    ("S  最小", "lower1", 1, 255),
    ("S  最大", "upper1", 1, 255),
    ("V  最小", "lower1", 2, 255),
    ("V  最大", "upper1", 2, 255),
]

COLOR_HEALTHY = QColor(0,  210,  0)
COLOR_UNRIPE  = QColor(230, 140,  0)
COLOR_NONE    = QColor(210,   0,  0)


# ==========================================================
# JSON 入出力
# ==========================================================
def load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {
        "fruit":     {k: list(v) for k, v in DEFAULT_FRUIT_HSV.items()},
        "red":       {k: list(v) for k, v in DEFAULT_RED_HSV.items()},
        "threshold": DEFAULT_THRESHOLD,
        "min_area":  DEFAULT_MIN_AREA,
    }


def save_config(cfg: dict):
    _JSON_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)


# ==========================================================
# 日本語パス対応
# ==========================================================
def imread_unicode(path: str) -> np.ndarray | None:
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    try:
        ext = Path(path).suffix.lower()
        ret, buf = cv2.imencode(ext, img)
        if not ret:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False


# ==========================================================
# 画像処理
# ==========================================================
def compute_mask(frame: np.ndarray, params: dict) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array(params["lower1"]), np.array(params["upper1"]))
    m2 = cv2.inRange(hsv, np.array(params["lower2"]), np.array(params["upper2"]))
    mask = cv2.bitwise_or(m1, m2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def compute_ripeness(
    frame: np.ndarray,
    fruit_params: dict,
    red_params: dict,
    min_area: int,
) -> tuple[float, bool, list]:
    """
    Returns:
        score      : 赤色占有率 (0.0〜1.0)
        fruit_found: min_area 以上の果実領域が存在するか
        rects      : 果実矩形リスト（min_area 以上のもの）
    """
    fruit_mask = compute_mask(frame, fruit_params)
    red_mask   = compute_mask(frame, red_params)
    # 果実マスク内に限定した赤色面積を計算
    red_in_fruit = cv2.bitwise_and(red_mask, fruit_mask)

    fruit_area = int(np.count_nonzero(fruit_mask))
    red_area   = int(np.count_nonzero(red_in_fruit))
    score = red_area / fruit_area if fruit_area > 0 else 0.0

    contours, _ = cv2.findContours(
        fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    rects = [
        cv2.boundingRect(c) for c in contours
        if cv2.contourArea(c) >= min_area
    ]
    return score, len(rects) > 0, rects


def make_thumb_pixmap(
    thumb: np.ndarray,
    size: int,
    score: float,
    fruit_found: bool,
    rects: list,
    threshold: float,
) -> QPixmap:
    """スコアと閾値で枠色を決定し、左上に占有率を描画したサムネイルを返す。"""
    if not fruit_found:
        border_color = COLOR_NONE
        label_text   = "---"
    elif score >= threshold:
        border_color = COLOR_HEALTHY
        label_text   = f"{score * 100:.0f}%"
    else:
        border_color = COLOR_UNRIPE
        label_text   = f"{score * 100:.0f}%"

    annotated = thumb.copy()
    for (rx, ry, rw, rh) in rects:
        cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (0, 220, 0), 2)

    h, w = annotated.shape[:2]
    scale = min(size / w, size / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = cv2.resize(annotated, (nw, nh))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data.tobytes(), nw, nh, 3 * nw, QImage.Format_RGB888)

    canvas = QPixmap(size, size)
    canvas.fill(QColor(28, 28, 28))
    painter = QPainter(canvas)
    painter.drawImage((size - nw) // 2, (size - nh) // 2, qimg)
    pen = QPen(border_color, 3)
    painter.setPen(pen)
    painter.drawRect(1, 1, size - 3, size - 3)
    painter.setFont(QFont("", 9, QFont.Bold))
    painter.setPen(QPen(border_color))
    painter.drawText(4, 13, label_text)
    painter.end()
    return canvas


# ==========================================================
# サムネイル生成バックグラウンドスレッド
# ==========================================================
class ThumbWorker(QThread):
    progress = Signal(int)
    finished = Signal(list)  # list[tuple[str, np.ndarray]]

    def __init__(self, paths: list):
        super().__init__()
        self.paths = paths

    def run(self):
        entries = []
        for i, path in enumerate(self.paths):
            img = imread_unicode(path)
            if img is not None:
                ih, iw = img.shape[:2]
                scale = min(THUMB_SIZE / iw, THUMB_SIZE / ih)
                nw = max(1, int(iw * scale))
                nh = max(1, int(ih * scale))
                thumb = cv2.resize(img, (nw, nh))
                entries.append((path, thumb))
            self.progress.emit(i + 1)
        self.finished.emit(entries)


# ==========================================================
# スコア計算スレッド（選択クラスのみ）
# ==========================================================
class ScoreWorker(QThread):
    finished = Signal(list, list, list)  # scores, fruit_founds, rects_list

    def __init__(
        self,
        items: list,
        fruit_params: dict,
        red_params: dict,
        min_area: int,
    ):
        super().__init__()
        self.items = items
        self.fruit_params = fruit_params
        self.red_params = red_params
        self.min_area = min_area

    def run(self):
        scores, fruit_founds, rects_list = [], [], []
        for _, thumb in self.items:
            s, ff, r = compute_ripeness(
                thumb, self.fruit_params, self.red_params, self.min_area
            )
            scores.append(s)
            fruit_founds.append(ff)
            rects_list.append(r)
        self.finished.emit(scores, fruit_founds, rects_list)


# ==========================================================
# 保存スレッド（元画像スケールで再検出して保存）
# ==========================================================
class SaveWorker(QThread):
    progress = Signal(int)
    finished = Signal(int, int)  # saved_count, total

    def __init__(
        self,
        cls_name: str,
        paths: list,
        fruit_params: dict,
        red_params: dict,
        min_area: int,
        threshold: float,  # 0.0〜1.0
        target: str,       # "healthy" or "unripe"
        save_mode: str,    # "copy" or "crop"
    ):
        super().__init__()
        self.cls_name = cls_name
        self.paths = paths
        self.fruit_params = fruit_params
        self.red_params = red_params
        self.min_area = min_area
        self.threshold = threshold
        self.target = target
        self.save_mode = save_mode

    def run(self):
        dest_dir = (HEALTHY_DIR if self.target == "healthy" else UNRIPE_DIR) / self.cls_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        saved = 0

        for i, path in enumerate(self.paths):
            img = imread_unicode(path)
            if img is not None:
                ih, iw = img.shape[:2]
                scale = min(THUMB_SIZE / iw, THUMB_SIZE / ih)
                # サムネイル換算の面積閾値を元画像スケールに変換
                scaled_min_area = max(1, int(self.min_area / (scale ** 2)))

                score, fruit_found, rects = compute_ripeness(
                    img, self.fruit_params, self.red_params, scaled_min_area
                )

                if fruit_found:
                    is_healthy = score >= self.threshold
                    should_save = (
                        (self.target == "healthy" and is_healthy) or
                        (self.target == "unripe"  and not is_healthy)
                    )
                    if should_save:
                        src = Path(path)
                        if self.save_mode == "copy":
                            shutil.copy2(str(src), str(dest_dir / src.name))
                            saved += 1
                        else:
                            for j, (rx, ry, rw, rh) in enumerate(rects):
                                crop = img[ry:ry + rh, rx:rx + rw]
                                if crop.size > 0:
                                    dest = dest_dir / f"{src.stem}_{j}{src.suffix}"
                                    if imwrite_unicode(str(dest), crop):
                                        saved += 1

            self.progress.emit(i + 1)

        self.finished.emit(saved, len(self.paths))


# ==========================================================
# HSVスライダーグループ（再利用可能ウィジェット）
# ==========================================================
class HsvSliderGroup(QGroupBox):
    changed = Signal()

    def __init__(self, title: str, default: dict, parent=None):
        super().__init__(title, parent)
        grid = QGridLayout(self)
        grid.setSpacing(4)
        grid.setColumnStretch(1, 1)
        self._sliders: list[QSlider] = []
        self._val_labels: list[QLabel] = []

        for row, (name, _, _, max_val) in enumerate(SLIDER_DEFS):
            lbl = QLabel(name)
            lbl.setFixedWidth(60)
            s = QSlider(Qt.Horizontal)
            s.setRange(0, max_val)
            s.setFixedHeight(18)
            s.valueChanged.connect(self._on_changed)
            vl = QLabel("0")
            vl.setFixedWidth(28)
            vl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._sliders.append(s)
            self._val_labels.append(vl)
            grid.addWidget(lbl, row, 0)
            grid.addWidget(s,   row, 1)
            grid.addWidget(vl,  row, 2)

        self.set_params(default)

    def _on_changed(self):
        for s, vl in zip(self._sliders, self._val_labels):
            vl.setText(str(s.value()))
        self.changed.emit()

    def set_params(self, params: dict):
        vals = [
            params["lower1"][0], params["upper1"][0],
            params["lower2"][0], params["upper2"][0],
            params["lower1"][1], params["upper1"][1],
            params["lower1"][2], params["upper1"][2],
        ]
        for s, vl, v in zip(self._sliders, self._val_labels, vals):
            s.blockSignals(True)
            s.setValue(v)
            vl.setText(str(v))
            s.blockSignals(False)

    def get_params(self) -> dict:
        vals = [s.value() for s in self._sliders]
        h1_min, h1_max, h2_min, h2_max, s_min, s_max, v_min, v_max = vals
        return {
            "lower1": [h1_min, s_min, v_min],
            "upper1": [h1_max, s_max, v_max],
            "lower2": [h2_min, s_min, v_min],
            "upper2": [h2_max, s_max, v_max],
        }


# ==========================================================
# サムネイルグリッド
# ==========================================================
class ThumbnailGrid(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self._container = QWidget()
        self._grid = QGridLayout(self._container)
        self._grid.setSpacing(3)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self.setWidget(self._container)
        self._labels: list[QLabel] = []

    def update_grid(
        self,
        items: list,
        scores: list,
        fruit_founds: list,
        rects_list: list,
        threshold: float,
    ):
        for lbl in self._labels:
            self._grid.removeWidget(lbl)
            lbl.deleteLater()
        self._labels.clear()

        for i, ((path, thumb), score, ff, rects) in enumerate(
            zip(items, scores, fruit_founds, rects_list)
        ):
            pix = make_thumb_pixmap(thumb, THUMB_SIZE, score, ff, rects, threshold)
            lbl = QLabel()
            lbl.setPixmap(pix)
            lbl.setFixedSize(THUMB_SIZE + 4, THUMB_SIZE + 4)
            lbl.setToolTip(
                f"{os.path.basename(path)}\n赤色占有率: {score * 100:.1f}%"
            )
            self._grid.addWidget(lbl, *divmod(i, GRID_COLS))
            self._labels.append(lbl)

    def clear_grid(self):
        for lbl in self._labels:
            self._grid.removeWidget(lbl)
            lbl.deleteLater()
        self._labels.clear()


# ==========================================================
# メインウィンドウ
# ==========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("果実熟度分類器（赤色占有率スコア）")
        self.resize(1500, 950)

        cfg = load_config()
        self._fruit_params: dict = cfg["fruit"]
        self._red_params:   dict = cfg["red"]
        self._threshold:    int  = cfg.get("threshold", DEFAULT_THRESHOLD)
        self._min_area:     int  = cfg.get("min_area",  DEFAULT_MIN_AREA)

        self._current_cls:  str  = ""
        self._path_cache: dict[str, list[str]] = {}                    # 起動時に収集（パスのみ）
        self._cache: dict[str, list[tuple[str, np.ndarray]]] = {}      # クラス選択時に生成
        self._scores:       list = []
        self._fruit_founds: list = []
        self._rects_list:   list = []
        self._worker:       ScoreWorker | None = None
        self._thumb_worker: ThumbWorker | None = None
        self._thumb_dlg:    QProgressDialog | None = None
        self._save_worker:  SaveWorker | None  = None
        self._save_dlg:     QProgressDialog | None = None
        self._save_ctx:     tuple[str, str] = ("", "")  # (target, mode_label)

        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._run_score)

        self._build_ui()
        self._load_images()

    # --------------------------------------------------
    # UI 構築
    # --------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── ヘッダー行1: クラス名 + 統計 + 設定保存 ──
        h1 = QHBoxLayout()
        self._cls_label = QLabel("クラス: （未選択）")
        self._cls_label.setFont(QFont("", 13, QFont.Bold))
        h1.addWidget(self._cls_label)

        self._stat_label = QLabel("healthy: -- / unripe: -- / 未検出: --")
        self._stat_label.setFont(QFont("", 11))
        h1.addWidget(self._stat_label)
        h1.addStretch()

        btn_cfg = QPushButton("設定を保存\n(_hsv_ripeness_config.json)")
        btn_cfg.setFixedSize(190, 46)
        btn_cfg.setToolTip("果実マスク・赤色マスク・閾値設定を JSON に保存します。")
        btn_cfg.clicked.connect(self._save_cfg)
        h1.addWidget(btn_cfg)
        root.addLayout(h1)

        # ── ヘッダー行2: 保存ボタン4つ ──
        h2 = QHBoxLayout()
        h2.addStretch()
        save_buttons = [
            ("healthy  コピー保存\n(ripeness_healthy/)",  "healthy", "copy"),
            ("healthy  クロップ保存\n(ripeness_healthy/)", "healthy", "crop"),
            ("unripe   コピー保存\n(ripeness_unripe/)",   "unripe",  "copy"),
            ("unripe   クロップ保存\n(ripeness_unripe/)",  "unripe",  "crop"),
        ]
        for label, target, mode in save_buttons:
            btn = QPushButton(label)
            btn.setFixedSize(175, 46)
            btn.clicked.connect(
                lambda _=False, t=target, m=mode: self._start_save(t, m)
            )
            h2.addWidget(btn)
        root.addLayout(h2)

        # ── メイン（左パネル + 右パネル） ──
        body = QHBoxLayout()
        body.setSpacing(6)
        root.addLayout(body)

        left = QVBoxLayout()
        left.setSpacing(4)
        body.addLayout(left, 0)

        self._fruit_grp = HsvSliderGroup("果実マスク（黄〜橙〜赤 全体）", self._fruit_params)
        self._fruit_grp.changed.connect(self._on_params_changed)
        left.addWidget(self._fruit_grp)

        self._red_grp = HsvSliderGroup("赤色マスク（赤のみ）", self._red_params)
        self._red_grp.changed.connect(self._on_params_changed)
        left.addWidget(self._red_grp)

        left.addWidget(self._build_filter_group())
        left.addWidget(self._build_class_list(), 1)

        self._grid = ThumbnailGrid()
        body.addWidget(self._grid, 1)

        self._sb = self.statusBar()
        self._sb.showMessage("画像読み込み中...")

    def _build_filter_group(self) -> QGroupBox:
        grp = QGroupBox("検出フィルタ")
        grid = QGridLayout(grp)
        grid.setSpacing(4)
        grid.setColumnStretch(1, 1)

        grid.addWidget(QLabel("最小面積"), 0, 0)
        self._area_slider = QSlider(Qt.Horizontal)
        self._area_slider.setRange(0, 3000)
        self._area_slider.setValue(self._min_area)
        self._area_slider.setFixedHeight(18)
        self._area_slider.valueChanged.connect(self._on_area)
        self._area_label = QLabel(str(self._min_area))
        self._area_label.setFixedWidth(48)
        self._area_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(self._area_slider, 0, 1)
        grid.addWidget(self._area_label,  0, 2)

        grid.addWidget(QLabel("赤色閾値(%)"), 1, 0)
        self._thr_slider = QSlider(Qt.Horizontal)
        self._thr_slider.setRange(0, 100)
        self._thr_slider.setValue(self._threshold)
        self._thr_slider.setFixedHeight(18)
        self._thr_slider.valueChanged.connect(self._on_threshold)
        self._thr_label = QLabel(f"{self._threshold}%")
        self._thr_label.setFixedWidth(48)
        self._thr_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(self._thr_slider, 1, 1)
        grid.addWidget(self._thr_label,  1, 2)

        return grp

    def _build_class_list(self) -> QGroupBox:
        grp = QGroupBox("クラス一覧（クリックで切り替え）")
        v = QVBoxLayout(grp)
        self._class_list = QListWidget()
        self._class_list.setFixedWidth(270)
        self._class_list.currentRowChanged.connect(self._on_class_selected)
        v.addWidget(self._class_list)
        return grp

    # --------------------------------------------------
    # 画像読み込み
    # --------------------------------------------------
    def _load_images(self):
        img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        train_path = Path(TRAIN_DIR)

        # 直下に画像がある場合はフラット構造としてフォルダ自体を1クラス扱い
        has_direct_images = any(
            f.is_file() and f.suffix.lower() in img_exts
            for f in train_path.iterdir()
        )
        if has_direct_images:
            class_dirs = [(train_path.name, train_path)]
        else:
            class_dirs = [
                (d.name, d) for d in sorted(train_path.iterdir())
                if d.is_dir() and not d.name.startswith("_")
            ]

        # ファイルパスのみ収集（サムネイル生成はクラス選択時に行う）
        for cls_name, class_dir in class_dirs:
            paths = [
                str(fpath) for fpath in sorted(class_dir.iterdir())
                if fpath.is_file() and fpath.suffix.lower() in img_exts
            ]
            if paths:
                self._path_cache[cls_name] = paths

        self._populate_class_list()
        if self._class_list.count() > 0:
            self._class_list.setCurrentRow(0)

    def _populate_class_list(self):
        self._class_list.clear()
        for cls, paths in self._path_cache.items():
            item = QListWidgetItem(f"  {cls}  ({len(paths)}枚)")
            item.setData(Qt.UserRole, cls)
            self._class_list.addItem(item)

    # --------------------------------------------------
    # クラス選択
    # --------------------------------------------------
    def _on_class_selected(self, row: int):
        if row < 0:
            return
        item = self._class_list.item(row)
        cls = item.data(Qt.UserRole)
        if cls == self._current_cls:
            return
        self._current_cls = cls
        self._cls_label.setText(f"クラス: {cls}")

        if cls in self._cache:
            # サムネイル生成済みならすぐ表示
            items = self._cache[cls]
            n = len(items)
            self._scores       = [0.0]   * n
            self._fruit_founds = [False] * n
            self._rects_list   = [[]]    * n
            self._grid.update_grid(
                items, self._scores, self._fruit_founds, self._rects_list,
                self._threshold / 100.0
            )
            self._run_score()
        else:
            # 未生成ならバックグラウンドで生成
            self._grid.clear_grid()
            self._scores = []
            self._fruit_founds = []
            self._rects_list = []
            self._stat_label.setText("読み込み中...")
            paths = self._path_cache.get(cls, [])
            self._start_thumb_loading(cls, paths)

    def _start_thumb_loading(self, cls_name: str, paths: list):
        if self._thumb_worker and self._thumb_worker.isRunning():
            self._thumb_worker.quit()
            self._thumb_worker.wait()
        if self._thumb_dlg:
            self._thumb_dlg.close()

        n = len(paths)
        self._sb.showMessage(f"サムネイル生成中... ({n} 枚)")
        self._thumb_dlg = QProgressDialog(
            f"画像を読み込んでいます... ({n} 枚)", None, 0, n, self
        )
        self._thumb_dlg.setWindowTitle("読み込み中")
        self._thumb_dlg.setWindowModality(Qt.WindowModal)
        self._thumb_dlg.setMinimumDuration(500)
        self._thumb_dlg.setValue(0)

        self._thumb_worker = ThumbWorker(paths)
        self._thumb_worker.progress.connect(self._thumb_dlg.setValue)
        self._thumb_worker.finished.connect(
            lambda entries: self._on_thumb_done(cls_name, entries)
        )
        self._thumb_worker.start()

    def _on_thumb_done(self, cls_name: str, entries: list):
        if self._thumb_dlg:
            self._thumb_dlg.close()
            self._thumb_dlg = None

        self._cache[cls_name] = entries

        if cls_name == self._current_cls:
            n = len(entries)
            self._scores       = [0.0]   * n
            self._fruit_founds = [False] * n
            self._rects_list   = [[]]    * n
            self._grid.update_grid(
                entries, self._scores, self._fruit_founds, self._rects_list,
                self._threshold / 100.0
            )
            self._run_score()

        self._sb.showMessage(f"読み込み完了: {len(entries)} 枚")

    # --------------------------------------------------
    # スライダー操作
    # --------------------------------------------------
    def _on_params_changed(self):
        self._fruit_params = self._fruit_grp.get_params()
        self._red_params   = self._red_grp.get_params()
        self._debounce.start(DEBOUNCE_MS)

    def _on_area(self, val: int):
        self._min_area = val
        self._area_label.setText(str(val))
        self._debounce.start(DEBOUNCE_MS)

    def _on_threshold(self, val: int):
        # 閾値変更はスコア再計算不要、表示の色判定だけ変わる
        self._threshold = val
        self._thr_label.setText(f"{val}%")
        self._refresh_grid()

    # --------------------------------------------------
    # スコア計算（バックグラウンド）
    # --------------------------------------------------
    def _run_score(self):
        if not self._current_cls:
            return
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait()
        items = self._cache.get(self._current_cls, [])
        if not items:
            return
        self._sb.showMessage("処理中...")
        self._worker = ScoreWorker(
            items, self._fruit_params, self._red_params, self._min_area
        )
        self._worker.finished.connect(self._on_score_done)
        self._worker.start()

    def _on_score_done(self, scores: list, fruit_founds: list, rects_list: list):
        self._scores       = scores
        self._fruit_founds = fruit_founds
        self._rects_list   = rects_list
        self._refresh_grid()
        self._sb.showMessage("完了")

    # --------------------------------------------------
    # 表示更新
    # --------------------------------------------------
    def _refresh_grid(self):
        if not self._current_cls:
            return
        items = self._cache.get(self._current_cls, [])
        thr = self._threshold / 100.0
        self._grid.update_grid(
            items, self._scores, self._fruit_founds, self._rects_list, thr
        )
        self._update_stat(thr)

    def _update_stat(self, thr: float):
        h = sum(
            1 for s, ff in zip(self._scores, self._fruit_founds)
            if ff and s >= thr
        )
        u = sum(
            1 for s, ff in zip(self._scores, self._fruit_founds)
            if ff and s < thr
        )
        no = sum(1 for ff in self._fruit_founds if not ff)
        self._stat_label.setText(
            f"<span style='color:lime;font-weight:bold;'>healthy: {h}</span>  /  "
            f"<span style='color:orange;font-weight:bold;'>unripe: {u}</span>  /  "
            f"<span style='color:tomato;'>未検出: {no}</span>"
        )

    # --------------------------------------------------
    # 保存
    # --------------------------------------------------
    def _save_cfg(self):
        save_config({
            "fruit":     self._fruit_params,
            "red":       self._red_params,
            "threshold": self._threshold,
            "min_area":  self._min_area,
        })
        self._sb.showMessage("設定を保存しました: json/_hsv_ripeness_config.json")

    def _start_save(self, target: str, save_mode: str):
        if not self._current_cls:
            QMessageBox.warning(self, "警告", "クラスを選択してください。")
            return
        # SaveWorker はパスのみ使うので _path_cache から取得（サムネイル未生成でも可）
        paths = self._path_cache.get(self._current_cls, [])
        if not paths:
            QMessageBox.information(self, "情報", "このクラスに画像がありません。")
            return

        mode_label = "コピー" if save_mode == "copy" else "クロップ"
        thr = self._threshold / 100.0
        count = sum(
            1 for s, ff in zip(self._scores, self._fruit_founds)
            if ff and (
                (target == "healthy" and s >= thr) or
                (target == "unripe"  and s <  thr)
            )
        )
        dest = (HEALTHY_DIR if target == "healthy" else UNRIPE_DIR) / self._current_cls

        reply = QMessageBox.question(
            self, "確認",
            f"クラス「{self._current_cls}」のうち {target} と判定された\n"
            f"約 {count} 枚を\n{dest}\nに{mode_label}保存します。よろしいですか？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._save_ctx = (target, mode_label)
        self._save_dlg = QProgressDialog(
            f"{target} 画像を{mode_label}保存中...", None, 0, len(paths), self
        )
        self._save_dlg.setWindowTitle(f"{mode_label}保存中")
        self._save_dlg.setWindowModality(Qt.WindowModal)
        self._save_dlg.setMinimumDuration(0)
        self._save_dlg.setValue(0)
        self._save_dlg.show()

        self._save_worker = SaveWorker(
            self._current_cls, paths,
            self._fruit_params, self._red_params,
            self._min_area, thr, target, save_mode
        )
        self._save_worker.progress.connect(self._save_dlg.setValue)
        self._save_worker.finished.connect(self._on_save_done)
        self._save_worker.start()

    def _on_save_done(self, saved: int, total: int):
        if self._save_dlg:
            self._save_dlg.close()
            self._save_dlg = None

        target, mode_label = self._save_ctx
        dest = (HEALTHY_DIR if target == "healthy" else UNRIPE_DIR) / self._current_cls

        QMessageBox.information(
            self, "保存完了",
            f"{total} 枚を処理し、{saved} 件を{mode_label}しました。\n"
            f"種別: {target}\n"
            f"保存先: {dest}"
        )
        self._sb.showMessage(f"{target} {mode_label}完了: {saved}/{total} 件")


# ==========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
