# -------------------------------------------------
# DCR_systemのGUIデザイン定義ファイル
# -------------------------------------------------
from __future__ import annotations
import sys
import os
from PySide6.QtWidgets import (
    QWidget, QMainWindow, QLabel, QPushButton, QCheckBox, QVBoxLayout, QGraphicsOpacityEffect
)
from PySide6.QtCore import Qt, Property, QPropertyAnimation, QEasingCurve, QPointF, Signal
from PySide6.QtGui import QPainter, QColor, QBrush, QPixmap

# ================================================
# メインウインドウ レイアウト定数
# ================================================
MARGIN_X = 10                           # 画面の余白x
MARGIN_Y = 10                           # 画面の余白y

MARGIN_CAM = MARGIN_X * 0.5             # 余白xの1/2倍

#WINDOW_W, WINDOW_H = 1280, 800          # 1920×1200 拡大/縮小 150%
#WINDOW_W, WINDOW_H = 1280, 720          # 1920×1080 拡大/縮小 150%
WINDOW_W, WINDOW_H = 1706, 1066         # 2560×1600 拡大/縮小 150%

# --- カメラ（正方形を維持して縮小。2×2で左上に配置）---
VIEW_CAM_SIZE_W = 400
VIEW_CAM_SIZE_H = 400

# --- アイコン ---
ICON_SETTING_SIZE = int(WINDOW_W * 0.09) // 10 * 10             # 設定アイコン（左下へ移動・小型化）
ICON_POWER_SIZE   = int(WINDOW_W * 0.1) // 10 * 10             # 電源アイコン（右上・従来通り）

# --- 右カラム基準（カメラ右端の右側）---
RIGHT_X = (VIEW_CAM_SIZE_W + MARGIN_X) * 2 + MARGIN_CAM   # 右カラム左端
RIGHT_W = WINDOW_W - RIGHT_X - MARGIN_X                            # 右カラム幅

# --- クラス表示リスト（横長）---
LABEL_HISTORY_SIZE_W = RIGHT_W + MARGIN_CAM
LABEL_HISTORY_SIZE_H = 430

# --- ステータスバー（run / stop / estop / error）---
LABEL_STATUS_SIZE_W = RIGHT_W + MARGIN_CAM
LABEL_STATUS_SIZE_H = 64

# --- 累計カウント統計（右下に配置）---
LABEL_STATS_SIZE_W = RIGHT_W * 0.6
LABEL_STATS_SIZE_H = 352

LABEL_MANAGEMENT_SIZE_W = int(WINDOW_W * 0.2) // 10 * 10        # 「システム管理」ラベルサイズw
LABEL_MANAGEMENT_SIZE_H = int(WINDOW_H * 0.1) // 10 * 10        # 「システム管理」ラベルサイズh

SWITCH_TOGGLE_SIZE_W = int(WINDOW_W * 0.2) // 10 * 10           # トグルスイッチサイズw（従来通り）
SWITCH_TOGGLE_SIZE_H = SWITCH_TOGGLE_SIZE_W / 2                  # トグルスイッチサイズh

LABEL_TOGGLE_SIZE_W = SWITCH_TOGGLE_SIZE_W                       # トグルスイッチ状態表示ラベルサイズw
LABEL_TOGGLE_SIZE_H = LABEL_TOGGLE_SIZE_W / 7                    # トグルスイッチ状態表示ラベルサイズh

# --- 左下：モード表示パネル ---
MODE_PANEL_H = 220                                              # モード表示パネル高さ


# ================================================
# クラス名（YOLOラベル）→ 表示情報マッピング
#   jp:    日本語表示名
#   color: クラスリスト（黒背景）上での文字色（黒地で視認できる明るめの色）
# ================================================
CLASS_DISPLAY = {
    "healthy":      {"jp": "健全果",     "color": "#FFFFFF"},
    "twin":         {"jp": "双子果",     "color": "#FF4D4D"},
    "unripe":       {"jp": "未熟果",     "color": "#FFFF66"},
    "mold":         {"jp": "カビ",       "color": "#DB7AE0"},
    "stemcrack":    {"jp": "果梗裂果",   "color": "#6E9BFF"},
    "birddamage":   {"jp": "鳥害",       "color": "#6FA8FF"},
    "malformation": {"jp": "奇形果",     "color": "#FF7A3D"},
    "crack":        {"jp": "裂果",       "color": "#4FC3F7"},
    "wilt":         {"jp": "萎凋果",     "color": "#C8956A"},
    "suturecrack":  {"jp": "縫合線裂果", "color": "#4FD6C7"},
    "brownrot":     {"jp": "灰星病",     "color": "#C98A5E"},
    "blacktwin":    {"jp": "黒双子",     "color": "#AEB9C7"},
    "insect":       {"jp": "虫害",       "color": "#B5D44D"},
    "kasure":       {"jp": "擦れ果",     "color": "#D4B483"},
}

# ================================================
# サブウインドウ レイアウト定数
# ================================================
SUB_WINDOW_W, SUB_WINDOW_H = 600, 500   # サブウインドウサイズ

ICON_UP_SPEED_SIZE_W   = 300            # speedアップアイコンサイズw
ICON_UP_SPEED_SIZE_H   = 150            # speedアップアイコンサイズh
LABEL_SPEED_SIZE_W     = 300            # speedラベルサイズw
LABEL_SPEED_SIZE_H     = 150            # speedラベルサイズh
ICON_DOWN_SPEED_SIZE_W = 300            # speedダウンアイコンサイズw
ICON_DOWN_SPEED_SIZE_H = 150            # speedダウンアイコンサイズh

BUTTON_BACK_SIZE = 130                  # 戻るボタンサイズ

# ================================================
# スタイル定義
# ================================================

# --- メインウインドウ ---
LABEL_CAM_STYLE = """
background-color: #333333; color: #FFFFFF;
font-size: 20px; font-weight: bold; border-radius: 5px;
qproperty-alignment: 'AlignCenter';
"""
LABEL_HISTORY_STYLE = """
font-family: "MS Gothic";
font-size: 20px; font-weight: bold;
color: #000000; background-color: #FFFFFF;
border: none;
qproperty-alignment: 'AlignCenter';
"""
LABEL_STATS_STYLE = """
font-family: "MS Gothic";
font-size: 15px; font-weight: bold;
color: #000000; background-color: #FFFFFF;
border: 2px solid #000000; border-radius: 5px;
qproperty-alignment: 'AlignCenter';
"""
LABEL_MANAGEMENT_STYLE = """
font-family: "Meiryo"; font-size: 40px; font-weight: bold;
color: #000000; background-color: #FFFFFF;
qproperty-alignment: 'AlignCenter';
"""
LABEL_TOGGLE_STYLE = """
font-family: "Meiryo"; font-size: 30px; font-weight: bold;
color: #888888; qproperty-alignment: 'AlignCenter';
"""
LABEL_CAM_NAME_STYLE = """
font-family: "Meiryo"; font-size: 14px; font-weight: bold;
color: #FFFFFF; background-color: #222222;
padding: 2px 4px; border-radius: 3px;
"""
LABEL_MODE_STYLE_PC = """
font-family: "Meiryo"; font-size: 22px; font-weight: bold;
color: #1E7A1E; qproperty-alignment: 'AlignLeft | AlignVCenter';
"""
LABEL_MODE_STYLE_STANDALONE = """
font-family: "Meiryo"; font-size: 22px; font-weight: bold;
color: #FF8C00; qproperty-alignment: 'AlignLeft | AlignVCenter';
"""

# --- 左下：モード表示パネル ---
LABEL_MODE_PANEL_STYLE = """
background-color: #F4F4F4; border: 2px solid #999999; border-radius: 8px;
"""
LABEL_MODE_TITLE_STYLE = """
font-family: "Meiryo"; font-size: 18px; font-weight: bold;
color: #555555; qproperty-alignment: 'AlignLeft | AlignVCenter';
"""
LABEL_MODE_CAPTION_STYLE = """
font-family: "Meiryo"; font-size: 20px; font-weight: bold;
color: #777777; qproperty-alignment: 'AlignLeft | AlignVCenter';
"""
LABEL_MODE_VALUE_STYLE = """
font-family: "Meiryo"; font-size: 22px; font-weight: bold;
color: #222222; qproperty-alignment: 'AlignLeft | AlignVCenter';
"""

# --- ステータスバー（run / stop / estop / error）---
#   実際の各セルは main 側で HTML 組み立てして強調表示する。
LABEL_STATUS_STYLE = """
background-color: #111111; border: 2px solid #555555; border-radius: 5px;
"""

CAM_NAME_LABEL_W = 120
CAM_NAME_LABEL_H = 22

# --- サブウインドウ ---
BUTTON_SUB_STYLE = """
font-family: "Meiryo"; font-size: 20px; font-weight: bold;
color: #00FFFF; background-color: #333333; border-radius: 50px;
"""
LABEL_SPEED_STYLE = """
font-family: "Meiryo"; font-size: 60px; font-weight: bold;
color: #000000; background-color: transparent;
qproperty-alignment: 'AlignCenter';
"""


# ================================================
# パスユーティリティ
# ================================================
def resource_path(relative_path: str) -> str:
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def resize_smooth_image(pixmap: QPixmap, button: QLabel) -> None:
    if not pixmap.isNull():
        scaled_pixmap = pixmap.scaled(
            button.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        button.setPixmap(scaled_pixmap)


# ================================================
# クリック可能なラベルクラス
# ================================================
class ClickableLabel(QLabel):
    clicked = Signal()      # クリックされたときに反応するシグナル

    # --- 左クリック時のみ送信する関数 ---
    def mousePressEvent(self, event) -> None:
        if self.isEnabled and event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    # --- ボタンをロック・アンロックする関数 ---
    def set_locked(self, locked: bool) -> None:
        """
        locked = True   :   クリック無効＆半透明
        locked = False  :   クリック有効＆通常表示
        """
        self.setEnabled(not locked)
        opacity_effect = QGraphicsOpacityEffect(self)
        opacity_effect.setOpacity(0.3 if locked else 1.0)
        self.setGraphicsEffect(opacity_effect)


# ================================================
# ToggleSwitch クラス
# ================================================
class ToggleSwitch(QCheckBox):
    def __init__(self, parent=None, width: int = 60, height: int = 30) -> None:
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.setCursor(Qt.PointingHandCursor)
        self._bg_color_off = QColor("#B0B0B0")
        self._bg_color_on  = QColor("#32CD32")
        self._circle_color = QColor("#FFFFFF")
        self._position     = 0.0
        self._animation    = QPropertyAnimation(self, b"position")
        self._animation.setDuration(200)
        self._animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.stateChanged.connect(self.setup_animation)

    @Property(float)
    def position(self) -> float:
        return self._position

    @position.setter
    def position(self, pos: float) -> None:
        self._position = pos
        self.update()

    def setup_animation(self, value: int) -> None:
        self._animation.stop()
        self._animation.setEndValue(1.0 if value else 0.0)
        self._animation.start()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width, height = self.width(), self.height()
        radius = height / 2

        # 背景色計算
        curr_col = QColor()
        r = self._bg_color_off.red()   + (self._bg_color_on.red()   - self._bg_color_off.red())   * self._position
        g = self._bg_color_off.green() + (self._bg_color_on.green() - self._bg_color_off.green()) * self._position
        b = self._bg_color_off.blue()  + (self._bg_color_on.blue()  - self._bg_color_off.blue())  * self._position
        curr_col.setRgb(int(r), int(g), int(b))

        painter.setBrush(QBrush(curr_col))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, width, height, radius, radius)

        # 丸の描画
        circle_r = radius - 3
        circle_x = radius + (width - 2 * radius) * self._position
        painter.setBrush(QBrush(self._circle_color))
        painter.drawEllipse(QPointF(circle_x, radius), circle_r, circle_r)

    def hitButton(self, pos) -> bool:
        return self.contentsRect().contains(pos)

    # --- スイッチをロック・アンロックする関数 ---
    def set_locked(self, locked: bool) -> None:
        """
        locked = True   :   クリック無効＆半透明
        locked = False  :   クリック有効＆通常表示
        """
        self.setEnabled(not locked)
        self.setCursor(Qt.ForbiddenCursor if locked else Qt.PointingHandCursor)
        opacity_effect = QGraphicsOpacityEffect(self)
        opacity_effect.setOpacity(0.3 if locked else 1.0)
        self.setGraphicsEffect(opacity_effect)


# ================================================
# スタートアップウインドウUI
# ================================================
class StartupWindowUI(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("起動確認")
        self.setFixedSize(500, 300)
        self.setStyleSheet("background-color: #CCCCCC;")

        self.label_msg = QLabel("システムを起動しますか？", self)
        self.label_msg.setFixedSize(400, 50)
        self.label_msg.move(50, 50)
        self.label_msg.setStyleSheet("color: black; font-size: 24px; font-weight: bold;")

        self.button_start = QPushButton("システム起動", self)
        self.button_start.setFixedSize(260, 80)
        self.button_start.move(120, 150)
        self.button_start.setStyleSheet(
            "background-color: #FF4500; color: white; font-size: 24px; border-radius: 40px;"
        )


# ================================================
# サブウインドウUI
# ================================================
class SubWindowUI(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("設定画面")
        self.setFixedSize(SUB_WINDOW_W, SUB_WINDOW_H)
        self.setStyleSheet("background-color: #EEEEEE;")
        self.setWindowModality(Qt.ApplicationModal)

        # --- speed変更エリア ---
        up_speed_x    = MARGIN_X * 5
        up_speed_y    = MARGIN_Y * 3
        speed_label_x = MARGIN_X * 5
        speed_label_y = ICON_UP_SPEED_SIZE_H + MARGIN_Y * 3
        down_speed_x  = MARGIN_X * 5
        down_speed_y  = SUB_WINDOW_H - ICON_DOWN_SPEED_SIZE_H - MARGIN_Y * 3

        pixmap = QPixmap(resource_path("Icon/up_speed.png"))
        self.button_up_speed = ClickableLabel(self)
        self.button_up_speed.setFixedSize(ICON_UP_SPEED_SIZE_W, ICON_UP_SPEED_SIZE_H)
        self.button_up_speed.move(up_speed_x, up_speed_y)
        self.button_up_speed.setCursor(Qt.PointingHandCursor)
        resize_smooth_image(pixmap, self.button_up_speed)

        self.label_current_speed = QLabel("5", self)  # 初期値5
        self.label_current_speed.setFixedSize(LABEL_SPEED_SIZE_W, LABEL_SPEED_SIZE_H)
        self.label_current_speed.setStyleSheet(LABEL_SPEED_STYLE)
        self.label_current_speed.move(speed_label_x, speed_label_y)

        pixmap = QPixmap(resource_path("Icon/down_speed.png"))
        self.button_down_speed = ClickableLabel(self)
        self.button_down_speed.setFixedSize(ICON_DOWN_SPEED_SIZE_W, ICON_DOWN_SPEED_SIZE_H)
        self.button_down_speed.move(down_speed_x, down_speed_y)
        self.button_down_speed.setCursor(Qt.PointingHandCursor)
        resize_smooth_image(pixmap, self.button_down_speed)

        # --- 戻るボタンエリア ---
        back_x = SUB_WINDOW_W - BUTTON_BACK_SIZE - MARGIN_X * 3
        back_y = SUB_WINDOW_H - BUTTON_BACK_SIZE - MARGIN_Y * 3

        self.button_back = QPushButton("戻る", self)
        self.button_back.setFixedSize(BUTTON_BACK_SIZE, BUTTON_BACK_SIZE)
        self.button_back.setStyleSheet(BUTTON_SUB_STYLE)
        self.button_back.move(back_x, back_y)
        self.button_back.setCursor(Qt.PointingHandCursor)


# ================================================
# カメラエラーウィンドウUI
# ================================================
class CameraErrorWindowUI(QWidget):
    def __init__(self, error_cams_text: str) -> None:
        super().__init__()
        self.setWindowTitle("カメラ接続エラー")
        self.setFixedSize(500, 350)
        self.setStyleSheet("background-color: #2B2B2B;")
        self.setWindowModality(Qt.ApplicationModal)

        layout = QVBoxLayout()

        self.label_title = QLabel("error", self)
        self.label_title.setStyleSheet("color: #FF0000; font-size: 28px; font-weight: bold;")
        self.label_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_title)

        self.label_msg = QLabel("以下のカメラとの接続が切れました:", self)
        self.label_msg.setStyleSheet("color: white; font-size: 16px;")
        self.label_msg.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_msg)

        self.label_cams = QLabel(error_cams_text, self)
        self.label_cams.setStyleSheet(
            "color: #FFA500; font-size: 20px; font-weight: bold; border: 1px solid #555; padding: 10px;"
        )
        self.label_cams.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_cams)

        self.label_instruction = QLabel("ケーブルを確認し、続行ボタンを押してください。", self)
        self.label_instruction.setStyleSheet("color: #AAA; font-size: 14px;")
        self.label_instruction.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_instruction)

        self.button_continue = QPushButton("続行", self)
        self.button_continue.setFixedSize(200, 60)
        self.button_continue.setStyleSheet(
            "background-color: #0078D7; color: white; font-size: 20px; font-weight: bold; border-radius: 10px;"
        )
        self.button_continue.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.button_continue, alignment=Qt.AlignCenter)

        self.setLayout(layout)


# ================================================
# メインウインドウUI
# ================================================
class MainWindowUI(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("サクランボ病害虫除去システム")
        self.setFixedSize(WINDOW_W, WINDOW_H)
        self.setStyleSheet("background-color: #FFFFFF;")

        cam_x_left, cam_x_right, cam_y_upper, cam_y_lower = self._build_camera_views()
        self._build_camera_name_overlays(cam_x_left, cam_x_right, cam_y_upper, cam_y_lower)
        self._build_power_button()

        history_y       = self._build_history_label()
        status_y         = self._build_status_bar(history_y)
        stats_x, stats_y = self._build_stats_label(status_y)
        self._build_management_panel(stats_x, stats_y)
        self._build_bottom_left_panel(cam_x_right, cam_y_lower)

        # モードパネル系ウィジェットより前面に出す
        self.button_setting.raise_()

        # 単体モード用ブロッキングオーバーレイ（電源ボタンより後に生成することで Z-order を制御）
        self.blocking_overlay = BlockingOverlay(self)
        self.blocking_overlay.setGeometry(0, 0, WINDOW_W, WINDOW_H)
        self.blocking_overlay.hide()

    # --- カメラ表示エリア ---
    def _build_camera_views(self) -> tuple[int, int, int, int]:
        cam_x_left  = MARGIN_X
        cam_x_right = VIEW_CAM_SIZE_W + MARGIN_X + MARGIN_CAM
        cam_y_upper = MARGIN_Y
        cam_y_lower = VIEW_CAM_SIZE_H + MARGIN_Y + MARGIN_CAM

        self.cam_in = QLabel("cam_inside", self)
        self.cam_in.setFixedSize(VIEW_CAM_SIZE_W, VIEW_CAM_SIZE_H)
        self.cam_in.setStyleSheet(LABEL_CAM_STYLE)
        self.cam_in.move(cam_x_left, cam_y_upper)

        self.cam_out = QLabel("cam_outside", self)
        self.cam_out.setFixedSize(VIEW_CAM_SIZE_W, VIEW_CAM_SIZE_H)
        self.cam_out.setStyleSheet(LABEL_CAM_STYLE)
        self.cam_out.move(cam_x_right, cam_y_upper)

        self.cam_under = QLabel("cam_under", self)
        self.cam_under.setFixedSize(VIEW_CAM_SIZE_W, VIEW_CAM_SIZE_H)
        self.cam_under.setStyleSheet(LABEL_CAM_STYLE)
        self.cam_under.move(cam_x_left, cam_y_lower)

        self.cam_top = QLabel("cam_top", self)
        self.cam_top.setFixedSize(VIEW_CAM_SIZE_W, VIEW_CAM_SIZE_H)
        self.cam_top.setStyleSheet(LABEL_CAM_STYLE)
        self.cam_top.move(cam_x_right, cam_y_lower)

        return cam_x_left, cam_x_right, cam_y_upper, cam_y_lower

    # --- カメラ名オーバーレイラベル（各カメラ映像の左上に重ねて表示）---
    def _build_camera_name_overlays(self, cam_x_left, cam_x_right, cam_y_upper, cam_y_lower) -> None:
        _cn_offset = 4
        self.cam_name_in = QLabel("cam_inside", self)
        self.cam_name_in.setFixedSize(CAM_NAME_LABEL_W, CAM_NAME_LABEL_H)
        self.cam_name_in.setStyleSheet(LABEL_CAM_NAME_STYLE)
        self.cam_name_in.move(cam_x_left + _cn_offset, cam_y_upper + _cn_offset)

        self.cam_name_out = QLabel("cam_outside", self)
        self.cam_name_out.setFixedSize(CAM_NAME_LABEL_W, CAM_NAME_LABEL_H)
        self.cam_name_out.setStyleSheet(LABEL_CAM_NAME_STYLE)
        self.cam_name_out.move(cam_x_right + _cn_offset, cam_y_upper + _cn_offset)

        self.cam_name_under = QLabel("cam_under", self)
        self.cam_name_under.setFixedSize(CAM_NAME_LABEL_W, CAM_NAME_LABEL_H)
        self.cam_name_under.setStyleSheet(LABEL_CAM_NAME_STYLE)
        self.cam_name_under.move(cam_x_left + _cn_offset, cam_y_lower + _cn_offset)

        self.cam_name_top = QLabel("cam_top", self)
        self.cam_name_top.setFixedSize(CAM_NAME_LABEL_W, CAM_NAME_LABEL_H)
        self.cam_name_top.setStyleSheet(LABEL_CAM_NAME_STYLE)
        self.cam_name_top.move(cam_x_right + _cn_offset, cam_y_lower + _cn_offset)

    # --- 電源アイコン（右上・従来通り）---
    def _build_power_button(self) -> None:
        power_x = WINDOW_W - ICON_POWER_SIZE - MARGIN_X
        power_y = MARGIN_Y

        pixmap = QPixmap(resource_path("Icon/power_supply.png"))
        self.button_power = ClickableLabel(self)
        self.button_power.setFixedSize(ICON_POWER_SIZE, ICON_POWER_SIZE)
        self.button_power.move(power_x, power_y)
        self.button_power.setCursor(Qt.PointingHandCursor)
        resize_smooth_image(pixmap, self.button_power)

    # --- クラス表示リスト（横長・右カラム上部）---
    def _build_history_label(self) -> int:
        history_x = RIGHT_X
        history_y = MARGIN_Y + ICON_POWER_SIZE + MARGIN_Y

        self.label_history = QLabel("入力待機中...", self)
        self.label_history.setFixedSize(LABEL_HISTORY_SIZE_W, LABEL_HISTORY_SIZE_H)
        self.label_history.setStyleSheet(LABEL_HISTORY_STYLE)
        self.label_history.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.label_history.move(history_x, history_y)

        return history_y

    # --- ステータスバー（run / stop / estop / error）---
    def _build_status_bar(self, history_y: int) -> int:
        status_x = RIGHT_X
        status_y = history_y + LABEL_HISTORY_SIZE_H + MARGIN_Y

        self.label_status = QLabel("", self)
        self.label_status.setFixedSize(LABEL_STATUS_SIZE_W, LABEL_STATUS_SIZE_H)
        self.label_status.setStyleSheet(LABEL_STATUS_STYLE)
        self.label_status.move(status_x, status_y)

        return status_y

    # --- 累計カウント統計（右下・左半分）---
    def _build_stats_label(self, status_y: int) -> tuple[int, int]:
        stats_x = RIGHT_X
        stats_y = status_y + LABEL_STATUS_SIZE_H + MARGIN_Y

        self.label_stats = QLabel("入力待機中...", self)
        self.label_stats.setFixedSize(LABEL_STATS_SIZE_W, LABEL_STATS_SIZE_H)
        self.label_stats.setStyleSheet(LABEL_STATS_STYLE)
        self.label_stats.move(stats_x, stats_y)

        return stats_x, stats_y

    # --- システム管理エリア（統計の右隣・トグルは従来サイズ）---
    def _build_management_panel(self, stats_x: int, stats_y: int) -> None:
        toggle_w = int(SWITCH_TOGGLE_SIZE_W)
        toggle_h = int(SWITCH_TOGGLE_SIZE_H)
        panel_x  = stats_x + LABEL_STATS_SIZE_W + MARGIN_X
        panel_w  = RIGHT_X + RIGHT_W - panel_x

        manage_x = panel_x + (panel_w - LABEL_MANAGEMENT_SIZE_W) // 2
        manage_y = stats_y
        toggle_x = panel_x + (panel_w - toggle_w) // 2
        toggle_y = manage_y + LABEL_MANAGEMENT_SIZE_H + MARGIN_Y
        toggle_status_x = panel_x + (panel_w - LABEL_TOGGLE_SIZE_W) // 2
        toggle_status_y = toggle_y + toggle_h + MARGIN_Y

        self.label_panel = QLabel("システム管理", self)
        self.label_panel.setFixedSize(LABEL_MANAGEMENT_SIZE_W, LABEL_MANAGEMENT_SIZE_H)
        self.label_panel.setStyleSheet(LABEL_MANAGEMENT_STYLE)
        self.label_panel.move(manage_x, manage_y)

        self.toggle_switch = ToggleSwitch(self, toggle_w, toggle_h)
        self.toggle_switch.move(toggle_x, toggle_y)

        self.label_toggle_status = QLabel("停止中", self)
        self.label_toggle_status.setFixedSize(LABEL_TOGGLE_SIZE_W, int(LABEL_TOGGLE_SIZE_H))
        self.label_toggle_status.setStyleSheet(LABEL_TOGGLE_STYLE)
        self.label_toggle_status.move(toggle_status_x, toggle_status_y)

    # --- 左下：設定ボタン＋モード表示パネル ---
    def _build_bottom_left_panel(self, cam_x_right: int, cam_y_lower: int) -> None:
        lb_y      = cam_y_lower + VIEW_CAM_SIZE_H + MARGIN_Y
        setting_x = MARGIN_X + 600
        setting_y = lb_y + 65

        pixmap = QPixmap(resource_path("Icon/setting.png"))
        self.button_setting = ClickableLabel(self)
        self.button_setting.setFixedSize(ICON_SETTING_SIZE, ICON_SETTING_SIZE)
        self.button_setting.move(setting_x, setting_y)
        self.button_setting.setCursor(Qt.PointingHandCursor)
        resize_smooth_image(pixmap, self.button_setting)

        # モード表示パネル（背景）
        mode_x = MARGIN_X
        mode_y = lb_y
        mode_w = (cam_x_right + VIEW_CAM_SIZE_W) - mode_x
        mode_h = WINDOW_H - mode_y - MARGIN_Y

        self.label_mode_panel = QLabel("", self)
        self.label_mode_panel.setFixedSize(mode_w, mode_h)
        self.label_mode_panel.setStyleSheet(LABEL_MODE_PANEL_STYLE)
        self.label_mode_panel.move(mode_x, mode_y)

        # パネル内：タイトル＋3行（動作モード / パルス速度 / モデル）
        pad     = 18
        cap_x   = mode_x + pad
        val_x   = mode_x + pad + 180
        cap_w   = 170
        val_w   = mode_w - pad * 2 - 180
        row_h   = 56
        title_y = mode_y + 12
        row1_y  = title_y + 46
        row2_y  = row1_y + row_h
        row3_y  = row2_y + row_h

        self.label_mode_title = QLabel("モード表示", self)
        self.label_mode_title.setFixedSize(mode_w - pad * 2, 30)
        self.label_mode_title.setStyleSheet(LABEL_MODE_TITLE_STYLE)
        self.label_mode_title.move(cap_x, title_y)

        cap1 = QLabel("動作モード", self)
        cap1.setStyleSheet(LABEL_MODE_CAPTION_STYLE)
        cap1.setFixedSize(cap_w, row_h)
        cap1.move(cap_x, row1_y)
        cap2 = QLabel("パルス速度", self)
        cap2.setStyleSheet(LABEL_MODE_CAPTION_STYLE)
        cap2.setFixedSize(cap_w, row_h)
        cap2.move(cap_x, row2_y)
        cap3 = QLabel("モデル", self)
        cap3.setStyleSheet(LABEL_MODE_CAPTION_STYLE)
        cap3.setFixedSize(cap_w, row_h)
        cap3.move(cap_x, row3_y)

        self.label_mode = QLabel("PCモード", self)
        self.label_mode.setFixedSize(val_w, row_h)
        self.label_mode.setStyleSheet(LABEL_MODE_STYLE_PC)
        self.label_mode.move(val_x, row1_y)

        self.label_pulse_speed = QLabel("-", self)
        self.label_pulse_speed.setFixedSize(val_w, row_h)
        self.label_pulse_speed.setStyleSheet(LABEL_MODE_VALUE_STYLE)
        self.label_pulse_speed.move(val_x, row2_y)

        self.label_model = QLabel("-", self)
        self.label_model.setFixedSize(val_w, row_h)
        self.label_model.setStyleSheet(LABEL_MODE_VALUE_STYLE)
        self.label_model.move(val_x, row3_y)


# ================================================
# 単体モード用ブロッキングオーバーレイ
#   ウィンドウ全体を半透明グレーで覆い、マウスイベントを吸収する。
#   電源ボタンは呼び出し側が raise_() で前面に保持することで操作可能に残す。
# ================================================
class BlockingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # マウスイベントを素通りさせない（デフォルトは False だが明示）
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(80, 80, 80, 140))
        painter.end()

    def mousePressEvent(self, event) -> None:
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        event.accept()