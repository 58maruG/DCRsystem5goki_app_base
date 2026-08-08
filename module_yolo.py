from __future__ import annotations
import cv2
import numpy as np
import datetime
import os
import time
import json
import threading
from ultralytics import YOLO
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml
from ultralytics.trackers.byte_tracker import BYTETracker
from PySide6.QtCore import QObject, Signal

# HSVマスク生成・整形は standalone/hsv_calibration.py と共有する（実装は hsv_mask_utils.py）。
# 校正ツールで合わせた閾値がそのまま本番の挙動になる前提を、実装の共有で担保している。
from hsv_mask_utils import (
    mask_from_hsv, remove_stem, remove_reflection_sat,
)

import log_config
log = log_config.get_logger("yolo")

# get_target_info はフレームごとに呼ばれるため、初回ロード時のみJSONを読む
_logged_hsv_paths: set[str] = set()
_hsv_config_cache: dict[str, dict] = {}


def _load_hsv_config(cam_name: str) -> dict:
    """カメラ別HSV設定（json/hsv_config_{cam}.json）を読む。初回のみJSONを読み、以降はキャッシュを返す。
    ファイルが無い/壊れている場合はここで例外を送出する（hsv_calibration.py で作成済みである前提）。"""
    cached = _hsv_config_cache.get(cam_name)
    if cached is not None:
        return cached

    path = os.path.join("json", f"hsv_config_{cam_name}.json")
    with open(path, 'r', encoding='utf-8-sig') as f:
        cfg = json.load(f)
    if path not in _logged_hsv_paths:
        log.info("HSV設定を読み込みました: %s", path)
        _logged_hsv_paths.add(path)

    # 整形（果柄除去・虚像除去）を走らせる必要があるかを一度だけ判定してキャッシュする。
    #   remove_reflection_sat と同じ「範囲が全域なら無効」の規則に合わせる。
    #   どれも無効なら 2段目の切り出し自体を省略し、旧実装と同じ経路になる。
    s_active = not (cfg['reflect_sat_lo'] <= 0 and cfg['reflect_sat_hi'] >= 255)
    v_active = not (cfg['reflect_v_lo']   <= 0 and cfg['reflect_v_hi']   >= 255)
    cfg['_refine'] = bool(int(cfg['stem_open']) > 0 or s_active or v_active)

    _hsv_config_cache[cam_name] = cfg
    return cfg

# ================================================
# モデル・入力設定
# ================================================
CENTER_THRESHOLD_X = 100

# 帯外常時HSV監視（get_target_info）の縮小率。1.0なら無効＝旧実装と1ビットも変わらない。
# min_blob_area・stem_open はこの値に応じて get_target_info 内で自動的に再較正され、
# mx/my/area/stat は呼び出し元が期待するネイティブ座標系へ復元して返す。
#
# 0.5を採用（学習データセット7,465枚・cam_top/inside/outsideで実測・2026-08-02検証）:
#   処理時間 5.01〜5.25倍高速化 / go-no-go一致率 99.5〜99.9% / 座標誤差 p95で1〜2px・面積相対誤差2.2%未満
#   検出漏れ(0.07〜0.52%, カメラにより変動)は全件「画面端に接触/近接した果実」の境界ケースに限られる
#   （20件を個別診断し、全件が同一原因であることを確認済み）。
#   原因: mask_from_hsv の OPEN/CLOSE カーネルが固定5x5でscale連動していないため、
#   縮小画像では整形後のブロブ形状がネイティブと変わり、端接触判定が逆転することがある。
#   画面端は推論ゲート窓（中心窓）の対象外であり、後続フレームでの再検出が見込めるため許容。
#   詳細: docs/findings_20260722_jetson_verdict.md §6
#   旧検証（2026-08-01時点のコードでは0.5は検出漏れ2.9〜3.3%）より大幅に改善しているのは、
#   その後の「左右端接触のみ棄却」変更（上下端接触を許容するよう緩和）による効果とみられる。
HSV_DETECT_SCALE = 0.5

# ================================================
# 推論ゲート（中心窓）設定
# ================================================
# 1カメラ・1果実あたりに推論するフレーム数。ここを変えるだけで枚数を調整できる
#   （窓幅は下の搬送速度から自動計算されるため、他の値を触る必要はない）。
#   増やす → 判定に使える情報が増えるがGPU負荷も比例して増える。
INFER_FRAMES_PER_CAM = 3

def infer_window_px(cam_name: str) -> int:
    """推論する中心窓の半幅[px]を返す。ROI中心から ±この距離に果実の重心がある間だけ推論する。

    窓幅 = 速度 × 枚数 / 2 とすることで、窓の通過中にちょうど INFER_FRAMES_PER_CAM 枚
    （位相により +1 枚）のフレームが入る。実際の採用数は上限カウンタで N 枚に切り詰めるため、
    果実の大きさや速度が多少ぶれても枚数は変わらない。
    旧方式（マスクが左右2本の線を跨ぐ間）は果実が大きいほど枚数が増えるサイズ依存だったが、
    重心基準にすることでサイズに依存しなくなる。"""
    cfg = _load_hsv_config(cam_name)
    speed = float(cfg['fruit_speed_px'])
    return max(1, int(round(speed * INFER_FRAMES_PER_CAM / 2.0)))

#MODEL_PATH = "Trained_Models/v2_11s.pt"
#MODEL_PATH   = "Trained_Models/v4_11s.pt"
MODEL_PATH   = "Trained_Models/SusHi_sample.engine"

YOLO_IMG_SIZE = 640
CONF_THRESHOLD = 0.5

# ================================================
# 健全/障害 二値判定設定
# ================================================
# 健全と判定するために必要な、異なるカメラでの healthy 検出台数（同一カメラの複数フレームは1カウント）。
#   障害系クラス（unripeを含む）は1カメラ・1検出でも即座に障害と判定する（低いハードル）。
#   健全と判定するには複数カメラでの一致を要求し、見落とし（偽健全）を防ぐ（高いハードル）。
HEALTHY_CONFIRM_MIN_CAMS = 2

# ================================================
# セグメンテーション（個体区切り）設定
# ================================================
# 不在タイムアウト（秒）: いずれのカメラもこの時間サクランボを検出しなければ
#   「1個分が通過し終わった」とみなして確定する（出口ヒステリシス）。
#   大きすぎる → 近接した2個が1個に統合される / 小さすぎる → 1個が複数IDに分割される
EMPTY_TIMEOUT_SEC = 0.5
# 最小可視時間（秒）: 個体が確定対象として「本物」と認められるための最小の可視継続。
#   これ未満しか見えず、かつYOLO検出も無かった瞬間的なノイズ blip は破棄し、
#   幽霊ID・黒タイルの量産を防ぐ（入口ヒステリシス）。
#   ※ YOLO検出が1度でもあれば、可視時間に依らず本物として確定する。
MIN_VISIBLE_SEC = 0.12

# ================================================
# ファイル保存設定
# ================================================
SAVE_DIR_TRAINING = "training_images"
MIN_TRAINING_AREA = 25000   # 学習用画像として保存する最小検出面積（小さすぎるフレームを除外）
FPS               = 20.0

# ================================================
# 追跡・可視化設定
# ================================================
# ByteTrack に渡す前段検出の信頼度しきい値。
# 低めにしてトラッカーへ多くの情報を渡し、カルマン予測の精度を上げる。
# ラベル採用の最終判定は CONF_THRESHOLD で行う（2段階フィルタ）。
PREDICT_CONF = 0.1
CAM_NAMES    = ['cam_top', 'cam_under', 'cam_inside', 'cam_outside']
COLORS = {
    "birddamage":  (225, 105,  65),
    "healthy":     (255, 255, 255),
    "mold":        (128,   0, 128),
    "stemcrack":   (255,   0,   0),
    "twin":        (  0,   0, 255),
    "unripe":      (  0, 255, 255),
    "malformation":(  0,  69, 255),
    "crack":       (255, 191,   0),
    "wilt":        ( 42, 107, 142),
    "suturecrack": (170, 178,  32),
    "brownrot":    ( 45,  82, 160),
    "blacktwin":   ( 79,  79,  47),
    "kasure":      (131, 180, 212),
}


# ================================================
# 判定結果データクラス
# ================================================
class YoloResult:
    def __init__(self, obj_id: int, label_name: str,
                 confidence: float, cam_name: str) -> None:
        self.id           = obj_id
        self.label_name   = label_name
        self.confidence   = confidence
        self.cam_name     = cam_name

        # 健全/障害の二値判定結果（_resolve_quality が確定時に設定）。
        #   True=障害（除去） / False=健全（運搬） / None=未確定。
        #   仕分け判定は必ずこちらを見る。label_name の "healthy" 一致では判定しないこと
        #   （複数カメラでの健全確証が無い場合、label_name="healthy" でも is_damaged=True になりうる）。
        self.is_damaged: bool | None = None

        # 個体確定時に _finalize_object が付与するサイクル集計（cycle ログ用）。
        # 既定値を持たせ、未確定の中間結果でも属性参照で落ちないようにする。
        self.num_detections     = 0      # この個体の総検出数（全フレーム・全カメラ）
        self.conf_max           = None   # 確定クラスの信頼度 最大
        self.conf_min           = None   # 確定クラスの信頼度 最小
        self.conf_avg           = None   # 確定クラスの信頼度 平均
        self.infer_avg_ms       = None   # この個体の推論時間平均(ms)
        self.preproc_ms         = None   # 前処理（クロップ・リサイズ）時間平均(ms)
        self.postproc_ms        = None   # 後処理（アノテーション・parse）時間平均(ms)
        self.capture_latency_ms = None   # カメラフレーム取得時間の平均(ms)
        self.frame_dropped      = None   # この個体の通過中にドロップしたフレーム数（全カメラ合計）
        self.hsv_pass           = None   # YOLO検出が1度でもあったか（1=あり/0=なし）
        self.hsv_mask_ratio     = None   # HSVマスク面積比の平均（0〜1）
        self.yolo_no_det_flag   = None   # HSV通過・YOLO無検出フラグ（1=HSV有でYOLO未検出 / 0=正常検出）
        # --- 1個体あたりの処理コスト内訳（4カメラ・全フレームの「合計」）---
        #   上の infer_avg_ms / preproc_ms / postproc_ms は1枚あたりの平均。
        #   目標「検出→前処理→推論が4カメラ合計で1秒/個未満」を判定するには
        #   合計値と枚数が要るので別に持つ。集計区間は前個体の確定〜この個体の確定。
        self.capture_sum_ms     = None   # 撮影（RGB8→BGR変換+複製）の合計(ms)
        self.capture_frames     = None   # 撮影した枚数（4カメラ合計）
        self.hsv_sum_ms         = None   # HSVマスク検出の合計(ms)
        self.hsv_frames         = None   # HSVを実行した枚数（帯内外を問わず）
        self.visible_frames     = None   # HSVが果実を捉えた枚数
        self.preproc_sum_ms     = None   # crop/resize の合計(ms)
        self.infer_sum_ms       = None   # 推論の合計(ms)
        self.infer_count        = None   # 推論した枚数
        self.postproc_sum_ms    = None   # ByteTrack+描画 の合計(ms)
        self.total_per_fruit_ms = None   # 上記4段の総計(ms) ← 目標指標
        self.visible_dur_s      = None   # 可視継続時間(s)
        self.cycle_dur_s        = None   # 集計区間の長さ(s)
        # この個体で検出された各クラスの「最大信頼度」を信頼度降順で並べたリスト。
        #   要素は (label_name, conf_max) のタプル。GUIの複数クラス表示が読む。
        self.class_breakdown    = []
        # カメラ別の検出内訳。GUIのカメラ別列（inside/outside/top/under）が読む。
        #   {cam_name: {"top": (label, conf), "final_conf": conf|None}}。
        #   そのカメラで有効検出が無ければキー自体が無い（GUIで "-"）。
        self.per_cam_breakdown  = {}


# ================================================
# 画像保存・CSV出力クラス
# ================================================
class OutputLogger:
    def __init__(self, dcr=None) -> None:
        self.dcr = dcr

        # 学習用画像ディレクトリ（YOLOクラス名別・実行をまたいで蓄積）
        self.valid_cam_names = ('cam_top', 'cam_under', 'cam_inside', 'cam_outside')
        #os.makedirs(SAVE_DIR_TRAINING, exist_ok=True)  # 画像保存一時停止 ← 再開時はコメントを外す

    def write_csv(self, obj_id: int, detections: list, final_label: str) -> None:
        """1個体（1 ID）の検出履歴を多ラベル long 形式で新ロガーへ送る。
        検出されたクラスごとに1件を出力し、同一クラスの複数検出は
        信頼度の min/max/平均と件数 n に集約する（異なるクラスは分けて記入）。
        final_label は仕分けで採用された確定クラス名。一致する行の is_final を1にする。"""
        if self.dcr is None:
            return
        # クラスごとに信頼度を集約（"None"＝未検出は除外）
        by_class: dict[str, list[float]] = {}
        for d in detections:
            if d.label_name == "None":
                continue
            by_class.setdefault(d.label_name, []).append(d.confidence)
        if not by_class:
            return

        items = []
        for label, confs in by_class.items():
            n = len(confs)
            items.append({
                "class":          label,
                "conf_ave":       round(sum(confs) / n, 2),
                "conf_min":       round(min(confs), 2),
                "conf_max":       round(max(confs), 2),
                "num_detections": n,
                "final_flag":     1 if label == final_label else 0,
            })
        self.dcr.detections(obj_id, items)

    def write_training_image(self, cam_name: str, frame, label_name: str | None = None) -> None:
        """学習用の生フレームをYOLOクラス名別ディレクトリに保存する。
        保存先は training_images/<クラス名>/ で、ファイル名は「推論クラス名_撮影時刻_カメラ名」。
        推論できなかった場合（label_name が None / 空 / "None"）はクラス名を NoClass とする。"""
        # 画像保存一時停止 ← 再開時はコメントを外す
        #if cam_name in self.valid_cam_names:
        #    cls     = label_name if label_name and label_name != "None" else "NoClass"
        #    cls_dir = os.path.join(SAVE_DIR_TRAINING, cls)
        #    os.makedirs(cls_dir, exist_ok=True)
        #    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        #    filename  = f"{timestamp}_{cls}_{cam_name}.jpg"
        #    filepath  = os.path.join(cls_dir, filename)
        #    cv2.imwrite(filepath, frame)


# ================================================
# 画像処理ユーティリティクラス
# ================================================
class ImageProcessor:
    @staticmethod
    def get_target_info(frame, cam_name: str) -> dict | None:
        # ここはYOLO判定ではなく「サクランボが視野内にいるか（存在）」を追うためのHSV赤マスク
        # HSV設定は json/hsv_config_{cam_name}.json（hsv_calibration.py で生成）から読む。
        cfg = _load_hsv_config(cam_name)
        h, w = frame.shape[:2]

        # 帯外常時監視のコスト削減用（HSV_DETECT_SCALE<1.0）: HSV変換以降を縮小画像で行う。
        # mx/my/area/stat は呼び出し元がネイティブ座標系前提のため、関数の出口で inv 倍して戻す。
        scale = HSV_DETECT_SCALE
        if scale != 1.0:
            det_frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                                    interpolation=cv2.INTER_AREA)
        else:
            det_frame = frame
        dh, dw = det_frame.shape[:2]
        inv = 1.0 / scale

        # ── 1段目: 果実検出用HSVマスク（熟度分類器 base と同一の共通実装）──
        #   mask_from_hsv は 2帯 inRange → OPEN(5x5,x2) → CLOSE(5x5,x2)。
        #   果実が居ないフレームはここで弾き、後段の整形を一切走らせない。
        hsv  = cv2.cvtColor(det_frame, cv2.COLOR_BGR2HSV)
        mask = mask_from_hsv(hsv, cfg)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 1:
            return None
        idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        # min_blob_area はネイティブpx^2基準の設定値。縮小画像の画素数に合わせ面積比(scale^2)で換算する。
        min_area = int(cfg['min_blob_area'] * scale * scale)
        if stats[idx, cv2.CC_STAT_AREA] < min_area:
            return None

        if not cfg['_refine']:
            # 果柄除去・虚像除去とも無効 → 1段目の結果をそのまま返す（旧実装と同一）
            s = stats[idx]
            bx = int(round(s[0] * inv)); by = int(round(s[1] * inv))
            bw = int(round(s[2] * inv)); bh = int(round(s[3] * inv))
            # 左右端接触のみ棄却（上下端は許容）
            if (bx <= 5) or ((bx + bw) >= (w - 5)):
                return None
            area_native = int(round(s[4] * inv * inv))
            return {'mx': int(round(centroids[idx][0] * inv)),
                    'my': int(round(centroids[idx][1] * inv)),
                    'area': area_native,
                    'stat': np.array([bx, by, bw, bh, area_native]),
                    'labels': labels, 'max_index': idx}

        # ── 2段目: 候補ブロブの周辺だけを切り出して整形する ────────────
        #   果柄除去は半径11の開処理で、全画面にかけると2ms超かかる。候補の
        #   外接矩形＋余白に限れば面積比で 1/4 以下になり、結果は変わらない。
        #   余白は演算の到達距離（開処理=半径 / 閉処理=4）以上を取る。整形は
        #   縮小方向の処理なので、この範囲の外へブロブがはみ出すことはない。
        #   stem_open はネイティブpx基準の半径なので、縮小画像上ではscale倍して使う。
        stem_open_scaled = max(0, int(round(cfg['stem_open'] * scale)))
        pad = max(stem_open_scaled, 4) + 1
        bx, by = int(stats[idx, cv2.CC_STAT_LEFT]), int(stats[idx, cv2.CC_STAT_TOP])
        bw, bh = int(stats[idx, cv2.CC_STAT_WIDTH]), int(stats[idx, cv2.CC_STAT_HEIGHT])
        x0, y0 = max(0, bx - pad),       max(0, by - pad)
        x1, y1 = min(dw, bx + bw + pad), min(dh, by + bh + pad)

        sub = mask[y0:y1, x0:x1]
        # 虚像（アクリル反射）除去: 「淡い かつ 明るい」画素を落とす。果実は彩度が高いので残る
        sub = remove_reflection_sat(
            sub, hsv[y0:y1, x0:x1, 1], hsv[y0:y1, x0:x1, 2],
            cfg['reflect_sat_lo'], cfg['reflect_sat_hi'],
            cfg['reflect_v_lo'],   cfg['reflect_v_hi'])
        # 果柄除去: 開処理で半径より細い突起を切り離す
        #   ※ 熟度分類器は THUMB_SIZE=112 上の値(既定2)。本システムのROIは
        #     500〜640px なので、同等の効果には 5〜6倍の半径が要る。
        sub = remove_stem(sub, stem_open_scaled)

        n2, lab2, st2, ce2 = cv2.connectedComponentsWithStats(sub)
        if n2 <= 1:
            return None
        j  = int(np.argmax(st2[1:, cv2.CC_STAT_AREA])) + 1
        s2 = st2[j].copy()
        if s2[cv2.CC_STAT_AREA] < min_area:
            return None

        # 切り出し座標(縮小スケール)を全画面座標へ戻してからネイティブ座標系へ復元する
        # （帯ゲート・クロップは呼び出し元でネイティブ座標基準のため）
        left_native   = int(round((s2[cv2.CC_STAT_LEFT] + x0) * inv))
        top_native    = int(round((s2[cv2.CC_STAT_TOP]  + y0) * inv))
        width_native  = int(round(s2[cv2.CC_STAT_WIDTH]  * inv))
        height_native = int(round(s2[cv2.CC_STAT_HEIGHT] * inv))
        area_native   = int(round(s2[cv2.CC_STAT_AREA] * inv * inv))
        # 左右端接触のみ棄却（上下端は許容）
        if (left_native <= 5) or ((left_native + width_native) >= (w - 5)):
            return None

        # 帯ゲートは labels[:, x] == max_index で全画面を走査する想定の実装だが、
        # 現在の呼び出し元(_process_frame)は mx/my/area/stat しか参照していない。
        # 縮小時にネイティブへ拡大コピーする分のコストを払わないよう、
        # labels は縮小スケールのまま返す（scale=1.0の既定では従来と完全に同一）。
        labels_full = np.zeros((dh, dw), dtype=np.int32)
        labels_full[y0:y1, x0:x1] = (lab2 == j)

        return {'mx': int(round((ce2[j][0] + x0) * inv)),
                'my': int(round((ce2[j][1] + y0) * inv)),
                'area': area_native,
                'stat': np.array([left_native, top_native, width_native, height_native, area_native]),
                'labels': labels_full, 'max_index': 1}

    @staticmethod
    def draw_inference_band(frame, cam_name, native_width, color=(0, 200, 255), thickness=2):
        """推論する中心窓の左右2本の縦線を frame に描画する（GUI可視化用）。
        果実の重心がこの2本の内側にある間だけ推論する（旧方式の「2本を跨ぐ」ではない）。
        窓は native_width(=ROI幅) 上の絶対pxで判定するため、表示フレームが640へ
        リサイズ済みでも比率に直して正しい位置へ描く。
        color は frame の色空間に合わせる（RGB画像へ描くなら RGB 並びで渡す）。"""
        h, w = frame.shape[:2]
        half = infer_window_px(cam_name)
        frac = (half / native_width) if native_width else 0.0
        xl = min(max(int(round((0.5 - frac) * w)), 0), w - 1)
        xr = min(max(int(round((0.5 + frac) * w)), 0), w - 1)
        cv2.line(frame, (xl, 0), (xl, h - 1), color, thickness)
        cv2.line(frame, (xr, 0), (xr, h - 1), color, thickness)
        return frame


# ================================================
# 検出器シグナル（カメラ別ワーカースレッド → GUIスレッドへのマーシャリング用）
# ================================================
class DetectorSignals(QObject):
    # frame_ready: (cam_name, rgb_frame[np.ndarray]) 表示用のRGBフレーム（640×640・帯線描画済み）
    frame_ready = Signal(str, object)
    # final_ready: (finalized YoloResult) 個体確定結果。GUIスレッドで process_final_result が受ける
    final_ready = Signal(object)


# ================================================
# YOLO検出クラス
# ================================================
class YoloDetector:
    def __init__(self, model_path: str = MODEL_PATH, dcr=None, cameras=None) -> None:
        log.info("YOLOモデル %s をロード中...", model_path)
        self.dcr     = dcr       # データログ（cycle は main、detections は logger 経由）
        self.cameras = cameras   # list[CameraController]（サイクル統計取得用）
        self.model   = YOLO(model_path)
        dummy_img    = np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8)
        self.model.predict(dummy_img, verbose=False)
        self.logger  = OutputLogger(dcr=dcr)

        # カメラごとに独立した ByteTracker（カメラ間で追跡状態が混ざらないよう分離）
        # track_buffer を FPS に合わせてスケール（旧 frame_rate 引数相当の補正）
        _cfg_dict = YAML.load(check_yaml("bytetrack.yaml"))
        _cfg_dict["track_buffer"] = int(_cfg_dict["track_buffer"] * FPS / 30.0)
        tracker_cfg  = IterableSimpleNamespace(**_cfg_dict)
        self.trackers = {cam: BYTETracker(tracker_cfg) for cam in CAM_NAMES}

        self.EMPTY_TIMEOUT_SEC = EMPTY_TIMEOUT_SEC
        self.MIN_VISIBLE_SEC   = MIN_VISIBLE_SEC

        self.current_cherry_id = 1
        self.last_seen_time    = time.monotonic()  # いずれかのカメラが最後に検出した時刻（不在判定用）

        # カメラごとの直近フレーム（世代ガードで推論結果を捨てた際の表示フォールバック用）
        self.last_frame_per_cam = {c: None for c in CAM_NAMES}

        # 現在追跡中の1個体の状態（_reset_object で初期化）
        self._obj_generation = 0   # _reset_object のたびに増加。古い推論結果を破棄する目印
        self._reset_object()

        # ── 並列パイプライン（案A）───────────────────────────────────
        # カメラごとに1本のワーカースレッドが HSV→前処理→推論→後処理→描画 を実行する。
        #   _state_lock : 4カメラ共有の確定・追跡状態を保護（RLock。_finalize_object 内の
        #                 _reset_object 再入に対応）。この下で確定処理を原子的に保つ。
        #   _infer_lock : model.predict をプロセス内で直列化する単一箇所。単一モデルへの
        #                 同時 predict はスレッド安全でないため。将来ここを batch=4 の一括
        #                 predict に差し替えれば Jetson 向けのバッチ化に移行できる。
        self.signals      = DetectorSignals()
        self._state_lock  = threading.RLock()
        self._infer_lock  = threading.Lock()
        self._running     = False          # 運転中フラグ（set_running で切替）
        self._workers_run = False
        self._workers     = []             # list[threading.Thread]（カメラ別）

    def model_precision(self) -> str:
        """ロード済みモデルの実際の重み精度を返す（startup ログ用）。
        half を明示していなくても torch の dtype から判定する。取得不可なら空。"""
        try:
            import torch
            dt = next(self.model.model.parameters()).dtype
            return {torch.float16: "fp16", torch.float32: "fp32",
                    torch.bfloat16: "bf16"}.get(dt, str(dt).replace("torch.", ""))
        except Exception:
            return ""

    # ------------------------------------------------------
    # 個体状態のリセット／更新ヘルパー
    # ------------------------------------------------------
    def _reset_object(self) -> None:
        self._obj_generation   += 1       # 古い推論結果を無効化するための世代番号
        self.obj_active         = False   # 追跡中の個体があるか
        self.obj_first_seen     = None    # 現個体が最初に見えた時刻
        self.obj_last_seen      = None    # 現個体が最後に見えた時刻
        self.obj_has_detection  = False   # 現個体でYOLO検出があったか
        self.obj_detections     = []      # 現個体のYoloResult履歴（全カメラ）
        self.obj_cam_train      = {}      # 学習用:   {cam: {'min_dist','frame'}}（生フレーム）
        self.obj_infer_ms_sum   = 0.0     # 現個体の推論時間の合計(ms)
        self.obj_infer_count    = 0       # 現個体で推論したフレーム数
        # カメラ別の推論投入数（INFER_FRAMES_PER_CAM で打ち切るためのカウンタ）。
        #   obj_infer_count はワーカーの完了時に増えるため、投入判定には使えない。
        self.obj_infer_frames_cam = {}    # {cam_name: 投入済み枚数}
        self.obj_preproc_ms_sum  = 0.0    # 前処理時間の合計(ms)
        self.obj_preproc_count   = 0
        self.obj_postproc_ms_sum = 0.0    # 後処理時間の合計(ms)
        self.obj_postproc_count  = 0
        self.obj_hsv_area_sum    = 0.0    # HSVマスク面積比の合計
        self.obj_hsv_area_count  = 0
        # --- 処理コスト計測（帯ゲートの手前で全フレームに走る段）---
        self.obj_hsv_ms_sum      = 0.0    # HSVマスク検出(get_target_info)の合計(ms)
        self.obj_hsv_frames      = 0      # HSVを実行したフレーム数（全カメラ・帯内外を問わず）
        self.obj_visible_frames  = 0      # HSVが果実を捉えたフレーム数（全カメラ）
        # 撮影段は各カメラスレッドが通算で積んでいるので、区間の起点をここで控える
        self._cap_base           = self._capture_totals()
        self._cycle_started_at   = time.monotonic()

    def _capture_totals(self) -> dict:
        """全カメラの撮影段コスト通算値を合算する。cameras 未設定なら 0。"""
        tot = {'cap_ms': 0.0, 'cap_frames': 0}
        for cam in (self.cameras or ()):
            try:
                t = cam.get_capture_totals()
            except Exception:
                continue   # 1台の取得失敗で計測全体を落とさない
            tot['cap_ms']     += t['cap_ms']
            tot['cap_frames'] += t['cap_frames']
        return tot

    def _finalize_object(self) -> YoloResult | None:
        """現個体を確定（または破棄）する。確定したら YoloResult を返す。
        入口ヒステリシス: 可視時間が MIN_VISIBLE_SEC 未満かつYOLO検出も無い blip は破棄しIDを進めない。"""
        result = None
        if self.obj_first_seen is not None:
            visible_dur = self.obj_last_seen - self.obj_first_seen
            confirmed   = self.obj_has_detection or (visible_dur >= self.MIN_VISIBLE_SEC)
            if confirmed:
                if self.obj_detections:
                    best = self._resolve_quality(self.obj_detections)
                    if best:
                        # long形式CSV: このIDの全検出をクラス別に集約して書き出す。
                        self.logger.write_csv(best.id, self.obj_detections, best.label_name)
                        # cycle ログ用の集計を best に添える（main が process_final_result で読む）
                        self._attach_cycle_stats(best, visible_dur=visible_dur)
                        result = best
                        # 学習用画像: 各カメラを「そのカメラの最高信頼度クラス」フォルダに保存
                        for cam, d in self.obj_cam_train.items():
                            self.logger.write_training_image(cam, d['frame'], d['label'])
                    self.current_cherry_id += 1
                else:
                    # 本物だがYOLO未検出 → 学習用保存 + cycle ログに yolo_no_det_flag=1 で記録
                    for cam, d in self.obj_cam_train.items():
                        self.logger.write_training_image(cam, d['frame'], d['label'])
                    no_det = YoloResult(self.current_cherry_id, "None", 0.0, "")
                    self._attach_cycle_stats(no_det, yolo_no_det=1, visible_dur=visible_dur)
                    result = no_det
                    self.current_cherry_id += 1
            # confirmed でない（短すぎる blip）→ 破棄し、IDは進めない
        self._reset_object()
        return result

    def _attach_cycle_stats(self, best: YoloResult, yolo_no_det: int = 0,
                             visible_dur: float | None = None) -> None:
        """確定個体の集計（検出数・確定クラスの信頼度統計・推論時間平均）を best に付与する。
        main 側の process_final_result がこれを読んで dcr.cycle(...) に渡す。
        yolo_no_det=1 のときは HSV通過・YOLO無検出ケース。"""
        best.yolo_no_det_flag = yolo_no_det
        self._attach_cost_breakdown(best, visible_dur)
        confs = [d.confidence for d in self.obj_detections if d.label_name == best.label_name]
        best.num_detections = len([d for d in self.obj_detections if d.label_name != "None"])
        if confs:
            best.conf_max = round(max(confs), 3)
            best.conf_min = round(min(confs), 3)
            best.conf_avg = round(sum(confs) / len(confs), 3)

        # クラス別の最大信頼度を集計し、信頼度降順で best に添える（GUIの複数クラス表示用）。
        #   "None"（未検出）は除外。write_csv の by_class と同じ集計方針。
        by_class: dict[str, float] = {}
        for d in self.obj_detections:
            if d.label_name == "None":
                continue
            if d.label_name not in by_class or d.confidence > by_class[d.label_name]:
                by_class[d.label_name] = d.confidence
        breakdown = sorted(by_class.items(), key=lambda kv: kv[1], reverse=True)
        best.class_breakdown = breakdown

        # カメラ別の検出内訳を集計する（GUIのカメラ別列用）。
        #   各カメラについて {クラス: そのカメラでの最大信頼度} を作り、そこから
        #     top        … そのカメラの最高信頼度クラス (label, conf)
        #     final_conf … 確定クラス(best.label_name)をそのカメラが検出していれば信頼度、無ければ None
        #   を取り出す。GUIは final_conf があれば「◎確定クラス」を優先表示し（そのカメラの
        #   最高信頼度クラスでなくても）、無ければ top を、検出ゼロなら "-" を出す。
        final_label = best.label_name
        cam_class_max: dict[str, dict[str, float]] = {}
        for d in self.obj_detections:
            if d.label_name == "None":
                continue
            m = cam_class_max.setdefault(d.cam_name, {})
            if d.label_name not in m or d.confidence > m[d.label_name]:
                m[d.label_name] = d.confidence
        per_cam: dict[str, dict] = {}
        for cam, cmax in cam_class_max.items():
            top_label = max(cmax, key=cmax.get)
            per_cam[cam] = {
                "top":        (top_label, cmax[top_label]),
                "final_conf": cmax.get(final_label),
            }
        best.per_cam_breakdown = per_cam
        if self.obj_infer_count > 0:
            best.infer_avg_ms = round(self.obj_infer_ms_sum / self.obj_infer_count, 2)
        if self.obj_preproc_count > 0:
            best.preproc_ms = round(self.obj_preproc_ms_sum / self.obj_preproc_count, 2)
        if self.obj_postproc_count > 0:
            best.postproc_ms = round(self.obj_postproc_ms_sum / self.obj_postproc_count, 2)
        if self.obj_hsv_area_count > 0:
            best.hsv_mask_ratio = round(self.obj_hsv_area_sum / self.obj_hsv_area_count, 4)
        best.hsv_pass = 1 if self.obj_has_detection else 0
        # カメラのサイクル統計を集約（cameras が渡されている場合のみ）
        if self.cameras:
            total_dropped = 0
            latencies     = []
            for cam in self.cameras:
                stats = cam.get_cycle_stats()
                total_dropped += stats['frame_dropped']
                if stats['capture_latency_ms'] is not None:
                    latencies.append(stats['capture_latency_ms'])
            best.frame_dropped = total_dropped
            if latencies:
                best.capture_latency_ms = round(sum(latencies) / len(latencies), 2)

    def _attach_cost_breakdown(self, best: YoloResult, visible_dur: float | None) -> None:
        """1個体あたりの処理コスト内訳（4カメラ・全フレームの合計）を best に付与する。
        目標「検出→前処理→推論が4カメラ合計で1秒/個未満」を cycle CSV 1行で判定できるようにする。
        集計区間は前個体の確定〜この個体の確定（= cycle_dur_s）。
          撮影   … 各カメラスレッドの通算値の差分（RGB8→BGR変換＋複製）
          前処理 … HSVマスク検出（全フレーム）＋ crop/resize（帯内のみ）
          推論   … model.predict（帯内のみ）
          後処理 … ByteTrack＋描画（帯内のみ）"""
        cap = self._capture_totals()
        best.capture_sum_ms = round(cap['cap_ms']     - self._cap_base['cap_ms'], 2)
        best.capture_frames = cap['cap_frames'] - self._cap_base['cap_frames']

        best.hsv_sum_ms      = round(self.obj_hsv_ms_sum, 2)
        best.hsv_frames      = self.obj_hsv_frames
        best.visible_frames  = self.obj_visible_frames
        best.preproc_sum_ms  = round(self.obj_preproc_ms_sum, 2)
        best.infer_sum_ms    = round(self.obj_infer_ms_sum, 2)
        best.infer_count     = self.obj_infer_count
        best.postproc_sum_ms = round(self.obj_postproc_ms_sum, 2)
        best.total_per_fruit_ms = round(
            best.capture_sum_ms + best.hsv_sum_ms + best.preproc_sum_ms
            + best.infer_sum_ms + best.postproc_sum_ms, 2)

        if visible_dur is not None:
            best.visible_dur_s = round(visible_dur, 3)
        best.cycle_dur_s = round(time.monotonic() - self._cycle_started_at, 3)

    def _resolve_quality(self, detections: list) -> YoloResult | None:
        """
        全カメラの履歴から健全/障害の二値判定（is_damaged）を行い、判定の根拠となった
        検出（最も信頼度が高いもの）を返す。label_name には引き続き具体的なクラス名が入るが、
        仕分け（リレー制御・GUI集計）に使うのは is_damaged の方であり、label_name 単体
        （"healthy" かどうか）では判定しない。

        障害判定（低いハードル）: healthy 以外のクラス（unripe を含む）が1件でも
          検出されていれば、カメラ台数によらず即座に障害と判定する。表示・ログ用の
          ラベルは、障害系検出のうち最も信頼度が高いものを採用する。
        健全判定（高いハードル）: 障害系検出が皆無で、かつ HEALTHY_CONFIRM_MIN_CAMS 台以上の
          異なるカメラで healthy が検出された場合のみ健全と判定する。条件を満たさない
          場合は見落とし（偽健全）を避けるため、安全側で障害として扱う。
        """
        if not detections:
            return None

        damage_list = [d for d in detections if d.label_name != "healthy"]
        if damage_list:
            best = max(damage_list, key=lambda x: x.confidence)
            best.is_damaged = True
            return best

        # ここに到達するのは healthy のみが検出されているケース。
        healthy_list = detections
        best = max(healthy_list, key=lambda x: x.confidence)
        healthy_cams = len({d.cam_name for d in healthy_list})
        best.is_damaged = healthy_cams < HEALTHY_CONFIRM_MIN_CAMS
        return best

    # ------------------------------------------------------
    # 運転状態・ワーカー制御
    # ------------------------------------------------------
    def set_running(self, running: bool) -> None:
        """運転中フラグを切り替える。GUIのトグルから呼ばれる。
        bool 代入は CPython では原子的なので、ワーカー側の読み取りとロック不要。"""
        self._running = bool(running)

    def start_workers(self) -> None:
        """カメラごとに1本の処理ワーカースレッドを起動する（カメラ起動後に呼ぶ）。"""
        if self._workers_run:
            return
        self._workers_run = True
        cam_by_name = {c.name: c for c in (self.cameras or [])}
        self._workers = []
        for cam_name in CAM_NAMES:
            camera = cam_by_name.get(cam_name)
            if camera is None:
                continue  # 接続に失敗したカメラは対象外
            t = threading.Thread(target=self._camera_worker, args=(cam_name, camera),
                                 name=f"cam-worker-{cam_name}", daemon=True)
            t.start()
            self._workers.append(t)
        log.info("カメラ別処理ワーカーを %d 本起動しました", len(self._workers))

    def restart_workers(self) -> None:
        """カメラ再初期化（復旧）後にワーカーを起動し直して新しいカメラ obj へ束ね直す。
        self.cameras は CameraManager.controllers と同一リスト参照で、init_cameras が
        in-place で中身を入れ替えるため新カメラを指している。ワーカーは起動時に旧カメラ obj を
        引数で掴んでいるので、停止→再起動して束ね直す必要がある。"""
        self._workers_run = False
        for t in self._workers:
            t.join(timeout=2.0)
        self._workers = []
        self.start_workers()

    # ------------------------------------------------------
    # カメラ別ワーカー（HSV → 前処理 → 推論 → 後処理 → 描画）
    # ------------------------------------------------------
    def _camera_worker(self, cam_name: str, camera) -> None:
        """1カメラ分のパイプラインを回すワーカー。新フレームごとに処理し、表示フレームと
        確定結果をシグナルでGUIスレッドへ渡す。共有状態は _state_lock、推論は _infer_lock で保護。"""
        last_seq = -1
        while self._workers_run:
            frame, seq = camera.get_next_frame(last_seq, timeout=0.1)
            if not self._workers_run:
                break
            if frame is None or seq == last_seq:
                continue  # 新フレームなし（遅延キュー未充填 / タイムアウト）
            last_seq = seq
            try:
                if not self._running:
                    # 停止中: 生フレームをそのまま表示（推論・状態更新はしない）
                    disp = cv2.resize(frame, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))
                    self._emit_display(cam_name, disp, frame.shape[1])
                    continue
                now = time.monotonic()
                annotated, finalized = self._process_frame(cam_name, frame, now)
                self._emit_display(cam_name, annotated, frame.shape[1])
                if finalized is not None:
                    self.signals.final_ready.emit(finalized)
            except Exception as e:
                log.error("カメラワーカー例外(%s): %s", cam_name, e)
        log.debug("カメラワーカー終了: %s", cam_name)

    def _emit_display(self, cam_name: str, bgr_frame, native_width: int) -> None:
        """表示用フレームをRGB化し、推論ゲートの帯線を描いてGUIスレッドへ送る。
        cvtColor・帯線描画はここ（ワーカースレッド）で行い、GUIスレッドの負荷を下げる。"""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        if hasattr(ImageProcessor, "draw_inference_band"):
            ImageProcessor.draw_inference_band(rgb, cam_name, native_width, color=(0, 200, 255))
        self.signals.frame_ready.emit(cam_name, rgb)

    def _process_frame(self, cam_name: str, frame, now: float):
        """1フレームを処理し (表示用BGRフレーム[640], finalized_result|None) を返す。
        HSV・前処理・推論はロック外、共有状態の判定/更新は _state_lock、predict は _infer_lock。"""
        # ── HSV（ロック外・frame のみに依存）──
        _t_hsv = time.perf_counter()
        target = ImageProcessor.get_target_info(frame, cam_name)
        hsv_ms = (time.perf_counter() - _t_hsv) * 1000.0
        found  = target is not None

        finalized_result = None
        do_infer = False
        gen = obj_id = center_dist = 0

        # ── 共有状態: HSVコスト集計・出口ヒステリシス・帯判定・枠予約 ──
        with self._state_lock:
            self.obj_hsv_ms_sum += hsv_ms
            self.obj_hsv_frames += 1
            if found:
                self.obj_visible_frames += 1

            # 出口ヒステリシス（いずれのカメラも一定時間検出なし → 確定）
            if self.obj_active and (now - self.last_seen_time) >= self.EMPTY_TIMEOUT_SEC:
                finalized_result = self._finalize_object()

            # 帯判定（重心が中心窓内 かつ このカメラの推論枚数が上限未満）
            if found:
                half   = infer_window_px(cam_name)
                cx     = frame.shape[1] / 2.0
                in_win = abs(target['mx'] - cx) <= half
                if in_win and self.obj_infer_frames_cam.get(cam_name, 0) < INFER_FRAMES_PER_CAM:
                    # HSVマスク面積比の積算（帯内のみ）
                    fp = frame.shape[0] * frame.shape[1]
                    if fp > 0:
                        self.obj_hsv_area_sum   += target['area'] / fp
                        self.obj_hsv_area_count += 1
                    # 実際に推論するフレームだけ枠を消費する
                    self.obj_infer_frames_cam[cam_name] = self.obj_infer_frames_cam.get(cam_name, 0) + 1
                    gen    = self._obj_generation
                    obj_id = self.current_cherry_id
                    do_infer = True

        if not do_infer:
            # 帯外: 生フレームを表示用に整えるだけ（推論しない）
            raw = cv2.resize(frame, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))
            with self._state_lock:
                self._buffer_frame(cam_name, raw)
            return raw, finalized_result

        # ── 前処理（ロック外・リサイズ）──
        # center_dist は中心からの距離。CENTER_THRESHOLD_X以上離れている場合は
        # 「遠い」ことだけ分かればよいので frame.shape[1]//2（画面半幅）に丸める。
        _t_pre = time.perf_counter()
        center_dist = abs(target['mx'] - frame.shape[1] // 2)
        if center_dist >= CENTER_THRESHOLD_X:
            center_dist = frame.shape[1] // 2
        input_img_resized = cv2.resize(frame, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))
        preproc_ms = (time.perf_counter() - _t_pre) * 1000.0

        # ── 推論（_infer_lock で直列化。単一モデルへの同時 predict を防ぐ）──
        _t_inf = time.perf_counter()
        try:
            with self._infer_lock:
                results = self.model.predict(input_img_resized, conf=PREDICT_CONF, verbose=False)
            det = results[0].boxes.cpu().numpy()
        except Exception as e:
            log.error("推論例外(%s): %s", cam_name, e)
            with self._state_lock:
                self._buffer_frame(cam_name, input_img_resized)
            return input_img_resized, finalized_result
        infer_ms = (time.perf_counter() - _t_inf) * 1000.0

        # ── 後処理・状態更新（_state_lock）──
        with self._state_lock:
            # 世代ガード: 推論中に別カメラが個体確定(_reset)していたら残留結果を捨てる
            if gen != self._obj_generation:
                annotated = self.last_frame_per_cam.get(cam_name)
                if annotated is None:
                    annotated = input_img_resized
                self._buffer_frame(cam_name, annotated)
                return annotated, finalized_result

            self.obj_preproc_ms_sum += preproc_ms
            self.obj_preproc_count  += 1
            self.obj_infer_ms_sum   += infer_ms
            self.obj_infer_count    += 1
            annotated = self._apply_tracks(cam_name, det, input_img_resized,
                                           now, obj_id, target, center_dist)
        return annotated, finalized_result

    def _apply_tracks(self, cam_name: str, det, img, now_ctx: float,
                      obj_id: int, target: dict, center_dist: int):
        """ByteTracker適用・状態更新・描画を行い、描画済みBGRフレーム(640)を返す。
        呼び出し側が _state_lock を保持している前提。"""
        t_post = time.perf_counter()
        tracks = self.trackers[cam_name].update(det, img)

        annotated_frame = img.copy()
        best_result     = YoloResult(obj_id, "None", 0.0, cam_name)
        has_valid_track = False

        for row in tracks:
            x1, y1, x2, y2 = map(int, row[:4])
            conf  = float(row[5])
            cls   = int(row[6])
            label = self.model.names[cls].lower()
            if conf < CONF_THRESHOLD:
                continue
            has_valid_track = True
            color      = COLORS.get(label, (0, 255, 0))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            label_text = f"{label} {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            back_y1 = max(0, y1 - text_h - 10)
            cv2.rectangle(annotated_frame, (x1, back_y1), (x1 + text_w, y1), color, -1)
            cv2.putText(annotated_frame, label_text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            if conf > best_result.confidence:
                best_result = YoloResult(obj_id, label, conf, cam_name)

        if has_valid_track:
            self.last_seen_time = now_ctx
            self.obj_last_seen  = now_ctx
            if not self.obj_active:
                self.obj_active     = True
                self.obj_first_seen = now_ctx
                if self.cameras:
                    for cam in self.cameras:
                        cam.reset_cycle_stats()

        self.obj_postproc_ms_sum += (time.perf_counter() - t_post) * 1000.0
        self.obj_postproc_count  += 1

        # 学習用フレーム保持（帯内なので target は非None）
        if target is not None and target['area'] >= MIN_TRAINING_AREA:
            entry = self.obj_cam_train.get(cam_name)
            if entry is None:
                entry = {'min_dist': float('inf'), 'frame': None, 'label': None, 'conf': -1.0}
                self.obj_cam_train[cam_name] = entry
            if center_dist < entry['min_dist']:
                entry['min_dist'] = center_dist
                entry['frame']    = img.copy()
            if best_result.label_name != "None" and best_result.confidence > entry['conf']:
                entry['conf']  = best_result.confidence
                entry['label'] = best_result.label_name

        # obj_detections への蓄積
        if best_result.label_name != "None":
            self.obj_detections.append(best_result)
            self.obj_has_detection = True

        self._buffer_frame(cam_name, annotated_frame)
        return annotated_frame

    def _buffer_frame(self, cam_name: str, frame) -> None:
        self.last_frame_per_cam[cam_name] = frame

    def close(self) -> None:
        # カメラ別ワーカーを停止して join する
        self._workers_run = False
        for t in self._workers:
            t.join(timeout=2.0)
        self._workers = []

        # 終了時: 追跡中の個体が残っていれば確定して保存する
        with self._state_lock:
            if self.obj_active:
                self._finalize_object()