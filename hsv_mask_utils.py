"""HSVマスク生成・整形の共通処理。

本番の果実検出（module_yolo.ImageProcessor.get_target_info）と、
閾値調整ツール（standalone/hsv_calibration.py）が同一の実装を使うための共有モジュール。
両者が同じマスクを作ることが、校正ツールで合わせた閾値をそのまま本番へ持ち込める前提になっている。

依存は cv2 と numpy のみ。ultralytics/torch を読まないため、
校正ツールのような軽量プロセスからも起動コストなしで import できる。

【重要】同一実装が C:\\Users\\kotan\\gohara\\cherry_yolo\\ripeness_classifier\\common\\mask_utils.py
  にも存在する。元々は熟度分類器（学習データ整備用GUI）と本番でこのファイルを共有していたが、
  分類器を cherry_yolo へ分離した際に本番側の実装を本ファイルへ移した。
  閾値そのものはJSON側なので影響しないが、開閉処理の回数・カーネルサイズ等の
  アルゴリズムを変更する場合は必ず両方を揃えること。
"""

import cv2
import numpy as np


def mask_from_hsv(hsv: np.ndarray, params: dict) -> np.ndarray:
    """HSV画像から2帯（lower1-upper1 / lower2-upper2）のマスクを作り、開閉で整える。"""
    m1 = cv2.inRange(hsv, np.array(params["lower1"]), np.array(params["upper1"]))
    m2 = cv2.inRange(hsv, np.array(params["lower2"]), np.array(params["upper2"]))
    mask = cv2.bitwise_or(m1, m2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def remove_stem(mask: np.ndarray, radius: int) -> np.ndarray:
    """開処理（erode→dilate）で radius より細い突起（果柄）を切り離して除去する。
    太い本体は開処理でサイズがほぼ保たれる（大きい物体は復元される）ため、
    結果の輪郭は果実本体にほぼ一致する。radius<=0 なら何もしない。"""
    if radius <= 0:
        return mask
    ksize = 2 * radius + 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)


def remove_reflection_sat(mask: np.ndarray, s_channel: np.ndarray, v_channel: np.ndarray,
                          s_lo: int = 0, s_hi: int = 255,
                          v_lo: int = 0, v_hi: int = 255) -> np.ndarray:
    """アクリル板の虚像（反射）を彩度・明度の範囲指定（HSVマスクと同じ左右ハンドル方式）で除去する。
      s_lo/s_hi : 彩度Sの範囲（0〜255）。S∈[s_lo, s_hi] の画素を虚像候補とする。
                  範囲が全域(0〜255)なら無効。
      v_lo/v_hi : 明度Vの範囲（0〜255・試験導入）。V∈[v_lo, v_hi] の画素を虚像候補とする。
                  範囲が全域(0〜255)なら無効。
    考え方: 実体（果実）は彩度が高く、虚像（白飛び）は明らかに淡く（低彩度）かつ明るい（高明度）。
      典型的には彩度は低い側（例: [0, 100]）、明度は高い側（例: [200, 255]）を指定して使う。
      両方が有効なときは両条件を同時に満たす画素だけを確実な虚像として除去する
      （どちらか一方だけでは実体の暗部・淡色部を誤って削る恐れがあるため、両条件のANDで厳しく絞る）。
      片方だけ有効な場合はその条件単独で判定し、両方無効なら何もしない。
      彩度で淡い虚像・アクリルの白飛びを落とすと、実体と虚像の“接地境界”も低彩度なので
      自然に分離される。縦並び双子果は房の境界も彩度が高いまま残るので分断されない
      （幾何くびれ方式と違い衝突しない）。実体内部の光沢テカリで開く穴は CLOSE で埋め戻す。"""
    s_active = not (s_lo <= 0 and s_hi >= 255)
    v_active = not (v_lo <= 0 and v_hi >= 255)
    if not s_active and not v_active:
        return mask

    s_in_range = (s_channel >= s_lo) & (s_channel <= s_hi)
    v_in_range = (v_channel >= v_lo) & (v_channel <= v_hi)
    if s_active and v_active:
        is_reflection = s_in_range & v_in_range
    elif s_active:
        is_reflection = s_in_range
    else:
        is_reflection = v_in_range

    # 虚像候補ではない画素だけ残す
    vivid = cv2.bitwise_and(mask, np.where(~is_reflection, 255, 0).astype(np.uint8))
    # 実体内部の光沢テカリで開いた穴を閉じる（外形はほぼ保たれる）
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    vivid = cv2.morphologyEx(vivid, cv2.MORPH_CLOSE, k, iterations=2)
    return vivid
