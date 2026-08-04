"""Task 3: 半解像度化による検出劣化を検証する（動画ファイル / 実機ライブ）。

【なぜやるのか】
処理時間だけ改善しても、HSVゲートが果実を取りこぼせばシステムとして後退する。
面積が1/4になることで以下のリスクがある。

  - 画角端に出現した果実の検出タイミング遅延（→ エジェクタの射出タイミングずれ）
  - 小径果・部分遮蔽果の検出漏れ
  - mask_from_hsv の OPEN/CLOSE カーネルが固定5x5で scale 連動していないため、
    縮小画像では整形後のブロブ形状がネイティブと変わる（module_yolo.py:66 の既知事項）

【フレーム一致率より個体単位の遅延が重要】
実害に直結するのは「個体の検出が何フレーム遅れるか」。本スクリプトは
  (a) フレーム単位の一致率・取りこぼし・誤検出
  (b) 個体（連続検出区間）単位の検出開始フレーム遅延
  (c) 中心窓（推論ゲート）への進入フレーム遅延  ← エジェクタのタイミングに直結
の3つを出す。(c) はプランに無いが、実際に排出タイミングを決めるのは
検出開始ではなく中心窓への進入なので追加した。

【合格基準】プラン §Task 3
  検出取りこぼし率      < 1%（100個体で1個体未満）
  検出開始フレーム遅延  中央値 ≤ 1フレーム / P95 ≤ 2フレーム

【使い方】
    # 動画ファイル（圃場試験動画）
    uv run python standalone/verify_hsv_scale_video.py --video data/run01.mp4 --cam cam_top

    # 複数動画をまとめて（カメラ名は動画ごとに指定）
    uv run python standalone/verify_hsv_scale_video.py \
        --video top.mp4:cam_top --video under.mp4:cam_under

    # 実機ライブ（4カメラを60秒。本番アプリは停止しておくこと）
    uv run python standalone/verify_hsv_scale_video.py --live --duration 60

    # 面積閾値の再チューニング（プラン §3-4）
    uv run python standalone/verify_hsv_scale_video.py --video run01.mp4 --cam cam_top \\
        --sweep-area 0.6,0.8,1.0,1.2,1.5

【注意】
  - --live は Basler カメラを排他的に開く。本番アプリ・hsv_calibration が起動中だと失敗する。
  - 動画はカメラのネイティブ解像度へ自動リサイズされる（違う解像度のまま測ると結果が無意味）。
  - 本番は既に HSV_DETECT_SCALE=0.5 を採用済み（module_yolo.py:73）。本スクリプトは
    その判断を別の観点（動画の時系列・実機ライブ）から追試するもの。
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jetson_probe_common as common  # noqa: E402


# ==========================================================
# 1フレームの評価
# ==========================================================
def evaluate_frame(frame: np.ndarray, cam: str, scale: float) -> dict:
    """本番の get_target_info を指定の縮小率で実行し、判定に使う値だけ取り出す。"""
    with common.detect_scale(scale):
        res = common.detect(frame, cam)
    if res is None:
        return {"detected": False, "mx": None, "area": 0, "in_window": False}
    w = frame.shape[1]
    return {
        "detected": True,
        "mx": int(res["mx"]),
        "my": int(res["my"]),
        "area": int(res["area"]),
        "in_window": common.in_infer_window(int(res["mx"]), w, cam),
    }


# ==========================================================
# 個体（連続検出区間）単位の集計
# ==========================================================
def find_episodes(flags: list, gap_tolerance: int = 2) -> list:
    """True の連続区間を「個体」として切り出す。

    gap_tolerance フレームまでの途切れは同一個体の一時的な失検出とみなして繋ぐ
    （本番も EMPTY_TIMEOUT_SEC=0.5秒 のヒステリシスで同様に扱う）。
    戻り値: [(start_idx, end_idx), ...]  end は区間に含む。"""
    episodes = []
    start = None
    gap = 0
    for i, f in enumerate(flags):
        if f:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > gap_tolerance:
                episodes.append((start, i - gap))
                start = None
                gap = 0
    if start is not None:
        episodes.append((start, len(flags) - 1 - gap))
    return episodes


def episode_delays(ref_flags: list, test_flags: list, gap_tolerance: int) -> dict:
    """基準（scale=1.0）の個体区間ごとに、比較側が何フレーム遅れて検出し始めたかを集計する。

    - 遅延が負（比較側の方が早い）こともある。そのまま記録する
    - 区間中に一度も検出しなければ「取りこぼし」として数える"""
    episodes = find_episodes(ref_flags, gap_tolerance)
    delays, misses, details = [], 0, []

    for (s, e) in episodes:
        first_test = None
        for i in range(s, e + 1):
            if test_flags[i]:
                first_test = i
                break
        if first_test is None:
            misses += 1
            details.append({"start": s, "end": e, "delay": None, "missed": True})
        else:
            d = first_test - s
            delays.append(d)
            details.append({"start": s, "end": e, "delay": d, "missed": False})

    out = {
        "episodes": len(episodes),
        "missed": misses,
        "miss_rate": round(misses / len(episodes), 4) if episodes else 0.0,
        "detail": details,
    }
    if delays:
        arr = np.array(delays)
        out.update({
            "delay_mean": round(float(arr.mean()), 3),
            "delay_median": float(statistics.median(delays)),
            "delay_p95": float(np.percentile(arr, 95)),
            "delay_max": int(arr.max()),
            "delay_min": int(arr.min()),
            "delay_ge2_count": int((arr >= 2).sum()),
            "delay_ge2_rate": round(float((arr >= 2).mean()), 4),
        })
    return out


def summarize(rows: list, gap_tolerance: int) -> dict:
    """フレーム単位と個体単位の両方をまとめ、合格基準に照らす。"""
    if not rows:
        return {"error": "フレームが1枚もありません"}

    ref_det = [r["ref_detected"] for r in rows]
    test_det = [r["test_detected"] for r in rows]
    ref_win = [r["ref_in_window"] for r in rows]
    test_win = [r["test_in_window"] for r in rows]
    n = len(rows)

    agree = sum(1 for a, b in zip(ref_det, test_det) if a == b)
    miss = sum(1 for a, b in zip(ref_det, test_det) if a and not b)
    false = sum(1 for a, b in zip(ref_det, test_det) if (not a) and b)

    out = {
        "frames": n,
        "frame_agreement": round(agree / n, 4),
        "frame_miss": miss,           # 基準で検出・比較側で非検出
        "frame_false": false,         # 基準で非検出・比較側で検出
        "frame_miss_rate": round(miss / n, 4),
        "detect_episodes": episode_delays(ref_det, test_det, gap_tolerance),
        "window_episodes": episode_delays(ref_win, test_win, gap_tolerance),
        # 逆方向: 比較側にだけ現れた個体＝幽霊個体。実害は「存在しない果実への排出動作」。
        #   ref/test を入れ替えて同じ集計を掛けると、missed がそのまま幽霊個体数になる。
        "phantom_episodes": episode_delays(test_det, ref_det, gap_tolerance),
    }

    # 座標・面積の誤差（両方が検出できたフレームのみ）
    both = [r for r in rows if r["ref_detected"] and r["test_detected"]]
    if both:
        dmx = np.array([abs(r["test_mx"] - r["ref_mx"]) for r in both], dtype=float)
        # 面積は比で見る（絶対値はカメラ・果実サイズで桁が変わるため）
        rel = np.array([abs(r["test_area"] - r["ref_area"]) / max(1, r["ref_area"])
                        for r in both], dtype=float)
        out["mx_err_px"] = {"p50": round(float(np.percentile(dmx, 50)), 2),
                            "p95": round(float(np.percentile(dmx, 95)), 2),
                            "max": round(float(dmx.max()), 2)}
        out["area_rel_err"] = {"p50": round(float(np.percentile(rel, 50)), 4),
                               "p95": round(float(np.percentile(rel, 95)), 4),
                               "max": round(float(rel.max()), 4)}

    # --- 合格基準の判定 ---
    ep = out["detect_episodes"]

    # 基準側が1個体も検出していない入力で「取りこぼし0件＝合格」と出すのは誤り。
    # 果実の写っていない動画・閾値の設定ミス・カメラ違いのいずれかなので判定を止める。
    if ep["episodes"] == 0:
        out["checks"] = []
        out["verdict"] = "判定不能"
        out["verdict_reason"] = (
            "基準（scale=1.0）側の検出が0個体。果実が写っていない入力か、"
            "カメラ名の取り違え（HSV設定・解像度が別カメラのもの）の可能性が高い。"
            f"比較側は {sum(test_det)} フレームで検出している。")
        return out

    checks = []
    checks.append({
        "name": "検出取りこぼし率 < 1%",
        "value": f"{ep['miss_rate']:.2%}（{ep['missed']}/{ep['episodes']}個体）",
        "pass": ep["miss_rate"] < 0.01,
    })
    # プランの合格基準には無いが、幽霊個体は「存在しない果実への排出」を招くため同格で扱う
    ph = out["phantom_episodes"]
    checks.append({
        "name": "幽霊個体（比較側にだけ現れた個体）が全体の1%未満",
        "value": f"{ph['missed']}件 / 比較側 {ph['episodes']}個体",
        "pass": (ph["missed"] / ph["episodes"] < 0.01) if ph["episodes"] else True,
    })
    if "delay_median" in ep:
        checks.append({"name": "検出開始フレーム遅延 中央値 ≤ 1",
                       "value": f"{ep['delay_median']}",
                       "pass": ep["delay_median"] <= 1})
        checks.append({"name": "検出開始フレーム遅延 P95 ≤ 2",
                       "value": f"{ep['delay_p95']}",
                       "pass": ep["delay_p95"] <= 2})
        checks.append({"name": "遅延2フレーム以上が全体の5%以下（プラン §3-3）",
                       "value": f"{ep['delay_ge2_rate']:.2%}",
                       "pass": ep["delay_ge2_rate"] <= 0.05})
    wep = out["window_episodes"]
    if "delay_median" in wep:
        checks.append({"name": "中心窓への進入遅延 中央値 ≤ 1（エジェクタ影響）",
                       "value": f"{wep['delay_median']}",
                       "pass": wep["delay_median"] <= 1})

    out["checks"] = checks
    out["verdict"] = "合格" if all(c["pass"] for c in checks) else "不合格"
    if ep["episodes"] < 100:
        out["sample_warning"] = (f"個体数 {ep['episodes']} 件。プランは最低100個体を要求している。"
                                 "この件数では取りこぼし率1%の判定に統計的な意味がない")
    return out


# ==========================================================
# 入力: 動画
# ==========================================================
def run_video(path: str, cam: str, scale: float,
              max_frames: int, diagnose: int) -> tuple[list, dict]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"動画を開けません: {path}")

    rows, diagnosed = [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and i >= max_frames:
            break
        frame = common.fit_native(frame, cam)
        ref = evaluate_frame(frame, cam, 1.0)
        test = evaluate_frame(frame, cam, scale)
        rows.append(make_row(i, cam, ref, test))

        # 不一致フレームだけ原因を診断する（analyze は表示用処理が入って重いため件数を絞る）
        if ref["detected"] != test["detected"] and len(diagnosed) < diagnose:
            diagnosed.append({
                "frame": i,
                "ref_status": common.analyze_status(frame, cam, 1.0),
                "test_status": common.analyze_status(frame, cam, scale),
                "ref_area": ref["area"], "test_area": test["area"],
            })
        i += 1
        if i % 200 == 0:
            print(f"  {os.path.basename(path)}: {i} フレーム処理")
    cap.release()
    print(f"  {os.path.basename(path)}: 合計 {i} フレーム")
    return rows, {"diagnosed": diagnosed}


def make_row(idx: int, cam: str, ref: dict, test: dict) -> dict:
    return {
        "frame": idx, "cam": cam,
        "ref_detected": ref["detected"], "test_detected": test["detected"],
        "ref_mx": ref["mx"] if ref["mx"] is not None else -1,
        "test_mx": test["mx"] if test["mx"] is not None else -1,
        "ref_area": ref["area"], "test_area": test["area"],
        "ref_in_window": ref["in_window"], "test_in_window": test["in_window"],
    }


# ==========================================================
# 入力: 実機ライブ
# ==========================================================
def run_live(cams: list, scale: float, duration: float,
             max_frames: int) -> tuple[dict, dict]:
    """Basler カメラから実時間でフレームを取り、同一フレームに対して両縮小率を適用する。

    同じフレームを両方で処理するため、フレーム落ちがあっても比較の公平性は保たれる。
    ただし本スクリプト自体は逐次処理なので、本番より取りこぼすフレームが増える。
    落ちた枚数は seq の飛びとして記録し、レポートへ出す。"""
    common.ensure_repo_cwd()
    from module_cameras_5goki import CameraManager   # pypylon を読む

    mgr = CameraManager()
    if not mgr.init_cameras():
        sys.exit("カメラを初期化できません。本番アプリや hsv_calibration が起動していないか確認してください。")

    targets = [c for c in mgr.controllers if c.name in cams]
    if not targets:
        mgr.stop_all_get_frame()
        sys.exit(f"指定カメラが見つかりません: {cams} / 接続済み: {[c.name for c in mgr.controllers]}")

    print(f"ライブ計測を開始します（{duration:.0f}秒 / {[c.name for c in targets]}）")
    mgr.start_all_get_frame()

    rows_by_cam = {c.name: [] for c in targets}
    seq_by_cam = {c.name: 0 for c in targets}
    dropped = {c.name: 0 for c in targets}

    try:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            for ctrl in targets:
                frame, seq = ctrl.get_next_frame(seq_by_cam[ctrl.name], timeout=0.05)
                if frame is None or seq == seq_by_cam[ctrl.name]:
                    continue
                # seq が2以上進んでいれば、その分は処理が追いつかず捨てたフレーム
                dropped[ctrl.name] += max(0, seq - seq_by_cam[ctrl.name] - 1)
                seq_by_cam[ctrl.name] = seq

                rows = rows_by_cam[ctrl.name]
                if max_frames and len(rows) >= max_frames:
                    continue
                ref = evaluate_frame(frame, ctrl.name, 1.0)
                test = evaluate_frame(frame, ctrl.name, scale)
                rows.append(make_row(len(rows), ctrl.name, ref, test))

            done = sum(len(r) for r in rows_by_cam.values())
            if done and done % 200 == 0:
                el = time.perf_counter() - t0
                print(f"  {el:5.1f}s  取得 {done} フレーム "
                      f"({ {k: len(v) for k, v in rows_by_cam.items()} })")
    except KeyboardInterrupt:
        print("\n中断されました。ここまでの結果で集計します。")
    finally:
        mgr.stop_all_get_frame()

    meta = {"dropped_frames": dropped,
            "note": ("本スクリプトは4カメラを逐次処理するため本番よりフレームを落とす。"
                     "同一フレームに両縮小率を適用しているため比較自体は公平。")}
    return rows_by_cam, meta


# ==========================================================
# 面積閾値のスイープ（プラン §3-4）
# ==========================================================
def sweep_area(rows_frames: list, cam: str, scale: float, factors: list,
               gap: int) -> list:
    """min_blob_area の倍率を振って、基準（scale=1.0）に対する F1 が最大の点を探す。

    面積閾値を単純に scale² 倍しても最適とは限らない。本番は
    min_area = min_blob_area * scale² を使っているので、その値に倍率を掛けて評価する。
    rows_frames は (frame, ref_result) の列。"""
    cfg = common.load_hsv_config(cam)
    base = int(cfg["min_blob_area"])
    results = []

    for f in factors:
        cfg["min_blob_area"] = int(round(base * f))     # キャッシュを一時的に書き換える
        try:
            rows = []
            for i, (frame, ref) in enumerate(rows_frames):
                test = evaluate_frame(frame, cam, scale)
                rows.append(make_row(i, cam, ref, test))
        finally:
            cfg["min_blob_area"] = base                 # 必ず戻す

        s = summarize(rows, gap)
        tp = sum(1 for r in rows if r["ref_detected"] and r["test_detected"])
        fp = s["frame_false"]
        fn = s["frame_miss"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        results.append({
            "factor": f,
            "min_blob_area": int(round(base * f)),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4),
            "episode_miss_rate": s["detect_episodes"]["miss_rate"],
            "delay_median": s["detect_episodes"].get("delay_median"),
        })
        print(f"  倍率 {f:<5g} min_blob_area={results[-1]['min_blob_area']:>7}  "
              f"F1={f1:.4f}  取りこぼし率={results[-1]['episode_miss_rate']:.2%}")
    return results


# ==========================================================
# 出力
# ==========================================================
def write_csv(path: str, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"→ {path}")


def write_report(path: str, result: dict) -> None:
    L = []
    A = L.append
    A("# Task 3 レポート: 半解像度化による検出劣化の検証\n\n")
    A(f"計測日時: {result['env']['timestamp']}  \n")
    A(f"入力: {result['source']}  \n")
    A(f"比較: scale=1.0（基準） vs scale={result['scale']}（比較側）\n")

    for cam, s in result["by_cam"].items():
        if "error" in s:
            A(f"\n## {cam}\n\n{s['error']}\n")
            continue
        A(f"\n## {cam} — **{s['verdict']}**\n\n")
        if s.get("verdict_reason"):
            A(f"> ⚠ {s['verdict_reason']}\n\n")
        A("### 合格基準\n\n| 指標 | 実測 | 判定 |\n|---|---|---|\n")
        for c in s["checks"]:
            A(f"| {c['name']} | {c['value']} | {'✅ 合格' if c['pass'] else '❌ 不合格'} |\n")
        if "sample_warning" in s:
            A(f"\n> ⚠ {s['sample_warning']}\n")

        A("\n### フレーム単位\n\n| 指標 | 値 |\n|---|---|\n")
        A(f"| 総フレーム | {s['frames']} |\n")
        A(f"| 一致率 | {s['frame_agreement']:.4f} |\n")
        A(f"| 取りこぼし（基準◯/比較✕） | {s['frame_miss']} ({s['frame_miss_rate']:.2%}) |\n")
        A(f"| 誤検出（基準✕/比較◯） | {s['frame_false']} |\n")
        if "mx_err_px" in s:
            A(f"| 重心x誤差 p50/p95/max [px] | {s['mx_err_px']['p50']} / "
              f"{s['mx_err_px']['p95']} / {s['mx_err_px']['max']} |\n")
            A(f"| 面積相対誤差 p50/p95/max | {s['area_rel_err']['p50']:.4f} / "
              f"{s['area_rel_err']['p95']:.4f} / {s['area_rel_err']['max']:.4f} |\n")

        for key, title in (
                ("detect_episodes", "個体単位（検出開始）"),
                ("window_episodes", "個体単位（中心窓への進入＝エジェクタ影響）"),
                ("phantom_episodes", "個体単位（逆方向・幽霊個体の検出）")):
            ep = s[key]
            A(f"\n### {title}\n\n| 指標 | 値 |\n|---|---|\n")
            A(f"| 個体数 | {ep['episodes']} |\n")
            A(f"| 取りこぼし | {ep['missed']} ({ep['miss_rate']:.2%}) |\n")
            if "delay_median" in ep:
                A(f"| 遅延 中央値 | {ep['delay_median']} フレーム |\n")
                A(f"| 遅延 平均 | {ep['delay_mean']} フレーム |\n")
                A(f"| 遅延 P95 | {ep['delay_p95']} フレーム |\n")
                A(f"| 遅延 最大 / 最小 | {ep['delay_max']} / {ep['delay_min']} |\n")
                A(f"| 遅延2フレーム以上 | {ep['delay_ge2_count']} 件 "
                  f"({ep['delay_ge2_rate']:.2%}) |\n")

        diag = result.get("diagnosis", {}).get(cam, [])
        if diag:
            A("\n### 不一致フレームの原因（先頭のみ）\n\n")
            A("| frame | 基準の状態 | 比較側の状態 | 基準面積 | 比較面積 |\n|---|---|---|---|---|\n")
            for d in diag:
                A(f"| {d['frame']} | {d['ref_status']} | {d['test_status']} | "
                  f"{d['ref_area']} | {d['test_area']} |\n")
            A("\n> `edge` は左右端接触による棄却。module_yolo.py:66 の既知事項"
              "（OPEN/CLOSEカーネルが5x5固定でscale連動しないため、"
              "縮小画像では端接触判定が逆転しうる）と一致するか確認すること。\n")

        sw = result.get("sweep", {}).get(cam)
        if sw:
            A("\n### 面積閾値のスイープ（プラン §3-4）\n\n")
            A("| 倍率 | min_blob_area | Precision | Recall | F1 | 個体取りこぼし率 | 遅延中央値 |\n")
            A("|---|---|---|---|---|---|---|\n")
            best = max(sw, key=lambda r: r["f1"])
            for r in sw:
                mark = " ←最良" if r is best else ""
                A(f"| {r['factor']}{mark} | {r['min_blob_area']} | {r['precision']:.4f} | "
                  f"{r['recall']:.4f} | {r['f1']:.4f} | {r['episode_miss_rate']:.2%} | "
                  f"{r['delay_median']} |\n")
            A(f"\n**推奨**: min_blob_area = {best['min_blob_area']}"
              f"（現行の {best['factor']}倍）。採用する場合は "
              "`json/hsv_config_{cam}.json` を更新すること。\n")

    A("\n## 実行環境\n\n```json\n")
    import json
    A(json.dumps(result["env"], ensure_ascii=False, indent=2))
    A("\n```\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(L))
    print(f"→ {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 3: 半解像度化の検出劣化検証")
    ap.add_argument("--video", action="append", default=[],
                    help="動画パス。'path:cam_name' 形式でカメラを個別指定できる（複数可）")
    ap.add_argument("--cam", default=None,
                    help="--video でカメラ名を省略した場合に使うカメラ名 / --live の対象（カンマ区切り）")
    ap.add_argument("--live", action="store_true", help="実機カメラからライブで計測する")
    ap.add_argument("--duration", type=float, default=60.0, help="--live の計測秒数")
    ap.add_argument("--scale", type=float, default=None,
                    help="比較側の縮小率。既定は module_yolo.HSV_DETECT_SCALE の現行値")
    ap.add_argument("--gap", type=int, default=2,
                    help="個体区間を繋ぐ許容途切れフレーム数")
    ap.add_argument("--max-frames", type=int, default=0, help="カメラごとの上限フレーム数（0=無制限）")
    ap.add_argument("--diagnose", type=int, default=20,
                    help="原因診断する不一致フレームの最大件数")
    ap.add_argument("--sweep-area", default=None,
                    help="min_blob_area の倍率をスイープ（例 0.6,0.8,1.0,1.2,1.5）")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="task3")
    a = ap.parse_args()

    if not a.video and not a.live:
        sys.exit("--video か --live のどちらかを指定してください。")

    common.ensure_repo_cwd()
    print("module_yolo を読み込んでいます（数秒かかります）...")
    scale = a.scale if a.scale is not None else common.production_scale()
    print(f"基準 scale=1.0 / 比較 scale={scale}")

    result = {"scale": scale, "env": common.env_info(),
              "by_cam": {}, "diagnosis": {}, "sweep": {}}
    odir = common.out_dir(base=a.out_dir)
    rows_by_cam: dict[str, list] = {}

    if a.live:
        cams = [c.strip() for c in (a.cam or ",".join(common.CAM_NAMES)).split(",")]
        rows_by_cam, meta = run_live(cams, scale, a.duration, a.max_frames)
        result["source"] = f"live ({a.duration:.0f}秒)"
        result["live_meta"] = meta
    else:
        specs = []
        for v in a.video:
            if ":" in v and not os.path.exists(v):
                path, cam = v.rsplit(":", 1)
            else:
                path, cam = v, a.cam
            if not cam:
                sys.exit(f"カメラ名が不明です: {v}（--cam を指定するか 'path:cam_name' 形式で書く）")
            if cam not in common.CAM_NAMES:
                sys.exit(f"未知のカメラ名です: {cam}")
            specs.append((path, cam))

        for path, cam in specs:
            print(f"\n{path} ({cam}) を処理します")
            rows, extra = run_video(path, cam, scale, a.max_frames, a.diagnose)
            rows_by_cam.setdefault(cam, []).extend(rows)
            if extra["diagnosed"]:
                result["diagnosis"].setdefault(cam, []).extend(extra["diagnosed"])
        result["source"] = ", ".join(f"{p} ({c})" for p, c in specs)

    # --- 集計 ---
    print("\n" + "=" * 78)
    for cam, rows in rows_by_cam.items():
        if not rows:
            result["by_cam"][cam] = {"error": "フレームが取得できませんでした"}
            print(f"{cam}: フレームなし")
            continue
        write_csv(os.path.join(odir, f"{a.tag}_frames_{cam}.csv"), rows)
        s = summarize(rows, a.gap)
        result["by_cam"][cam] = s
        ep = s["detect_episodes"]
        print(f"\n{cam}: {s['verdict']}")
        print(f"  フレーム一致率 {s['frame_agreement']:.4f}  "
              f"取りこぼし {s['frame_miss']}  誤検出 {s['frame_false']}  "
              f"総 {s['frames']}")
        if s.get("verdict_reason"):
            print(f"  [警告] {s['verdict_reason']}")
            continue
        print(f"  個体 {ep['episodes']} 件 / 取りこぼし {ep['missed']} "
              f"({ep['miss_rate']:.2%}) / 幽霊個体 {s['phantom_episodes']['missed']} 件")
        if "delay_median" in ep:
            print(f"  検出開始遅延 中央値 {ep['delay_median']} / "
                  f"P95 {ep['delay_p95']} / 最大 {ep['delay_max']} フレーム")
        for c in s["checks"]:
            print(f"    {'OK ' if c['pass'] else 'NG '} {c['name']}: {c['value']}")
        if "sample_warning" in s:
            print(f"  [警告] {s['sample_warning']}")

    # --- 面積閾値のスイープ（動画モードのみ。フレームを再読み込みする） ---
    if a.sweep_area and not a.live:
        factors = [float(x) for x in a.sweep_area.split(",")]
        for path, cam in specs:
            print(f"\n{cam}: 面積閾値スイープ")
            cap = cv2.VideoCapture(path)
            pairs = []
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok or (a.max_frames and i >= a.max_frames):
                    break
                frame = common.fit_native(frame, cam)
                pairs.append((frame, evaluate_frame(frame, cam, 1.0)))
                i += 1
            cap.release()
            result["sweep"].setdefault(cam, []).extend(
                sweep_area(pairs, cam, scale, factors, a.gap))
    elif a.sweep_area and a.live:
        print("[warn] --sweep-area はライブモードでは使えません（フレームの再処理が必要なため）")

    common.write_json(os.path.join(odir, f"{a.tag}_summary.json"), result)
    write_report(os.path.join(odir, f"{a.tag}_report.md"), result)

    verdicts = {c: s.get("verdict") for c, s in result["by_cam"].items()}
    print("\n" + "=" * 78)
    print(f"総合: {verdicts}")
    if any(v == "不合格" for v in verdicts.values()):
        print("不合格のカメラがある。プラン §4 の通り、これが解消するまで Task 1/2 を実行しても"
              "投影値の前提が崩れているため意味がない。")
    if any(v == "判定不能" for v in verdicts.values()):
        print("判定不能のカメラがある。入力（果実が写っているか・カメラ名が正しいか）を"
              "確認してから測り直すこと。")


if __name__ == "__main__":
    main()
