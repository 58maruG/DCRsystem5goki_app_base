"""Task 2（最優先）: 制約条件下でのCPUスケール係数 k を実測する。

【なぜやるのか】
ゼミ資料の k = 2.0 / 3.0 という仮定に根拠がない。i9-13900HX の P-core は実効5GHz級・
AVX2 対応、対する Cortex-A78AE は約1.7GHz・NEON は128bit幅。HSV変換は x86 SIMD 最適化が
最も効く処理であり、k = 4〜5 は現実的にあり得る。判定式より k > 5.4 で目標未達となるため、
ここが購入可否の分岐点になる。

【本スクリプトが測るもの】
**本番の module_yolo.ImageProcessor.get_target_info をそのまま呼ぶ。**
検証プラン §2-1 のサンプルコード（単帯 inRange + 5x5 OPEN + countNonZero）は
実装と別物であり、それを測ると軽い処理の値が出て k を過小評価する。実装は
  2帯 inRange → OPEN(5x5)x2 → CLOSE(5x5)x2 → connectedComponentsWithStats
  → 虚像除去 → 果柄除去(半径20) → 左右端接触の棄却
であり、支配的なのは morphology と connectedComponents。ここを測らないと意味がない。

【使い方】
    # 2-2 基準値（フルスペック）
    uv run python standalone/bench_hsv_jetson.py --tag ref_full

    # 2-3(a) コア数制限（1コア / 6コア）
    uv run python standalone/bench_hsv_jetson.py --cores 1 --cv-threads 1 --tag c1
    uv run python standalone/bench_hsv_jetson.py --cores 6 --cv-threads 1 --tag c6

    # 2-3(c) クロック制限も掛けた状態で（クロック制限はOS側で行う。下記参照）
    uv run python standalone/bench_hsv_jetson.py --cores 6 --cv-threads 1 --tag jetson_proxy

    # 実画像を使う（推奨。無ければ擬似フレームへ自動フォールバック）
    uv run python standalone/bench_hsv_jetson.py --image-dir training_images/healthy --limit 50

    # 縮小率の効果だけを見る（1.0 と 0.5 を同条件で比較）
    uv run python standalone/bench_hsv_jetson.py --scale-sweep 1.0,0.7,0.5 --tag sweep

【クロック制限（スクリプトの外でやること）】
  Windows (管理者PowerShell): A78AE 約1.7GHz / i9ブースト約5.4GHz → 31%前後
      powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 31
      powercfg /setactive SCHEME_CURRENT
      # 計測後は必ず 100 に戻す。戻し忘れると以降の学習が全部遅くなる
  Linux:
      sudo cpupower frequency-set -u 1700MHz    # 計測後 -u 5400MHz で復帰

  実クロックは出力JSONの env.cpu_freq_mhz に記録される。制限が効いているか必ず確認すること。
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jetson_probe_common as common  # noqa: E402


# ==========================================================
# 計測本体
# ==========================================================
def bench_camera(cam_name: str, frames: list, scale: float,
                 iters: int, warmup: int) -> dict:
    """1カメラ分の get_target_info を iters 回まわして計測する。

    frames が複数あるときは順番に使い回す（実画像セットの分布をならすため）。
    1回の呼び出しに対して1つのサンプルを取り、外れ値の影響を見るため p95/p99 も残す。"""
    detect = common.detect
    n = len(frames)

    with common.detect_scale(scale):
        for i in range(warmup):
            detect(frames[i % n], cam_name)

        samples = []
        hits = 0
        for i in range(iters):
            frame = frames[i % n]
            t0 = time.perf_counter()
            res = detect(frame, cam_name)
            samples.append((time.perf_counter() - t0) * 1000.0)
            if res is not None:
                hits += 1

    h, w = frames[0].shape[:2]
    out = common.stats_ms(samples)
    out.update({
        "cam": cam_name,
        "scale": scale,
        # 入力はネイティブ解像度。HSV変換以降は縮小後の解像度で走る（get_target_info と同じ）
        "input_resolution": f"{w}x{h}",
        "processed_resolution": f"{max(1, round(w * scale))}x{max(1, round(h * scale))}",
        "detect_rate": round(hits / max(1, iters), 4),
    })
    if hits == 0:
        out["warning"] = ("検出0件。果実の写っていないフレームばかりだと1段目で抜けるため"
                          "軽い経路しか測れていない。--image-dir に果実ありの画像を指定すること")
    elif hits < iters * 0.5:
        out["warning"] = f"検出率 {out['detect_rate']:.0%}。2段目（整形）を通った回数が少ない"
    return out


def bench_breakdown(cam_name: str, frame: np.ndarray, scale: float,
                    iters: int) -> dict:
    """内訳（参考値）。get_target_info の中で使われている OpenCV 呼び出しを
    同じ入力に対して個別に計測する。判定ロジックは再実装せず、素の関数呼び出しの
    時間だけを取るので、本番との乖離は生じない。

    合計は get_target_info の実測値と一致しない（切り出し・分岐・numpy 変換が
    入らないため）。どの工程が支配的かを見るためだけに使う。"""
    from hsv_mask_utils import mask_from_hsv, remove_stem, remove_reflection_sat

    cfg = common.load_hsv_config(cam_name)
    h, w = frame.shape[:2]

    def timeit(fn, n=iters):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000.0)
        return round(statistics.median(ts), 3)

    res = {}
    if scale != 1.0:
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        res["resize"] = timeit(lambda: cv2.resize(frame, size,
                                                  interpolation=cv2.INTER_AREA))
        det = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    else:
        res["resize"] = 0.0
        det = frame

    res["cvtColor"] = timeit(lambda: cv2.cvtColor(det, cv2.COLOR_BGR2HSV))
    hsv = cv2.cvtColor(det, cv2.COLOR_BGR2HSV)

    res["mask_from_hsv"] = timeit(lambda: mask_from_hsv(hsv, cfg))
    mask = mask_from_hsv(hsv, cfg)

    res["connectedComponents"] = timeit(
        lambda: cv2.connectedComponentsWithStats(mask))

    # 2段目（整形）は候補ブロブ周辺の切り出しに対して行われる。全画面に掛けると
    # 本番の4倍以上のコストになるため、本番と同じ切り出し範囲で測る。
    n_lab, _lab, stats, _cent = cv2.connectedComponentsWithStats(mask)
    if n_lab > 1 and cfg.get("_refine"):
        idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        stem = max(0, int(round(cfg["stem_open"] * scale)))
        pad = max(stem, 4) + 1
        dh, dw = det.shape[:2]
        bx, by = int(stats[idx, cv2.CC_STAT_LEFT]), int(stats[idx, cv2.CC_STAT_TOP])
        bw, bh = int(stats[idx, cv2.CC_STAT_WIDTH]), int(stats[idx, cv2.CC_STAT_HEIGHT])
        x0, y0 = max(0, bx - pad), max(0, by - pad)
        x1, y1 = min(dw, bx + bw + pad), min(dh, by + bh + pad)
        sub = mask[y0:y1, x0:x1]
        s_ch, v_ch = hsv[y0:y1, x0:x1, 1], hsv[y0:y1, x0:x1, 2]
        res["remove_reflection_sat"] = timeit(
            lambda: remove_reflection_sat(sub, s_ch, v_ch,
                                          cfg["reflect_sat_lo"], cfg["reflect_sat_hi"],
                                          cfg["reflect_v_lo"], cfg["reflect_v_hi"]))
        sub_r = remove_reflection_sat(sub, s_ch, v_ch,
                                      cfg["reflect_sat_lo"], cfg["reflect_sat_hi"],
                                      cfg["reflect_v_lo"], cfg["reflect_v_hi"])
        res["remove_stem"] = timeit(lambda: remove_stem(sub_r, stem))
        res["_refine_roi"] = f"{x1-x0}x{y1-y0}"
    else:
        res["_refine"] = "整形なし（設定が全域＝無効、または候補ブロブ無し）"

    return res


# ==========================================================
# main
# ==========================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Task 2: HSV検出のCPUスケール係数の実測")
    ap.add_argument("--cam", default=",".join(common.CAM_NAMES),
                    help="対象カメラ（カンマ区切り）")
    ap.add_argument("--scale", type=float, default=None,
                    help="縮小率。既定は module_yolo.HSV_DETECT_SCALE の現行値")
    ap.add_argument("--scale-sweep", default=None,
                    help="複数の縮小率をまとめて計測（例 1.0,0.7,0.5）")
    ap.add_argument("--cores", type=int, default=0,
                    help="使用論理コア数（0=制限なし）。Nano Super 相当は 6")
    ap.add_argument("--cv-threads", type=int, default=-1,
                    help="cv2.setNumThreads の値（-1=触らない / 1=notes §9 の推奨）")
    ap.add_argument("--torch-threads", type=int, default=-1,
                    help="torch.set_num_threads の値（-1=触らない）")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--image", default=None, help="実カメラ画像1枚のパス")
    ap.add_argument("--image-dir", default=None,
                    help="実画像のディレクトリ（再帰。カメラ共通で使う）")
    ap.add_argument("--limit", type=int, default=30,
                    help="--image-dir から読む最大枚数")
    ap.add_argument("--breakdown", action="store_true", help="工程別の内訳（参考値）も出す")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="baseline")
    a = ap.parse_args()

    common.ensure_repo_cwd()

    # 実行条件の固定。module_yolo の import（＝ultralytics の副作用）より後に
    # cv2.setNumThreads を掛けるため、この1呼び出しに順序をまとめてある。
    print("実行条件を固定しています（module_yolo の import に数秒かかります）...")
    applied = common.apply_cpu_limits(a.cores, a.cv_threads, a.torch_threads)

    cams = [c.strip() for c in a.cam.split(",") if c.strip()]
    for c in cams:
        if c not in common.CAM_NAMES:
            sys.exit(f"未知のカメラ名です: {c}（{common.CAM_NAMES} のいずれか）")

    if a.scale_sweep:
        scales = [float(s) for s in a.scale_sweep.split(",")]
    else:
        scales = [a.scale if a.scale is not None else common.production_scale()]

    result = {
        "tag": a.tag,
        "applied_limits": applied,
        "iters": a.iters,
        "warmup": a.warmup,
        "env": common.env_info(),
        "cameras": {},
    }

    print(f"\ncv2 threads={cv2.getNumThreads()} / affinity={applied.get('affinity')} / "
          f"scales={scales}\n")

    for cam in cams:
        frames, src = common.load_frames(cam, a.image, a.image_dir, a.limit)
        entry = {"input": src, "by_scale": {}}
        for sc in scales:
            st = bench_camera(cam, frames, sc, a.iters, a.warmup)
            entry["by_scale"][f"{sc:g}"] = st
            warn = f"  ※ {st['warning']}" if "warning" in st else ""
            print(f"{cam:<12} scale={sc:<4g} {st['processed_resolution']:>9}  "
                  f"p50={st['p50_ms']:>7.3f}ms  mean={st['mean_ms']:>7.3f}ms  "
                  f"p95={st['p95_ms']:>7.3f}ms  検出率={st['detect_rate']:.0%}{warn}")
            if a.breakdown:
                entry.setdefault("breakdown", {})[f"{sc:g}"] = bench_breakdown(
                    cam, frames[0], sc, min(a.iters, 200))
        result["cameras"][cam] = entry

    # 4台合計（1フレーム分の全カメラHSVコスト）。これが「帯外常時監視」の1周期分。
    result["total_4cam_ms"] = {}
    for sc in scales:
        key = f"{sc:g}"
        total_p50 = sum(result["cameras"][c]["by_scale"][key]["p50_ms"] for c in cams)
        total_mean = sum(result["cameras"][c]["by_scale"][key]["mean_ms"] for c in cams)
        result["total_4cam_ms"][key] = {"p50": round(total_p50, 3),
                                        "mean": round(total_mean, 3)}
        print(f"\n{len(cams)}台合計 scale={sc:g}: p50={total_p50:.3f}ms / "
              f"mean={total_mean:.3f}ms（1フレーム分）")

    if a.breakdown:
        print("\n工程別の内訳（参考値・中央値[ms]）:")
        for cam in cams:
            for sc, bd in result["cameras"][cam].get("breakdown", {}).items():
                items = "  ".join(f"{k}={v}" for k, v in bd.items()
                                  if not k.startswith("_"))
                print(f"  {cam:<12} scale={sc:<4}  {items}")

    odir = common.out_dir(base=a.out_dir)
    path = common.write_json(os.path.join(odir, f"bench_hsv_{a.tag}.json"), result)
    print(f"\n→ {path}")
    print("\n次の手順: 基準値と制限条件の2つが揃ったら\n"
          f"  uv run python standalone/project_jetson.py "
          f"--ref {os.path.join(odir, 'bench_hsv_ref_full.json')} "
          f"--proxy {os.path.join(odir, 'bench_hsv_jetson_proxy.json')}")


if __name__ == "__main__":
    main()
