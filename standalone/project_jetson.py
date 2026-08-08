"""Task 2-5: 実測したスケール係数を1個体あたり処理時間へ換算し、GO/NO GO を判定する。

【判定式】
  Nano Super の GPU 側は 12枚 × 7.17ms = 86.0ms で固定。目標 1000ms から引くと
  CPU 側の予算は 914ms。半解像度化後のCPU側処理が現行PCで 170ms なので、
  許容スケール係数の上限は k_max = 914 / 170 ≈ 5.38。

  | 実測 k        | 判定        | アクション |
  |---------------|-------------|-----------|
  | k ≤ 4.0       | GO          | 安全マージン25%以上。購入可 |
  | 4.0 < k ≤ 5.4 | 条件付きGO  | GPUオフロード（VPI）を前提に購入 |
  | k > 5.4       | NO GO       | 現構成では目標未達。要再設計 or NX検討 |

【使い方】
    # ベンチ結果2つから k を自動算出して判定する
    uv run python standalone/project_jetson.py \
        --ref  analysis/jetson_validation/bench_hsv_ref_full.json \
        --proxy analysis/jetson_validation/bench_hsv_jetson_proxy.json

    # k を直接与える（実測済みの値を手で入れる場合）
    uv run python standalone/project_jetson.py --k-measured 3.2

    # k の一覧表だけ見る（従来のスライド投影の再現）
    uv run python standalone/project_jetson.py

【ISA差の補正について】
x86 の AVX2 は256bit幅、ARM の NEON は128bit幅。cvtColor/inRange/morphology は
8bit整数SIMDで最も差が出る領域なので、クロックとコア数を揃えただけの k_measured は
実機の劣化を過小評価する。安全側に 1.4倍（--isa-factor で変更可）を上乗せする。

**この係数自体が推定値である**点は報告に必ず明記すること。1.0 と 1.4 と 1.5 で
判定が変わる場合は、その旨も併記する（本スクリプトは境界をまたぐと警告を出す）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jetson_probe_common as common  # noqa: E402


# ==========================================================
# 既定値（第16回ゼミ発表資料の実測・投影値）
# ==========================================================
DEFAULTS = {
    # 半解像度化後のCPU側合計[ms]（HSV + 前処理 + 後処理。現行PC実測）
    "cpu_ms_pc": 170.0,
    # 各機種の TensorRT FP16 推論[ms/回]
    "gpu_ms_per_infer": {"NanoSuper": 7.17, "OrinNX": 6.41},
    # 1個体あたりの推論枚数（4カメラ × INFER_FRAMES_PER_CAM）
    "infer_frames": 12,
    "target_ms": 1000.0,
    "isa_factor": 1.4,
}

VERDICT_GO, VERDICT_COND, VERDICT_NG = "GO", "条件付きGO", "NO GO"


def k_max(cpu_ms_pc: float, gpu_ms: float, target_ms: float) -> float:
    """CPU側予算から許容スケール係数の上限を求める。"""
    return (target_ms - gpu_ms) / cpu_ms_pc


def project(k: float, cpu_ms_pc: float, gpu_ms: float) -> float:
    """スケール係数 k のときの1個体あたり処理時間[ms]。"""
    return cpu_ms_pc * k + gpu_ms


def judge(k: float, k_go: float, k_limit: float) -> str:
    if k <= k_go:
        return VERDICT_GO
    if k <= k_limit:
        return VERDICT_COND
    return VERDICT_NG


def load_bench(path: str) -> dict:
    if not os.path.exists(path):
        sys.exit(f"ベンチ結果が見つかりません: {path}\n"
                 "  先に bench_hsv_jetson.py を実行してください。")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def total_ms(bench: dict, scale: str | None) -> tuple[float, str]:
    """ベンチ結果から「4台合計・1フレーム分」の p50 を取り出す。"""
    totals = bench.get("total_4cam_ms") or {}
    if not totals:
        sys.exit(f"total_4cam_ms がありません: tag={bench.get('tag')}")
    if scale and scale in totals:
        key = scale
    elif len(totals) == 1:
        key = next(iter(totals))
    else:
        sys.exit(f"縮小率が複数あります（{list(totals)}）。--scale で1つ指定してください。")
    return float(totals[key]["p50"]), key


def compare_conditions(ref: dict, proxy: dict, scale: str | None) -> dict:
    """基準値と Jetson 相当条件を突き合わせて k_measured を出す。"""
    ref_ms, ref_scale = total_ms(ref, scale)
    proxy_ms, proxy_scale = total_ms(proxy, scale)

    if ref_scale != proxy_scale:
        sys.exit(f"縮小率が揃っていません（ref={ref_scale} / proxy={proxy_scale}）。"
                 "同じ縮小率どうしで比較してください。")

    out = {
        "ref_tag": ref.get("tag"), "proxy_tag": proxy.get("tag"),
        "scale": ref_scale,
        "ref_total_ms": round(ref_ms, 3),
        "proxy_total_ms": round(proxy_ms, 3),
        "k_measured": round(proxy_ms / ref_ms, 3) if ref_ms > 0 else None,
        "ref_limits": ref.get("applied_limits"),
        "proxy_limits": proxy.get("applied_limits"),
    }

    # 条件が本当に変わっているかを確認する。クロック制限の戻し忘れ・掛け忘れは
    # 「k がほぼ1.0」という形で現れ、これを見逃すと誤って GO を出してしまう。
    notes = []
    if out["k_measured"] is not None and out["k_measured"] < 1.15:
        notes.append("k_measured が 1.15 未満。コア数・クロックの制限が実際には"
                     "掛かっていない可能性が高い。env.cpu_freq_mhz を両者で確認すること")
    rf = (ref.get("env") or {}).get("cpu_freq_mhz")
    pf = (proxy.get("env") or {}).get("cpu_freq_mhz")
    if rf and pf:
        out["ref_freq_mhz"] = rf
        out["proxy_freq_mhz"] = pf
        if pf.get("current", 0) > rf.get("current", 0) * 0.8:
            notes.append(f"クロックが下がっていない（ref {rf['current']}MHz → "
                         f"proxy {pf['current']}MHz）。制限が効いているか確認すること")
    if notes:
        out["warnings"] = notes
    return out


def build_table(k_list, cpu_ms_pc, gpu, target_ms, k_go, k_limits) -> list:
    rows = []
    for k in k_list:
        row = {"k": k}
        for name, gpu_ms in gpu.items():
            ms = project(k, cpu_ms_pc, gpu_ms)
            row[name] = {"ms": round(ms, 1),
                         "ok": ms < target_ms,
                         "verdict": judge(k, k_go, k_limits[name])}
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 2-5: Jetson換算と GO/NO GO 判定")
    ap.add_argument("--ref", default=None, help="基準値（フルスペック）のベンチJSON")
    ap.add_argument("--proxy", default=None, help="Jetson相当条件のベンチJSON")
    ap.add_argument("--scale", default=None,
                    help="比較に使う縮小率（ベンチが複数縮小率を含む場合に指定）")
    ap.add_argument("--k-measured", type=float, default=None,
                    help="k を直接指定する（--ref/--proxy を使わない場合）")
    ap.add_argument("--isa-factor", type=float, default=DEFAULTS["isa_factor"],
                    help="ISA差(NEON128bit vs AVX2)の安全側補正。既定 1.4")
    ap.add_argument("--cpu-ms-pc", type=float, default=DEFAULTS["cpu_ms_pc"],
                    help="半解像度化後のCPU側合計[ms]（現行PC実測）。既定 170.0")
    ap.add_argument("--infer-frames", type=int, default=DEFAULTS["infer_frames"],
                    help="1個体あたりの推論枚数。既定 12")
    ap.add_argument("--target-ms", type=float, default=DEFAULTS["target_ms"],
                    help="目標スループット[ms/個]。既定 1000")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="task2")
    a = ap.parse_args()

    common.ensure_repo_cwd()

    gpu = {name: round(ms * a.infer_frames, 2)
           for name, ms in DEFAULTS["gpu_ms_per_infer"].items()}
    k_limits = {name: k_max(a.cpu_ms_pc, ms, a.target_ms) for name, ms in gpu.items()}
    # GO の閾値は「マージン25%以上」＝ 上限の 0.75 倍。プランの k≤4.0 とほぼ同じ値になる。
    k_go = round(min(k_limits.values()) * 0.75, 2)

    result = {
        "params": {"cpu_ms_pc": a.cpu_ms_pc, "infer_frames": a.infer_frames,
                   "target_ms": a.target_ms, "isa_factor": a.isa_factor,
                   "gpu_ms_total": gpu},
        "k_max": {k: round(v, 3) for k, v in k_limits.items()},
        "k_go_threshold": k_go,
    }

    print("=" * 78)
    print("判定式")
    print("=" * 78)
    for name, gpu_ms in gpu.items():
        print(f"  {name:<10} GPU固定 {gpu_ms:>6.1f}ms（{DEFAULTS['gpu_ms_per_infer'][name]}ms × "
              f"{a.infer_frames}枚） → CPU予算 {a.target_ms - gpu_ms:>6.1f}ms "
              f"→ k_max = {k_limits[name]:.2f}")
    print(f"  GO 閾値（マージン25%）: k ≤ {k_go}")

    # --- k の決定 ---
    k_measured = a.k_measured
    if a.ref and a.proxy:
        cmp_ = compare_conditions(load_bench(a.ref), load_bench(a.proxy), a.scale)
        result["comparison"] = cmp_
        k_measured = cmp_["k_measured"]
        print("\n" + "=" * 78)
        print("実測比較")
        print("=" * 78)
        print(f"  基準  ({cmp_['ref_tag']}):   {cmp_['ref_total_ms']:>8.3f} ms  "
              f"cores={cmp_['ref_limits'].get('affinity')} "
              f"cv_threads={cmp_['ref_limits'].get('cv2_num_threads')}")
        print(f"  制限  ({cmp_['proxy_tag']}): {cmp_['proxy_total_ms']:>8.3f} ms  "
              f"cores={cmp_['proxy_limits'].get('affinity')} "
              f"cv_threads={cmp_['proxy_limits'].get('cv2_num_threads')}")
        print(f"  縮小率: {cmp_['scale']}")
        print(f"  k_measured = {k_measured:.3f}")
        for w in cmp_.get("warnings", []):
            print(f"  [警告] {w}")
    elif a.ref or a.proxy:
        sys.exit("--ref と --proxy は両方指定してください。")

    # --- 換算表 ---
    k_list = [2.0, 3.0, 4.0, 5.0, 5.4, 6.0]
    if k_measured is not None:
        k_list = sorted(set(k_list + [round(k_measured, 2),
                                      round(k_measured * a.isa_factor, 2)]))
    result["table"] = build_table(k_list, a.cpu_ms_pc, gpu, a.target_ms, k_go, k_limits)

    print("\n" + "=" * 78)
    print(f"1個体あたり処理時間（CPU {a.cpu_ms_pc}ms × k + GPU固定）")
    print("=" * 78)
    print(f"  {'k':>6} | {'NanoSuper':<22} | {'OrinNX':<22}")
    print(f"  {'-'*6}-+-{'-'*22}-+-{'-'*22}")
    for row in result["table"]:
        mark = ""
        if k_measured is not None:
            if abs(row["k"] - round(k_measured, 2)) < 1e-9:
                mark = "  ← k_measured"
            elif abs(row["k"] - round(k_measured * a.isa_factor, 2)) < 1e-9:
                mark = f"  ← k_estimated (×{a.isa_factor})"
        cells = []
        for name in gpu:
            c = row[name]
            cells.append(f"{c['ms']:>8.1f}ms {'OK' if c['ok'] else 'NG'} "
                         f"{c['verdict']}".ljust(22))
        print(f"  {row['k']:>6.2f} | {cells[0]} | {cells[1]}{mark}")

    # --- 最終判定 ---
    if k_measured is not None:
        k_est = k_measured * a.isa_factor
        verdict = judge(k_est, k_go, k_limits["NanoSuper"])
        ms_nano = project(k_est, a.cpu_ms_pc, gpu["NanoSuper"])
        margin = (1.0 - ms_nano / a.target_ms) * 100.0

        result["k_measured"] = round(k_measured, 3)
        result["k_estimated"] = round(k_est, 3)
        result["projected_ms_nano_super"] = round(ms_nano, 1)
        result["margin_percent"] = round(margin, 1)
        result["verdict"] = verdict

        print("\n" + "=" * 78)
        print("最終判定")
        print("=" * 78)
        print(f"  k_measured   = {k_measured:.3f}（コア数・クロック制限の実測比）")
        print(f"  k_estimated  = {k_est:.3f}（× ISA差補正 {a.isa_factor}）")
        print(f"  Nano Super 1個体あたり = {ms_nano:.1f} ms / 目標 {a.target_ms:.0f} ms")
        print(f"  マージン = {margin:.1f}%")
        print(f"\n  === {verdict} ===")
        if verdict == VERDICT_GO:
            print("  安全マージン25%以上。購入して問題ない。")
        elif verdict == VERDICT_COND:
            print("  マージン不足。VPI等によるGPUオフロード（Appendix A）を前提に購入する。")
        else:
            print("  現構成では目標未達。HSVのGPUオフロードを先に実装するか、Orin NX を検討する。")

        # ISA補正の係数次第で判定が変わるなら、その旨を必ず出す。
        alt = {f: judge(k_measured * f, k_go, k_limits["NanoSuper"])
               for f in (1.0, 1.3, 1.4, 1.5)}
        if len(set(alt.values())) > 1:
            result["verdict_sensitivity"] = alt
            print("\n  [注意] ISA補正の係数で判定が変わる。報告には幅を明記すること:")
            for f, v in alt.items():
                print(f"    ×{f}: k={k_measured*f:.2f} → {v}")
    else:
        print("\n※ k の実測値が未指定のため判定は保留。")
        print("  bench_hsv_jetson.py を ref_full / jetson_proxy の2条件で実行し、"
              "--ref/--proxy を渡してください。")

    odir = common.out_dir(base=a.out_dir)
    path = common.write_json(os.path.join(odir, f"{a.tag}_projection.json"), result)
    print(f"\n→ {path}")


if __name__ == "__main__":
    main()
