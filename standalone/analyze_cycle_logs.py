"""本番が書いた cycle ログから「1個体あたりの実処理コスト」を集計し、k を実測する。

【なぜマイクロベンチではなくこれを使うのか】
bench_hsv_jetson.py は HSV 単体を隔離して測る。実際の締切を決めるのは
「カメラ取得＋HSV＋前処理＋推論＋後処理が4カメラ・全フレームで積み上がった合計」で、
本番は既にそれを1個体ごとに logs_cycle_5goki/cycle_YYYYMMDD.csv へ書いている
（dcr_logger.CYCLE_COLUMNS の total_per_fruit[ms] とその内訳）。

run_as_jetson.py で「制限なし」と「Jetson相当」の2本を同じ条件で運転すれば、
その2つの区間を突き合わせるだけで **実アプリのスケール係数** が出る。
マイクロベンチと違い、GUI・スレッド競合・カメラ取得の影響が全部入っている。

【使い方】
    # 1本の実行区間を集計する
    uv run python standalone/analyze_cycle_logs.py \
        --run analysis/jetson_validation/jetson_live_summary.json

    # 2本を突き合わせて k を出す
    uv run python standalone/analyze_cycle_logs.py \
        --ref  analysis/jetson_validation/baseline_live_summary.json \
        --proxy analysis/jetson_validation/jetson_live_summary.json

    # 実行区間のJSONが無い場合は時刻で切る
    uv run python standalone/analyze_cycle_logs.py \
        --since "2026-08-04 13:00:00" --until "2026-08-04 13:20:00"

【工程別に k を出す理由】
GPU（推論本体）は本スクリプトのCPU制限では遅くならない。全部まとめた k を取ると
GPUの分だけ薄まって過小評価になる。工程ごとに k を出せば、
  - hsv_sum / preproc_sum / postproc_sum / capture_sum … CPU律速。ここの k が Task 2 の答え
  - infer_sum … k≒1 ならGPU律速、k≫1 ならPython/ultralytics側のCPUオーバーヘッド律速
    （notes §15 では 36.08ms のうち 27.7ms がPython側。TensorRT化では消えない）
の切り分けまで同時にできる。
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jetson_probe_common as common  # noqa: E402


TS_FMT = "%Y-%m-%d %H:%M:%S.%f"
LOG_GLOB = os.path.join("logs_cycle_5goki", "cycle_*.csv")

# 集計する列。CPU側／GPU側の切り分けもここで持つ
STAGE_COLUMNS = [
    ("capture_sum[ms]",  "撮影(変換+複製)", "cpu"),
    ("hsv_sum[ms]",      "HSV検出",         "cpu"),
    ("preproc_sum[ms]",  "前処理(crop/resize)", "cpu"),
    ("infer_sum[ms]",    "推論",            "gpu"),
    ("postproc_sum[ms]", "後処理(track/描画)", "cpu"),
]
TOTAL_COLUMN = "total_per_fruit[ms]"
COUNT_COLUMNS = ["hsv_frames[n]", "visible_frames[n]", "infer_count[n]",
                 "capture_frames[n]", "cycle_dur[s]"]


def parse_ts(s: str):
    s = (s or "").strip()
    for fmt in (TS_FMT, "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_cycles(since: datetime | None, until: datetime | None,
                pattern: str = LOG_GLOB) -> list:
    """cycle CSV を読み、時刻範囲で絞る。'#' 始まりのセッション境界行は読み飛ばす。"""
    paths = sorted(glob.glob(os.path.join(common.repo_root(), pattern)))
    if not paths:
        sys.exit(f"cycle ログが見つかりません: {pattern}\n"
                 "  本番を一度運転してからもう一度実行してください。")

    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8", newline="") as f:
            lines = [ln for ln in f if not ln.startswith("#")]
        for row in csv.DictReader(lines):
            ts = parse_ts(row.get("timestamp", ""))
            if ts is None:
                continue
            if since and ts < since:
                continue
            if until and ts > until:
                continue
            row["_ts"] = ts
            rows.append(row)
    return rows


def aggregate(rows: list) -> dict:
    """1個体ごとの値を工程別に集計する。中央値を主指標にする（外れ値に強いため）。"""
    if not rows:
        return {"cycles": 0, "error": "対象区間に個体が1件もありません"}

    out = {"cycles": len(rows),
           "from": rows[0]["_ts"].strftime(TS_FMT),
           "to": rows[-1]["_ts"].strftime(TS_FMT),
           "stages": {}}

    for col, label, kind in STAGE_COLUMNS:
        vals = [v for v in (to_float(r.get(col)) for r in rows) if v is not None]
        if vals:
            out["stages"][col] = {
                "label": label, "kind": kind, "n": len(vals),
                "p50": round(statistics.median(vals), 2),
                "mean": round(statistics.fmean(vals), 2),
                "p95": round(sorted(vals)[min(len(vals) - 1, int(len(vals) * 0.95))], 2),
            }

    totals = [v for v in (to_float(r.get(TOTAL_COLUMN)) for r in rows) if v is not None]
    if totals:
        st = sorted(totals)
        out["total_per_fruit_ms"] = {
            "p50": round(statistics.median(totals), 2),
            "mean": round(statistics.fmean(totals), 2),
            "p95": round(st[min(len(st) - 1, int(len(st) * 0.95))], 2),
            "max": round(max(totals), 2),
            "over_1000ms": sum(1 for v in totals if v >= 1000.0),
            "over_1000ms_rate": round(sum(1 for v in totals if v >= 1000.0) / len(totals), 4),
        }
        # CPU側だけの合計（GPU＝推論を除く）。Jetson換算で k を掛ける対象はこちら
        cpu_p50 = sum(s["p50"] for c, s in out["stages"].items() if s["kind"] == "cpu")
        gpu_p50 = sum(s["p50"] for c, s in out["stages"].items() if s["kind"] == "gpu")
        out["cpu_side_p50_ms"] = round(cpu_p50, 2)
        out["gpu_side_p50_ms"] = round(gpu_p50, 2)

    for col in COUNT_COLUMNS:
        vals = [v for v in (to_float(r.get(col)) for r in rows) if v is not None]
        if vals:
            out.setdefault("counts", {})[col] = round(statistics.median(vals), 2)
    return out


def compare(ref: dict, proxy: dict) -> dict:
    """工程別に k を出す。GPU工程の k が1に近ければ、CPU制限が効いている証拠になる。"""
    out = {"stages": {}}
    for col, label, kind in STAGE_COLUMNS:
        a = ref["stages"].get(col)
        b = proxy["stages"].get(col)
        if not a or not b or a["p50"] <= 0:
            continue
        out["stages"][col] = {
            "label": label, "kind": kind,
            "ref_p50": a["p50"], "proxy_p50": b["p50"],
            "k": round(b["p50"] / a["p50"], 3),
        }

    if ref.get("cpu_side_p50_ms") and proxy.get("cpu_side_p50_ms"):
        out["k_cpu_side"] = round(proxy["cpu_side_p50_ms"] / ref["cpu_side_p50_ms"], 3)
        out["cpu_side_ref_ms"] = ref["cpu_side_p50_ms"]
        out["cpu_side_proxy_ms"] = proxy["cpu_side_p50_ms"]
    if ref.get("total_per_fruit_ms") and proxy.get("total_per_fruit_ms"):
        out["k_total"] = round(proxy["total_per_fruit_ms"]["p50"] /
                               ref["total_per_fruit_ms"]["p50"], 3)

    notes = []
    gpu = [v for v in out["stages"].values() if v["kind"] == "gpu"]
    if gpu:
        kg = gpu[0]["k"]
        if kg > 1.5:
            notes.append(f"推論の k が {kg} と大きい。GPU律速ではなく Python/ultralytics 側の"
                         "CPUオーバーヘッドが支配的（notes §15 と同じ傾向）。"
                         "TensorRT化ではこの分は消えないため、Jetson換算では推論もCPU側として"
                         "扱う必要がある")
        elif kg < 1.15:
            notes.append(f"推論の k が {kg} でほぼ1.0。GPU側はCPU制限の影響を受けておらず、"
                         "制限が意図通りCPUだけに効いている")
    if out.get("k_cpu_side", 0) and out["k_cpu_side"] < 1.15:
        notes.append("CPU側の k が 1.15 未満。コア数・クロックの制限が実際には効いていない"
                     "可能性が高い。run_as_jetson.py の出力で clock.applied を確認すること")
    if notes:
        out["notes"] = notes
    return out


def load_run_window(path: str) -> tuple:
    """run_as_jetson.py が書いた live_summary.json から実行区間を取り出す。"""
    if not os.path.exists(path):
        sys.exit(f"実行サマリが見つかりません: {path}")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    since = parse_ts(d.get("started_at_iso", ""))
    until = parse_ts(d.get("ended_at_iso", ""))
    return since, until, d


def print_agg(title: str, a: dict) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    if a.get("error"):
        print(f"  {a['error']}")
        return
    print(f"  区間: {a['from']} 〜 {a['to']}  /  個体数: {a['cycles']}")
    print(f"\n  {'工程':<22} {'p50[ms]':>9} {'mean[ms]':>9} {'p95[ms]':>9}  種別")
    print(f"  {'-'*22} {'-'*9} {'-'*9} {'-'*9}  ----")
    for col, s in a["stages"].items():
        print(f"  {s['label']:<22} {s['p50']:>9.2f} {s['mean']:>9.2f} "
              f"{s['p95']:>9.2f}  {s['kind'].upper()}")
    if a.get("total_per_fruit_ms"):
        t = a["total_per_fruit_ms"]
        print(f"  {'-'*22} {'-'*9} {'-'*9} {'-'*9}")
        print(f"  {'1個体あたり合計':<22} {t['p50']:>9.2f} {t['mean']:>9.2f} {t['p95']:>9.2f}")
        print(f"\n  CPU側合計(p50) = {a['cpu_side_p50_ms']:.2f} ms  /  "
              f"GPU側(p50) = {a['gpu_side_p50_ms']:.2f} ms")
        print(f"  目標1000ms超過: {t['over_1000ms']} 件 / {a['cycles']} 件 "
              f"({t['over_1000ms_rate']:.2%})")
    if a.get("counts"):
        print("\n  1個体あたりの枚数（中央値）: " +
              "  ".join(f"{k}={v:g}" for k, v in a["counts"].items()))


def main() -> None:
    ap = argparse.ArgumentParser(description="cycle ログから1個体あたりの実コストを集計する")
    ap.add_argument("--run", default=None, help="1本の実行区間（live_summary.json）")
    ap.add_argument("--ref", default=None, help="基準側の live_summary.json")
    ap.add_argument("--proxy", default=None, help="制限側の live_summary.json")
    ap.add_argument("--since", default=None, help="開始時刻 'YYYY-MM-DD HH:MM:SS'")
    ap.add_argument("--until", default=None, help="終了時刻 'YYYY-MM-DD HH:MM:SS'")
    ap.add_argument("--glob", default=LOG_GLOB, help="cycle CSV のパターン")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="cycle_analysis")
    a = ap.parse_args()

    common.ensure_repo_cwd()
    result = {"source_glob": a.glob}

    if a.ref and a.proxy:
        s1, u1, d1 = load_run_window(a.ref)
        s2, u2, d2 = load_run_window(a.proxy)
        agg_ref = aggregate(load_cycles(s1, u1, a.glob))
        agg_proxy = aggregate(load_cycles(s2, u2, a.glob))
        print_agg(f"基準 ({d1.get('tag')})", agg_ref)
        print_agg(f"制限 ({d2.get('tag')})", agg_proxy)

        if agg_ref.get("error") or agg_proxy.get("error"):
            sys.exit("\nどちらかの区間に個体が無いため比較できません。")

        cmp_ = compare(agg_ref, agg_proxy)
        result.update({"ref": agg_ref, "proxy": agg_proxy, "comparison": cmp_,
                       "ref_run": d1.get("tag"), "proxy_run": d2.get("tag")})

        print(f"\n{'=' * 78}\n工程別スケール係数\n{'=' * 78}")
        print(f"  {'工程':<22} {'基準[ms]':>10} {'制限[ms]':>10} {'k':>7}  種別")
        print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*7}  ----")
        for col, s in cmp_["stages"].items():
            print(f"  {s['label']:<22} {s['ref_p50']:>10.2f} {s['proxy_p50']:>10.2f} "
                  f"{s['k']:>7.3f}  {s['kind'].upper()}")
        if "k_cpu_side" in cmp_:
            print(f"\n  CPU側合計の k = {cmp_['k_cpu_side']:.3f}"
                  f"（{cmp_['cpu_side_ref_ms']:.1f} → {cmp_['cpu_side_proxy_ms']:.1f} ms）")
        if "k_total" in cmp_:
            print(f"  全体の k       = {cmp_['k_total']:.3f}"
                  "（GPUが薄めるので、判定にはCPU側の k を使うこと）")
        for n in cmp_.get("notes", []):
            print(f"\n  [注意] {n}")

        if "k_cpu_side" in cmp_:
            print(f"\n次の手順: この k を Jetson 換算に掛ける")
            print(f"  uv run python standalone/project_jetson.py "
                  f"--k-measured {cmp_['k_cpu_side']} "
                  f"--cpu-ms-pc {cmp_['cpu_side_ref_ms']}")
    else:
        if a.run:
            since, until, d = load_run_window(a.run)
            title = f"実行区間 ({d.get('tag')})"
        else:
            since, until = parse_ts(a.since or ""), parse_ts(a.until or "")
            title = "指定区間" if (since or until) else "全期間"
        agg = aggregate(load_cycles(since, until, a.glob))
        print_agg(title, agg)
        result["aggregate"] = agg
        if not agg.get("error") and agg.get("cpu_side_p50_ms"):
            print(f"\n次の手順: 制限側の実行と突き合わせて k を出す")
            print(f"  uv run python standalone/analyze_cycle_logs.py "
                  f"--ref <baseline>_live_summary.json --proxy <jetson>_live_summary.json")

    odir = common.out_dir(base=a.out_dir)
    print(f"\n→ {common.write_json(os.path.join(odir, f'{a.tag}.json'), result)}")


if __name__ == "__main__":
    main()
