"""Task 1: HSV常時監視の実効並列度を特定する。

【なぜやるのか】
現行PCは32論理コア。帯外HSV監視が実際に何コア分を使っているかが分からないと、
Task 2 で測るスケール係数 k を正しく解釈できない。6コアの Nano Super で再現
できない並列度に依存していれば、k だけでは表現できない追加劣化が乗る。

見落としやすいのは「Pythonコード上はシングルスレッドに見えても、OpenCV が
内部で全論理コアを使っている」ケース。本スクリプトは以下の3つを同時に押さえる。

  1-1  プロセス全体・スレッド別のCPU使用率を実測（実効並列度）
  1-2  並列化構文の静的な洗い出し（grep 相当）
  1-3  OpenCV / torch の内部スレッド数とビルド構成の記録

【使い方】
    # 別ターミナルで本番を起動しておく（uv run main.py）
    uv run python standalone/profile_cpu_usage.py --duration 60

    # PID を直接指定する（同名プロセスが複数ある場合はこちらが確実）
    uv run python standalone/profile_cpu_usage.py --pid 12345

    # 計測せず、静的解析と環境情報だけ出す（本番を動かせない環境で使う）
    uv run python standalone/profile_cpu_usage.py --static-only

【判定】プラン §Task 1 の基準
    cpu_percent ≒ 100%     実質1コア。安全側。Task 2 の係数がそのまま使える
    cpu_percent 200〜600%   2〜6コア並列。Nano Super でギリギリ再現可能
    cpu_percent > 600%      6コアでは再現不可。並列度削減による追加劣化を別途見積もる
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jetson_probe_common as common  # noqa: E402


# 1-2 で探す並列化の痕跡。notes §9 の「入れるべき設定」が入っているかも同時に見る。
PARALLEL_PATTERNS = [
    (r"ThreadPoolExecutor",        "スレッドプール"),
    (r"ProcessPoolExecutor",       "プロセスプール"),
    (r"threading\.Thread",         "明示スレッド"),
    (r"multiprocessing",           "マルチプロセス"),
    (r"cv2\.setNumThreads",        "OpenCVスレッド数の明示（notes §9 の必須対策）"),
    (r"torch\.set_num_threads",    "torchスレッド数の明示（notes §9 の必須対策）"),
    (r"QThreadPool|QRunnable|QThread", "Qt側スレッド"),
    (r"joblib|Parallel\(",         "joblib並列"),
    (r"OMP_NUM_THREADS",           "OpenMP環境変数"),
]

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "bench", "analysis",
             "training_images", "evaluated_images", "Trained_Models"}

# 検証スクリプト自身は走査対象から外す。パターン定義行や説明文が引っかかると
# 「cv2.setNumThreads は既に呼ばれている」と誤判定してしまうため。
SELF_SCRIPTS = {"jetson_probe_common.py", "profile_cpu_usage.py",
                "bench_hsv_jetson.py", "project_jetson.py",
                "verify_hsv_scale_video.py"}

SETNUMTHREADS_LABEL = "OpenCVスレッド数の明示（notes §9 の必須対策）"


# ==========================================================
# 1-1 CPU使用率の実測
# ==========================================================
def pick_process(psutil, pid: int | None, name: str):
    """監視対象プロセスを決める。--pid 指定が最優先。"""
    if pid:
        try:
            return psutil.Process(pid)
        except psutil.NoSuchProcess:
            sys.exit(f"PID {pid} のプロセスが見つかりません")

    cands = []
    for p in psutil.process_iter(["name", "pid", "cmdline"]):
        try:
            pname = (p.info["name"] or "").lower()
            cmd = " ".join(p.info["cmdline"] or [])
            if name.lower() in pname or "main.py" in cmd:
                if p.pid == os.getpid():
                    continue          # 自分自身は除く
                cands.append((p, cmd))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not cands:
        sys.exit(f"プロセスが見つかりません: {name}\n"
                 f"  先に別ターミナルで本番（uv run main.py）を起動するか、--pid で指定してください。")

    if len(cands) > 1:
        print("候補が複数あります。CPU使用率が最大のものを選びます:")
        for p, cmd in cands:
            print(f"  PID={p.pid:>7}  {p.info['name']}  {cmd[:80]}")

    # 一度サンプリングしてから最大のものを選ぶ（初回の cpu_percent は必ず0を返すため）
    for p, _ in cands:
        try:
            p.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    time.sleep(0.5)

    def usage(p):
        try:
            return p.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return -1.0

    return max((p for p, _ in cands), key=usage)


def measure(psutil, proc, duration: float, interval: float, csv_path: str) -> dict:
    """プロセス全体とスレッド別のCPU使用率を interval 秒ごとに記録する。

    プロセス全体の cpu_percent は「100% = 1論理コア相当」。
    スレッド別は psutil.Process.threads() のCPU時間差分から自前で算出する
    （プラン §1-1 のスクリプトには無い。どのスレッドが食っているかが分からないと
    「HSVが並列なのか、推論やGUIが並列なのか」を切り分けられないため追加した）。"""
    n_cores = os.cpu_count() or 1
    print(f"監視対象: PID={proc.pid} {proc.name()} / 論理コア {n_cores} / "
          f"{duration:.0f}秒間 {interval:.1f}秒ごと")

    rows = []
    thread_totals: dict[int, float] = {}
    proc.cpu_percent(interval=None)          # 基準点を作る（初回は必ず0）
    prev_threads = {t.id: t.user_time + t.system_time for t in proc.threads()}
    prev_t = time.perf_counter()

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        time.sleep(interval)
        try:
            now = time.perf_counter()
            elapsed = now - prev_t
            prev_t = now

            cpu = proc.cpu_percent(interval=None)
            threads = {t.id: t.user_time + t.system_time for t in proc.threads()}
            mem = proc.memory_info().rss / 1024 ** 2

            # スレッド別のCPU使用率[%]（100% = 1論理コア）
            per_thread = {}
            for tid, total in threads.items():
                delta = total - prev_threads.get(tid, total)
                pct = (delta / elapsed) * 100.0 if elapsed > 0 else 0.0
                per_thread[tid] = pct
                thread_totals[tid] = thread_totals.get(tid, 0.0) + delta
            prev_threads = threads

            busy = sorted(per_thread.values(), reverse=True)
            rows.append({
                "t": round(now - t0, 2),
                "cpu_percent": round(cpu, 1),
                "cores_equiv": round(cpu / 100.0, 2),
                "num_threads": len(threads),
                "rss_mb": round(mem, 1),
                # 「50%以上のCPUを使っているスレッド数」＝実質的に走っているスレッド数
                "threads_busy": sum(1 for v in busy if v >= 50.0),
                "top1_thread_pct": round(busy[0], 1) if busy else 0.0,
                "top4_thread_pct": round(sum(busy[:4]), 1),
            })
            print(f"  t={rows[-1]['t']:>6.1f}s  CPU={cpu:>6.1f}%  "
                  f"({rows[-1]['cores_equiv']:>4.2f}コア相当)  "
                  f"threads={len(threads):>3} (busy {rows[-1]['threads_busy']})  "
                  f"RSS={mem:>7.1f}MB")
        except psutil.NoSuchProcess:
            print("[warn] 対象プロセスが終了しました。計測を打ち切ります。")
            break

    if not rows:
        return {"error": "サンプルが取れませんでした"}

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"→ {csv_path}")

    cpus = [r["cpu_percent"] for r in rows]
    cpus_sorted = sorted(cpus)
    p50 = cpus_sorted[len(cpus_sorted) // 2]
    p95 = cpus_sorted[min(len(cpus_sorted) - 1, int(len(cpus_sorted) * 0.95))]
    peak = max(cpus)

    return {
        "pid": proc.pid,
        "samples": len(rows),
        "cpu_percent_p50": p50,
        "cpu_percent_p95": p95,
        "cpu_percent_peak": peak,
        "cores_equiv_p50": round(p50 / 100.0, 2),
        "cores_equiv_peak": round(peak / 100.0, 2),
        "threads_max": max(r["num_threads"] for r in rows),
        "threads_busy_max": max(r["threads_busy"] for r in rows),
        "rss_mb_peak": max(r["rss_mb"] for r in rows),
        "csv": csv_path,
        "verdict": verdict(p95),
        # 全区間で最もCPU時間を使ったスレッド上位5本（並列度の内訳）
        "top_threads_cpu_sec": [
            {"tid": tid, "cpu_sec": round(sec, 2)}
            for tid, sec in sorted(thread_totals.items(),
                                   key=lambda kv: kv[1], reverse=True)[:5]
        ],
    }


def verdict(cpu_p95: float) -> str:
    """プラン §Task 1 の判定基準に当てはめる。"""
    if cpu_p95 <= 150.0:
        return ("実質1コア（安全側）: Task 2 のスケール係数をそのまま適用できる。"
                "Jetson側では6コアへ並列化する余地が残っている")
    if cpu_p95 <= 600.0:
        return (f"{cpu_p95/100.0:.1f}コア相当の並列: Nano Super（6コア）で再現可能な範囲。"
                "ただし他スレッドと食い合う点は実機で要確認")
    return (f"{cpu_p95/100.0:.1f}コア相当: 6コアでは再現不可。"
            "並列度削減による追加劣化を k とは別に見積もる必要がある")


# ==========================================================
# 1-2 並列化箇所の静的な洗い出し
# ==========================================================
def scan_parallel_constructs(root: str) -> dict:
    """リポジトリ内の並列化構文を列挙する（プラン §1-2 の rg 相当を Python で実装）。
    rg / grep が入っていない環境でも同じ結果が出るようにしてある。

    本番コード（リポジトリ直下）と standalone/ のツール類を分けて返す。
    Jetson の6コアを奪い合うのは本番プロセスだけなので、判断に効くのは前者。"""
    hits: dict[str, dict] = {label: {"production": [], "tools": []}
                             for _pat, label in PARALLEL_PATTERNS}
    compiled = [(re.compile(pat), label) for pat, label in PARALLEL_PATTERNS]

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if not fn.endswith(".py") or fn in SELF_SCRIPTS:
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            bucket = "tools" if rel.startswith("standalone/") else "production"
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if line.lstrip().startswith("#"):
                            continue          # コメント行は実装ではない
                        for rx, label in compiled:
                            if rx.search(line):
                                hits[label][bucket].append(
                                    f"{rel}:{i}: {line.strip()[:100]}")
            except OSError:
                continue
    return {k: v for k, v in hits.items() if v["production"] or v["tools"]}


# ==========================================================
# 1-3 内部スレッド数・ビルド構成
# ==========================================================
def thread_config() -> dict:
    """OpenCV / torch の内部スレッド数を、本番と同じ import 状態で記録する。

    ultralytics の import が cv2.getNumThreads() を1へ書き換える副作用（notes §14.1）が
    あるため、import 前後の両方を残す。ここが食い違うと Task 2 の条件設定を誤る。"""
    import cv2
    before = cv2.getNumThreads()

    try:
        common.load_module_yolo()      # 本番と同じく ultralytics を読み込む
        after = cv2.getNumThreads()
        import_error = None
    except ImportError as e:
        # ultralytics/torch/PySide6 が入っていない環境。静的解析だけは続行する。
        after = before
        import_error = f"{type(e).__name__}: {e}"

    cfg = {
        "cv2_threads_before_import": before,
        "cv2_threads_after_module_yolo_import": after,
        "ultralytics_import_side_effect": before != after,
        "logical_cores": os.cpu_count(),
    }
    if import_error:
        cfg["module_yolo_import_error"] = import_error
        cfg["ultralytics_import_side_effect"] = "未確認（module_yolo を import できず）"
    try:
        import torch
        cfg["torch_num_threads"] = torch.get_num_threads()
        cfg["torch_num_interop_threads"] = torch.get_num_interop_threads()
    except ImportError:
        cfg["torch"] = "未インストール"

    cfg["explicit_setNumThreads_in_code"] = False   # scan 結果で後から上書きする
    return cfg


# ==========================================================
# レポート出力
# ==========================================================
def write_report(path: str, result: dict) -> None:
    L = []
    A = L.append
    A("# Task 1 レポート: HSV常時監視の実効並列度\n")
    A(f"計測日時: {result['env']['timestamp']}\n")

    A("\n## 1-1 CPU使用率の実測\n")
    m = result.get("measure")
    if not m:
        A("\n（--static-only のため未計測）\n")
    elif "error" in m:
        A(f"\n{m['error']}\n")
    else:
        A("\n| 指標 | 値 |\n|---|---|\n")
        A(f"| 対象PID | {m['pid']} |\n")
        A(f"| サンプル数 | {m['samples']} |\n")
        A(f"| CPU使用率 p50 | {m['cpu_percent_p50']:.1f}% ({m['cores_equiv_p50']:.2f}コア相当) |\n")
        A(f"| CPU使用率 p95 | {m['cpu_percent_p95']:.1f}% |\n")
        A(f"| CPU使用率 ピーク | {m['cpu_percent_peak']:.1f}% ({m['cores_equiv_peak']:.2f}コア相当) |\n")
        A(f"| OSスレッド数 最大 | {m['threads_max']} |\n")
        A(f"| 稼働スレッド数（50%超）最大 | {m['threads_busy_max']} |\n")
        A(f"| RSS ピーク | {m['rss_mb_peak']:.1f} MB |\n")
        A(f"\n**判定**: {m['verdict']}\n")
        A("\nCPU時間上位スレッド:\n\n| スレッドID | CPU時間[s] |\n|---|---|\n")
        for t in m["top_threads_cpu_sec"]:
            A(f"| {t['tid']} | {t['cpu_sec']:.2f} |\n")
        A(f"\n生データ: `{m['csv']}`\n")

    A("\n## 1-2 並列化箇所の一覧\n")
    A("\n本番プロセス（リポジトリ直下）と standalone/ のツール類を分けて示す。"
      "Jetson の6コアを奪い合うのは本番プロセスだけなので、判断に効くのは前者。\n")
    hits = result["parallel_constructs"]
    if not hits:
        A("\n該当なし。\n")
    for label, buckets in hits.items():
        A(f"\n### {label}"
          f"（本番 {len(buckets['production'])}件 / ツール {len(buckets['tools'])}件）\n")
        if buckets["production"]:
            A("\n**本番コード**\n\n")
            for ln in buckets["production"]:
                A(f"- `{ln}`\n")
        if buckets["tools"]:
            A("\n<details><summary>standalone/ ツール類</summary>\n\n")
            for ln in buckets["tools"]:
                A(f"- `{ln}`\n")
            A("\n</details>\n")

    A("\n## 1-3 内部スレッド数・ビルド構成\n\n")
    tc = result["thread_config"]
    A("| 項目 | 値 |\n|---|---|\n")
    for k, v in tc.items():
        A(f"| {k} | {v} |\n")
    if tc.get("module_yolo_import_error"):
        A(f"\n> ⚠ `module_yolo` を import できなかったため、ultralytics の副作用は未確認: "
          f"`{tc['module_yolo_import_error']}`\n")
    elif tc.get("ultralytics_import_side_effect") is True:
        A("\n> `module_yolo`（＝ultralytics）の import だけで cv2 のスレッド数が "
          f"{tc['cv2_threads_before_import']} → {tc['cv2_threads_after_module_yolo_import']} "
          "へ変わっている。JETSON_MIGRATION_NOTES.md §14.1 の再現。"
          "ultralytics のバージョン依存の非公式な副作用なので、"
          "`cv2.setNumThreads(1)` は明示的に呼ぶべき。\n")
    if not tc.get("explicit_setNumThreads_in_code"):
        A("\n> **`cv2.setNumThreads()` の明示的な呼び出しがコード中に見つからない。**"
          " notes §9 が「最優先の必須対策」としている設定が未適用の状態。\n")

    A("\n## OpenCV ビルド構成\n\n")
    A("PC↔Jetson の比較では、この差分を必ず記録すること"
      "（JetPack 同梱の OpenCV は CUDA 無効ビルドで並列フレームワークも異なる）。\n\n")
    A("| 項目 | 値 |\n|---|---|\n")
    for k, v in result["env"]["opencv_build"].items():
        A(f"| {k} | {v} |\n")

    A("\n## 実行環境\n\n```json\n")
    import json
    A(json.dumps({k: v for k, v in result["env"].items() if k != "opencv_build"},
                 ensure_ascii=False, indent=2))
    A("\n```\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(L))
    print(f"→ {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 1: 実効並列度の計測")
    ap.add_argument("--pid", type=int, default=None, help="監視対象PID（最優先）")
    ap.add_argument("--name", default="python", help="PID未指定時に探すプロセス名")
    ap.add_argument("--duration", type=float, default=60.0, help="計測秒数")
    ap.add_argument("--interval", type=float, default=1.0, help="サンプリング間隔[秒]")
    ap.add_argument("--static-only", action="store_true",
                    help="プロセス計測をせず、静的解析と環境情報のみ出力する")
    ap.add_argument("--out-dir", default=None, help="出力先（既定 analysis/jetson_validation）")
    ap.add_argument("--tag", default="task1", help="出力ファイル名の接尾辞")
    a = ap.parse_args()

    common.ensure_repo_cwd()
    odir = common.out_dir(base=a.out_dir)

    result = {"parallel_constructs": scan_parallel_constructs(common.repo_root())}
    result["thread_config"] = thread_config()
    # 「対策済みか」は本番コード側の呼び出しだけで判断する（ツール側にあっても本番には効かない）
    result["thread_config"]["explicit_setNumThreads_in_code"] = bool(
        result["parallel_constructs"].get(SETNUMTHREADS_LABEL, {}).get("production"))
    result["env"] = common.env_info()

    if not a.static_only:
        try:
            import psutil
        except ImportError:
            sys.exit("psutil が必要です（uv add psutil、または --static-only で静的解析のみ実行）")
        proc = pick_process(psutil, a.pid, a.name)
        csv_path = os.path.join(odir, f"{a.tag}_cpu_usage.csv")
        result["measure"] = measure(psutil, proc, a.duration, a.interval, csv_path)

    common.write_json(os.path.join(odir, f"{a.tag}_summary.json"), result)
    write_report(os.path.join(odir, f"{a.tag}_report.md"), result)

    if result.get("measure", {}).get("verdict"):
        print("\n判定: " + result["measure"]["verdict"])


if __name__ == "__main__":
    main()
