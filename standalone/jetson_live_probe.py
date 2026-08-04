"""本番（main.py）を動かしたまま Task 1〜3 を同時に計測するプローブ。

run_as_jetson.py から使う。単体で import しても副作用は無い。

──────────────────────────────────────────────────────────────
何をどう測るか
──────────────────────────────────────────────────────────────
Task 1  実効並列度
    プロセス自身を psutil で1秒ごとにサンプリングする。スレッドIDを
    threading.enumerate() の native_id と突き合わせるので、
    「cam-worker-cam_top が何%使っているか」まで名前付きで出る。

Task 2  HSV検出の実処理時間
    ImageProcessor.get_target_info をラップして、本番が実際に呼んだ回数・時間を
    カメラ別に積む。マイクロベンチと違い、GUI・推論・カメラ取得と競合した
    「実際の条件下」の値になる。1個体あたりの合計は本番が既に
    logs_cycle_5goki/*.csv へ書いているので、そちらを analyze_cycle_logs.py で読む。

Task 3  半解像度化の検出劣化
    サンプリング窓の間だけ、本番が処理したのと同じフレームを検証スレッドへ渡し、
    **独立コピーの module_yolo（縮小率1.0固定）** で基準側を計算して突き合わせる。
    グローバル差し替えを使わないので本番の処理に干渉しない（jetson_probe_common
    .load_reference_module の説明を参照）。

──────────────────────────────────────────────────────────────
計測が計測対象を変えてしまう点について
──────────────────────────────────────────────────────────────
Task 3 の検証スレッドは全解像度HSVを走らせるので、その窓の間は CPU を余分に食う
（PC実測で1フレームあたり8ms前後）。**Task 1/2 の数値をきれいに取りたい実行では
Task 3 を切ること**。既定は「30秒検証 → 90秒休み」の間欠なので、休止中の区間だけを
切り出せば Task 1/2 の値も汚れない（レポートに検証窓の時刻を残してある）。
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jetson_probe_common as common  # noqa: E402


# ==========================================================
# Task 1: CPUサンプラ（プロセス内）
# ==========================================================
class CpuSampler:
    """自プロセスのCPU使用率・スレッド別内訳を一定間隔で記録する。"""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.rows: list[dict] = []
        self.thread_cpu: dict[str, float] = {}     # スレッド名 → 累積CPU秒
        self._stop = threading.Event()
        self._t: threading.Thread | None = None
        self._psutil = None

    def start(self) -> bool:
        try:
            import psutil
        except ImportError:
            print("[probe] psutil が無いため Task 1 の計測は行いません")
            return False
        self._psutil = psutil
        self._t = threading.Thread(target=self._run, name="jetson-probe-cpu", daemon=True)
        self._t.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=self.interval * 2 + 1.0)

    @staticmethod
    def _thread_names() -> dict:
        """OSスレッドID → Python側のスレッド名。付いていないものは 'unnamed'。"""
        names = {}
        for t in threading.enumerate():
            nid = getattr(t, "native_id", None)
            if nid is not None:
                names[nid] = t.name
        return names

    def _run(self) -> None:
        psutil = self._psutil
        proc = psutil.Process()
        proc.cpu_percent(interval=None)
        prev = {t.id: t.user_time + t.system_time for t in proc.threads()}
        prev_t = time.perf_counter()

        while not self._stop.wait(self.interval):
            try:
                now = time.perf_counter()
                elapsed = now - prev_t
                prev_t = now
                cpu = proc.cpu_percent(interval=None)
                threads = {t.id: t.user_time + t.system_time for t in proc.threads()}
                names = self._thread_names()

                per_thread = {}
                for tid, total in threads.items():
                    delta = total - prev.get(tid, total)
                    name = names.get(tid, f"tid-{tid}")
                    per_thread[name] = per_thread.get(name, 0.0) + \
                        (delta / elapsed * 100.0 if elapsed > 0 else 0.0)
                    self.thread_cpu[name] = self.thread_cpu.get(name, 0.0) + delta
                prev = threads

                busy = sorted(per_thread.values(), reverse=True)
                self.rows.append({
                    "t": round(time.time(), 2),
                    "cpu_percent": round(cpu, 1),
                    "cores_equiv": round(cpu / 100.0, 2),
                    "num_threads": len(threads),
                    "threads_busy": sum(1 for v in busy if v >= 50.0),
                    "rss_mb": round(proc.memory_info().rss / 1024 ** 2, 1),
                    "top_threads": {k: round(v, 1) for k, v in
                                    sorted(per_thread.items(), key=lambda kv: -kv[1])[:6]},
                })
            except Exception as e:      # 計測でアプリを落とさない
                self.rows.append({"t": round(time.time(), 2), "error": str(e)})

    def summary(self) -> dict:
        vals = [r["cpu_percent"] for r in self.rows if "cpu_percent" in r]
        if not vals:
            return {"error": "サンプルなし"}
        s = sorted(vals)
        p50 = s[len(s) // 2]
        p95 = s[min(len(s) - 1, int(len(s) * 0.95))]
        return {
            "samples": len(s),
            "cpu_percent_p50": p50,
            "cpu_percent_p95": p95,
            "cpu_percent_peak": max(s),
            "cores_equiv_p50": round(p50 / 100.0, 2),
            "cores_equiv_peak": round(max(s) / 100.0, 2),
            "threads_max": max(r.get("num_threads", 0) for r in self.rows),
            "threads_busy_max": max(r.get("threads_busy", 0) for r in self.rows),
            "rss_mb_peak": max(r.get("rss_mb", 0) for r in self.rows),
            "cpu_sec_by_thread": dict(sorted(self.thread_cpu.items(),
                                             key=lambda kv: -kv[1])[:15]),
        }


# ==========================================================
# Task 2 + 3: get_target_info のラッパ
# ==========================================================
class HsvProbe:
    """本番の HSV 検出をラップし、時間計測（Task 2）と縮小率比較（Task 3）を行う。"""

    def __init__(self, verify: bool = False, ref_scale: float = 1.0,
                 window_sec: float = 30.0, period_sec: float = 120.0,
                 queue_size: int = 60):
        self.verify = verify
        self.ref_scale = ref_scale
        self.window_sec = window_sec
        self.period_sec = period_sec

        self.lock = threading.Lock()
        self.timings: dict[str, list] = {}      # cam → [ms, ...]
        self.calls: dict[str, int] = {}
        self.hits: dict[str, int] = {}

        self._orig = None
        self._module = None
        self._ref_mod = None
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._verify_thread: threading.Thread | None = None
        self._stop = threading.Event()

        # 検証窓の管理。窓ごとに連番を振り、キュー溢れがあった窓は遅延解析から外す
        self._t0 = time.monotonic()
        self._window_rows: dict[tuple, list] = {}    # (cam, window_id) → rows
        self._window_drops: dict[int, int] = {}
        self._windows: list[dict] = []

    # ---------- 窓の判定 ----------
    def _window_id(self) -> int | None:
        """いま検証窓の中なら窓ID、外なら None。"""
        if not self.verify:
            return None
        el = time.monotonic() - self._t0
        cycle = int(el // self.period_sec)
        if (el - cycle * self.period_sec) <= self.window_sec:
            return cycle
        return None

    # ---------- インストール ----------
    def install(self) -> None:
        my = common.load_module_yolo()
        self._module = my
        self._orig = my.ImageProcessor.get_target_info

        orig = self._orig
        lock = self.lock
        timings, calls, hits = self.timings, self.calls, self.hits

        def wrapper(frame, cam_name):
            t0 = time.perf_counter()
            res = orig(frame, cam_name)
            dt = (time.perf_counter() - t0) * 1000.0

            with lock:
                timings.setdefault(cam_name, []).append(dt)
                calls[cam_name] = calls.get(cam_name, 0) + 1
                if res is not None:
                    hits[cam_name] = hits.get(cam_name, 0) + 1

            if self.verify:
                wid = self._window_id()
                if wid is not None:
                    # frame は camera 側で毎フレーム新しく複製された配列で、
                    # get_target_info も書き換えないため、参照を渡すだけでよい
                    try:
                        self._q.put_nowait((wid, cam_name, frame, res))
                    except queue.Full:
                        with lock:
                            self._window_drops[wid] = self._window_drops.get(wid, 0) + 1
            return res

        # staticmethod として差し戻す（本番は ImageProcessor.get_target_info(...) で呼ぶ）
        my.ImageProcessor.get_target_info = staticmethod(wrapper)

        if self.verify:
            self._ref_mod = common.load_reference_module(self.ref_scale)
            self._verify_thread = threading.Thread(
                target=self._verify_loop, name="jetson-probe-verify", daemon=True)
            self._verify_thread.start()
            print(f"[probe] Task 3 検証を有効化（基準 scale={self.ref_scale} / "
                  f"{self.window_sec:.0f}秒計測 → {self.period_sec:.0f}秒周期）")

    def uninstall(self) -> None:
        if self._module is not None and self._orig is not None:
            self._module.ImageProcessor.get_target_info = staticmethod(self._orig)
        self._stop.set()
        if self._verify_thread:
            self._verify_thread.join(timeout=5.0)

    # ---------- 検証スレッド ----------
    def _verify_loop(self) -> None:
        import verify_hsv_scale_video as v3      # make_row を共有する

        while not self._stop.is_set():
            try:
                wid, cam, frame, prod_res = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                ref_res = self._ref_mod.ImageProcessor.get_target_info(frame, cam)
                w = frame.shape[1]
                ref = self._to_row(ref_res, w, cam)
                test = self._to_row(prod_res, w, cam)
                key = (cam, wid)
                rows = self._window_rows.setdefault(key, [])
                rows.append(v3.make_row(len(rows), cam, ref, test))
            except Exception as e:
                print(f"[probe] 検証スレッド例外（計測のみ停止）: {type(e).__name__}: {e}")

    @staticmethod
    def _to_row(res, width: int, cam: str) -> dict:
        if res is None:
            return {"detected": False, "mx": None, "area": 0, "in_window": False}
        return {"detected": True, "mx": int(res["mx"]), "area": int(res["area"]),
                "in_window": common.in_infer_window(int(res["mx"]), width, cam)}

    # ---------- 集計 ----------
    def timing_summary(self) -> dict:
        with self.lock:
            out = {}
            for cam, ts in self.timings.items():
                s = common.stats_ms(ts)
                s["calls"] = self.calls.get(cam, 0)
                s["detect_rate"] = round(self.hits.get(cam, 0) /
                                         max(1, self.calls.get(cam, 0)), 4)
                out[cam] = s
            total_p50 = sum(v["p50_ms"] for v in out.values() if "p50_ms" in v)
            return {"by_cam": out, "total_all_cams_p50_ms": round(total_p50, 3)}

    def verify_summary(self, gap: int = 2) -> dict:
        """窓ごとに集計する。キュー溢れのあった窓は連続性が壊れているので
        遅延解析から外し、その旨を残す（黙って混ぜると遅延が過大に出る）。"""
        if not self.verify:
            return {}
        import verify_hsv_scale_video as v3

        by_cam: dict[str, dict] = {}
        excluded = []
        for (cam, wid), rows in sorted(self._window_rows.items()):
            if self._window_drops.get(wid):
                excluded.append({"cam": cam, "window": wid,
                                 "dropped": self._window_drops[wid]})
                continue
            if len(rows) < 10:
                continue
            # 窓をまたぐと時系列が切れるので、窓ごとに集計してから件数で束ねる
            by_cam.setdefault(cam, {"windows": [], "rows": []})
            by_cam[cam]["windows"].append({"window": wid, "frames": len(rows),
                                           "summary": v3.summarize(rows, gap)})
            by_cam[cam]["rows"].extend(rows)

        out = {"excluded_windows": excluded, "by_cam": {}}
        for cam, d in by_cam.items():
            # 窓ごとの結果を足し合わせた「通し」の集計も出す（個体数を稼ぐため）。
            # フレーム番号は窓ごとに0始まりなので、境界で偽の個体が1件増えうる点に注意。
            merged = v3.summarize(d["rows"], gap)
            merged["windows"] = [{"window": w["window"], "frames": w["frames"],
                                  "verdict": w["summary"].get("verdict")}
                                 for w in d["windows"]]
            merged["note"] = ("複数の検証窓を連結して集計している。窓の境界では"
                              "時系列が不連続なため、個体数が窓の数だけ過大になりうる")
            out["by_cam"][cam] = merged
        return out

    def rows_for_csv(self) -> list:
        rows = []
        for (cam, wid), rs in sorted(self._window_rows.items()):
            for r in rs:
                r = dict(r)
                r["window"] = wid
                rows.append(r)
        return rows


# ==========================================================
# 全体をまとめる
# ==========================================================
class LiveProbeSet:
    def __init__(self, out_dir: str, tag: str, profile_cpu: bool = True,
                 verify: bool = False, cpu_interval: float = 1.0,
                 verify_window: float = 30.0, verify_period: float = 120.0,
                 gap: int = 2):
        self.out_dir = out_dir
        self.tag = tag
        self.gap = gap
        self.cpu = CpuSampler(cpu_interval) if profile_cpu else None
        self.hsv = HsvProbe(verify=verify, window_sec=verify_window,
                            period_sec=verify_period)
        self.started_at = None
        self.ended_at = None

    def start(self) -> None:
        self.started_at = time.time()
        self.hsv.install()
        if self.cpu:
            self.cpu.start()
        print(f"[probe] 計測開始 tag={self.tag} → {self.out_dir}")

    def stop_and_report(self, extra: dict | None = None) -> dict:
        self.ended_at = time.time()
        self.hsv.uninstall()
        if self.cpu:
            self.cpu.stop()

        result = {
            "tag": self.tag,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "started_at_iso": time.strftime("%Y-%m-%d %H:%M:%S",
                                            time.localtime(self.started_at)),
            "ended_at_iso": time.strftime("%Y-%m-%d %H:%M:%S",
                                          time.localtime(self.ended_at)),
            "duration_sec": round(self.ended_at - self.started_at, 1),
            "env": common.env_info(),
            "task2_hsv_timing": self.hsv.timing_summary(),
        }
        if extra:
            result.update(extra)
        if self.cpu:
            result["task1_cpu"] = self.cpu.summary()
            result["probe_overhead"] = self._overhead(result["task1_cpu"])
            self._write_cpu_csv()
        if self.hsv.verify:
            result["task3_scale_verify"] = self.hsv.verify_summary(self.gap)
            self._write_verify_csv()

        path = os.path.join(self.out_dir, f"{self.tag}_live_summary.json")
        common.write_json(path, result)
        self._print_summary(result)
        print(f"\n→ {path}")
        return result

    @staticmethod
    def _overhead(task1: dict) -> dict:
        """計測自身が使ったCPUの割合。Task 3 を有効にすると全解像度HSVを余分に走らせるので、
        ここが大きい実行の Task 1/2 の数値はそのまま Jetson 換算に使えない。"""
        by_thread = task1.get("cpu_sec_by_thread", {})
        if not by_thread:
            return {}
        total = sum(by_thread.values())
        probe = sum(v for k, v in by_thread.items() if k.startswith("jetson-probe-"))
        share = probe / total if total > 0 else 0.0
        out = {"probe_cpu_sec": round(probe, 2), "total_cpu_sec": round(total, 2),
               "probe_share": round(share, 4)}
        if share >= 0.10:
            out["warning"] = (
                f"計測プローブがCPU時間の {share:.0%} を使っている。"
                "Task 1（実効並列度）と Task 2（HSV時間）の値はこの分だけ悪化しているため、"
                "Jetson換算にはこの実行ではなく --no-profile-cpu を外した"
                "「Task 3 無効（--verify-scale なし）」の実行を使うこと。"
                "Task 3 の一致率・遅延の結論はこの影響を受けない（同じフレームを比べているため）")
        return out

    def _write_cpu_csv(self) -> None:
        import csv
        rows = [r for r in self.cpu.rows if "cpu_percent" in r]
        if not rows:
            return
        path = os.path.join(self.out_dir, f"{self.tag}_cpu_usage.csv")
        cols = ["t", "cpu_percent", "cores_equiv", "num_threads", "threads_busy",
                "rss_mb", "top_threads"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                r = dict(r)
                r["top_threads"] = json.dumps(r.get("top_threads", {}), ensure_ascii=False)
                w.writerow(r)
        print(f"→ {path}")

    def _write_verify_csv(self) -> None:
        import csv
        rows = self.hsv.rows_for_csv()
        if not rows:
            return
        path = os.path.join(self.out_dir, f"{self.tag}_scale_frames.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"→ {path}")

    @staticmethod
    def _print_summary(r: dict) -> None:
        print("\n" + "=" * 78)
        print(f"計測結果  {r['started_at_iso']} 〜 {r['ended_at_iso']}"
              f"（{r['duration_sec']:.0f}秒）")
        print("=" * 78)

        t1 = r.get("task1_cpu")
        if t1 and "error" not in t1:
            print(f"\n[Task 1] CPU p50={t1['cpu_percent_p50']:.1f}% "
                  f"({t1['cores_equiv_p50']:.2f}コア相当) / "
                  f"p95={t1['cpu_percent_p95']:.1f}% / "
                  f"peak={t1['cpu_percent_peak']:.1f}%")
            print(f"          スレッド {t1['threads_max']} 本（稼働 {t1['threads_busy_max']} 本）"
                  f" / RSS ピーク {t1['rss_mb_peak']:.0f} MB")
            print("          CPU時間の内訳[秒]:")
            for name, sec in list(t1["cpu_sec_by_thread"].items())[:8]:
                print(f"            {name:<28} {sec:8.1f}")

        t2 = r.get("task2_hsv_timing", {})
        if t2.get("by_cam"):
            print(f"\n[Task 2] HSV検出（本番実行中の実測）")
            for cam, s in t2["by_cam"].items():
                print(f"          {cam:<12} p50={s['p50_ms']:>6.2f}ms  "
                      f"p95={s['p95_ms']:>6.2f}ms  {s['calls']:>6}回  "
                      f"検出率={s['detect_rate']:.0%}")
            print(f"          4台合計 p50 = {t2['total_all_cams_p50_ms']:.2f} ms/フレーム")

        t3 = r.get("task3_scale_verify", {})
        if t3.get("by_cam"):
            print(f"\n[Task 3] 縮小率の比較（基準1.0 vs 本番）")
            for cam, s in t3["by_cam"].items():
                ep = s.get("detect_episodes", {})
                print(f"          {cam:<12} {s.get('verdict')}  "
                      f"一致率={s.get('frame_agreement')}  "
                      f"個体={ep.get('episodes')} 取りこぼし={ep.get('missed')} "
                      f"遅延中央値={ep.get('delay_median')}")
            if t3.get("excluded_windows"):
                print(f"          ※ キュー溢れで除外した窓: {t3['excluded_windows']}")

        ov = r.get("probe_overhead", {})
        if ov.get("warning"):
            print(f"\n[!] {ov['warning']}")
            print(f"    プローブ {ov['probe_cpu_sec']}秒 / 全体 {ov['total_cpu_sec']}秒")
