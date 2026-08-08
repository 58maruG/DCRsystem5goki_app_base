"""本番（main.py）を動かしたまま、実効並列度・実クロック・HSV処理時間を同時に計測するプローブ。

run_as_jetson.py から使う。単体で import しても副作用は無い。

──────────────────────────────────────────────────────────────
何をどう測るか
──────────────────────────────────────────────────────────────
CPU（実効並列度）
    プロセス自身を psutil で1秒ごとにサンプリングする。スレッドIDを
    threading.enumerate() の native_id と突き合わせるので、
    「cam-worker-cam_top が何%使っているか」まで名前付きで出る。

クロック（実効クロック）
    Windows の `% Processor Performance` カウンタを PowerShell 越しに継続ポーリングする。
    powercfg PROCTHROTTLEMAX は自己申告（run_as_jetson.py の clock.applied）だけでは
    「本当にそのクロックまで落ちたか」を保証しないため、実行中ずっと直接実測する
    （旧 verify_clock_throttle.ps1 の手動検証をこのプローブに統合した）。

HSV検出の実処理時間
    ImageProcessor.get_target_info をラップして、本番が実際に呼んだ回数・時間を
    カメラ別に積む。マイクロベンチと違い、GUI・推論・カメラ取得と競合した
    「実際の条件下」の値になる。1個体あたりの合計は本番が既に
    logs_cycle_5goki/*.csv へ書いているので、そちらを analyze_cycle_logs.py で読む。
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jetson_probe_common as common  # noqa: E402


# ==========================================================
# CPUサンプラ（プロセス内の実効並列度）
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
            print("[probe] psutil が無いため CPU 計測は行いません")
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
            "cores_equiv_p95": round(p95 / 100.0, 2),
            "cores_equiv_peak": round(max(s) / 100.0, 2),
            "threads_max": max(r.get("num_threads", 0) for r in self.rows),
            "threads_busy_max": max(r.get("threads_busy", 0) for r in self.rows),
            "rss_mb_peak": max(r.get("rss_mb", 0) for r in self.rows),
            "cpu_sec_by_thread": dict(sorted(self.thread_cpu.items(),
                                             key=lambda kv: -kv[1])[:15]),
        }


# ==========================================================
# クロックサンプラ（実効クロックの継続実測。Windows専用）
# ==========================================================
class ClockSampler:
    """`% Processor Performance` カウンタを PowerShell 越しに継続ポーリングする。

    powercfg PROCTHROTTLEMAX は「コマンドが通ったか」しか分からない自己申告値
    （run_as_jetson.py の clock.applied）なので、Turbo Boost/Speed Shift 搭載の
    CPU では狙った値まで落ちないことがある（実測で確認済み）。本サンプラは
    実行中ずっとカウンタを直接読み、実効クロックを実測値として残す。
    """

    COUNTER = r"\Processor Information(_Total)\% Processor Performance"

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def start(self) -> bool:
        if platform.system() != "Windows":
            return False
        self._t = threading.Thread(target=self._run, name="jetson-probe-clock", daemon=True)
        self._t.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=15.0)   # powershell.exe 起動中でも取りこぼさないよう長めに待つ

    def _run(self) -> None:
        cmd = ["powershell", "-NoProfile", "-Command",
               f"(Get-Counter '{self.COUNTER}' -SampleInterval 1 -MaxSamples 1)"
               ".CounterSamples.CookedValue"]
        # powershell.exe の起動コスト自体が実測4〜6秒ある（Get-Counterの1秒待ちより支配的）。
        # interval はあくまで「最短間隔」として扱い、実測にかかった時間ぶんは差し引く。
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=15.0)
                v = float(r.stdout.strip())
                self.samples.append(v)
            except (ValueError, subprocess.SubprocessError, OSError):
                pass  # 1回の失敗でスキップ（管理者権限が無くても読めるカウンタだが念のため）
            remaining = self.interval - (time.monotonic() - t0)
            if self._stop.wait(max(0.0, remaining)):
                break

    def summary(self, base_mhz: float) -> dict:
        """base_mhz: env_info() の cpu_freq_mhz.max（%はこの値に対する比率）。"""
        if not self.samples or not base_mhz:
            return {}
        s = sorted(self.samples)
        n = len(s)

        def pct(p: float) -> float:
            return s[min(n - 1, int(n * p))]

        mean = sum(s) / n
        return {
            "n": n,
            "base_mhz": base_mhz,
            "pct_p50": round(pct(0.50), 1),
            "pct_mean": round(mean, 1),
            "pct_max": round(max(s), 1),
            "effective_ghz_p50": round(base_mhz / 1000.0 * pct(0.50) / 100.0, 3),
            "effective_ghz_mean": round(base_mhz / 1000.0 * mean / 100.0, 3),
            "effective_ghz_max": round(base_mhz / 1000.0 * max(s) / 100.0, 3),
        }


# ==========================================================
# HSV検出のラッパ（実処理時間の計測）
# ==========================================================
class HsvProbe:
    """本番の HSV 検出をラップし、実際に呼ばれた回数・時間をカメラ別に積む。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.timings: dict[str, list] = {}      # cam → [ms, ...]
        self.calls: dict[str, int] = {}
        self.hits: dict[str, int] = {}
        self._orig = None
        self._module = None

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
            return res

        # staticmethod として差し戻す（本番は ImageProcessor.get_target_info(...) で呼ぶ）
        my.ImageProcessor.get_target_info = staticmethod(wrapper)

    def uninstall(self) -> None:
        if self._module is not None and self._orig is not None:
            self._module.ImageProcessor.get_target_info = staticmethod(self._orig)

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


# ==========================================================
# 全体をまとめる
# ==========================================================
class LiveProbeSet:
    def __init__(self, out_dir: str, tag: str, profile_cpu: bool = True,
                 profile_clock: bool = True, cpu_interval: float = 1.0,
                 clock_interval: float = 2.0):
        self.out_dir = out_dir
        self.tag = tag
        self.cpu = CpuSampler(cpu_interval) if profile_cpu else None
        self.clock = ClockSampler(clock_interval) if profile_clock else None
        self.hsv = HsvProbe()
        self.started_at = None
        self.ended_at = None

    def start(self) -> None:
        self.started_at = time.time()
        self.hsv.install()
        if self.cpu:
            self.cpu.start()
        if self.clock:
            ok = self.clock.start()
            if not ok:
                print("[probe] クロック実測は Windows 専用のためスキップします")
        print(f"[probe] 計測開始 tag={self.tag} → {self.out_dir}")

    def stop_and_report(self, extra: dict | None = None) -> dict:
        self.ended_at = time.time()
        self.hsv.uninstall()
        if self.cpu:
            self.cpu.stop()
        if self.clock:
            self.clock.stop()

        env = common.env_info()
        result = {
            "tag": self.tag,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "started_at_iso": time.strftime("%Y-%m-%d %H:%M:%S",
                                            time.localtime(self.started_at)),
            "ended_at_iso": time.strftime("%Y-%m-%d %H:%M:%S",
                                          time.localtime(self.ended_at)),
            "duration_sec": round(self.ended_at - self.started_at, 1),
            "env": env,
            "hsv_timing": self.hsv.timing_summary(),
        }
        if extra:
            result.update(extra)
        if self.cpu:
            result["cpu_usage"] = self.cpu.summary()
            result["probe_overhead"] = self._overhead(result["cpu_usage"])
            self._write_cpu_csv()
        if self.clock:
            base_mhz = env.get("cpu_freq_mhz", {}).get("max", 0)
            result["clock_measured"] = self.clock.summary(base_mhz)

        path = os.path.join(self.out_dir, f"{self.tag}_live_summary.json")
        common.write_json(path, result)
        self._print_summary(result)
        print(f"\n→ {path}")
        return result

    @staticmethod
    def _overhead(cpu_usage: dict) -> dict:
        """計測プローブ自身が使ったCPUの割合。大きいとTask本体の数値が悪化して見える。"""
        by_thread = cpu_usage.get("cpu_sec_by_thread", {})
        if not by_thread:
            return {}
        total = sum(by_thread.values())
        probe = sum(v for k, v in by_thread.items() if k.startswith("jetson-probe-"))
        share = probe / total if total > 0 else 0.0
        out = {"probe_cpu_sec": round(probe, 2), "total_cpu_sec": round(total, 2),
               "probe_share": round(share, 4)}
        if share >= 0.10:
            out["warning"] = f"計測プローブがCPU時間の {share:.0%} を使っています。参考値として扱ってください。"
        return out

    def _write_cpu_csv(self) -> None:
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

    @staticmethod
    def _print_summary(r: dict) -> None:
        print("\n" + "=" * 78)
        print(f"計測結果  {r['started_at_iso']} 〜 {r['ended_at_iso']}"
              f"（{r['duration_sec']:.0f}秒）")
        print("=" * 78)

        t1 = r.get("cpu_usage")
        if t1 and "error" not in t1:
            print(f"\n[CPU] p50={t1['cpu_percent_p50']:.1f}% "
                  f"({t1['cores_equiv_p50']:.2f}コア相当) / "
                  f"p95={t1['cpu_percent_p95']:.1f}% / "
                  f"peak={t1['cpu_percent_peak']:.1f}%")
            print(f"      スレッド {t1['threads_max']} 本（稼働 {t1['threads_busy_max']} 本）"
                  f" / RSS ピーク {t1['rss_mb_peak']:.0f} MB")

        ck = r.get("clock_measured")
        if ck:
            print(f"\n[クロック実測] 実効 {ck['effective_ghz_p50']:.2f}GHz（p50） "
                  f"/ 平均 {ck['effective_ghz_mean']:.2f}GHz "
                  f"/ 最大 {ck['effective_ghz_max']:.2f}GHz"
                  f"（基準 {ck['base_mhz']/1000:.2f}GHz 比、n={ck['n']}）")

        t2 = r.get("hsv_timing", {})
        if t2.get("by_cam"):
            print(f"\n[HSV検出] 本番実行中の実測")
            for cam, s in t2["by_cam"].items():
                print(f"      {cam:<12} p50={s['p50_ms']:>6.2f}ms  "
                      f"p95={s['p95_ms']:>6.2f}ms  {s['calls']:>6}回  "
                      f"検出率={s['detect_rate']:.0%}")
            print(f"      4台合計 p50 = {t2['total_all_cams_p50_ms']:.2f} ms/フレーム")

        ov = r.get("probe_overhead", {})
        if ov.get("warning"):
            print(f"\n[!] {ov['warning']}")
