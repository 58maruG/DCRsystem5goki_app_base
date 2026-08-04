"""本番（main.py）を Jetson Orin Nano Super 相当の制約下で起動し、Task 1〜3 を同時計測する。

    uv run python standalone/run_as_jetson.py --preset baseline   # 無制限（基準取り）
    uv run python standalone/run_as_jetson.py --preset jetson      # 6コア/1.7GHz相当
    uv run python standalone/run_as_jetson.py --preset jetson --verify-scale   # Task 3 も同時に

GUIを閉じる（またはCtrl-C）と計測を締めてレポートを書き、クロック設定を元に戻す。

──────────────────────────────────────────────────────────────
何を「Jetson相当」にできて、何ができないか
──────────────────────────────────────────────────────────────
できる:
  コア数      CPU affinity で6コアへ制限（Orin Nano Super は A78AE 6コア）
  クロック    Windows は powercfg、Linux は cpupower で上限を下げる
              i9-13900HX ブースト約5.4GHz に対し A78AE 約1.7GHz → 既定31%
  スレッド数  cv2.setNumThreads(1) / torch.set_num_threads(2) / OMP_NUM_THREADS
              （notes §9 の必須対策。OpenMPプールは import 時に確定するため
                本スクリプトは cv2/torch を読む前に環境変数を設定する）

できない（数値の解釈時に必ず考慮すること）:
  ISA        AVX2(256bit) を NEON(128bit) にはできない。HSV処理はここで最も差が出る。
             project_jetson.py の --isa-factor（既定1.4）で安全側に補正する
  GPU        RTX 4090 Laptop を Orin の iGPU にはできない。アーキテクチャもSM数も別物で、
             クロックを下げても等価にならない。GPU側は TensorRT FP16 の公表実測値
             （Nano Super 7.17ms/回）を使って project_jetson.py で合成する
  メモリ     8GB共有メモリの再現はしない。cgroup/Job Object で絞ると計測中にOOMで
             落ちるだけで得るものが無いため。代わりに RSS を記録し、
             8GB に対する余裕をレポートに出す（Appendix B のリスク評価用）
  熱         サーマルスロットリングは再現不可。実機で要確認（Appendix B）

──────────────────────────────────────────────────────────────
使い方（Task 2 の k を実測する手順）
──────────────────────────────────────────────────────────────
  1. 基準を取る（制限なし）。数分〜10分ほど普通に運転する
       uv run python standalone/run_as_jetson.py --preset baseline --tag ref_full
  2. 制限を掛けて同じだけ運転する
       uv run python standalone/run_as_jetson.py --preset jetson --tag jetson_proxy
  3. 本番が書いた1個体あたりの実コストを突き合わせる
       uv run python standalone/analyze_cycle_logs.py \
           --ref  analysis/jetson_validation/ref_full_live_summary.json \
           --proxy analysis/jetson_validation/jetson_proxy_live_summary.json

【クロック設定の戻し忘れについて】
正常終了・Ctrl-C・例外のいずれでも自動で戻す。ただし強制終了（kill -9 / タスクの
強制終了）では戻らない。起動時に復元コマンドを表示するので、その場合は手で戻すこと。
"""

from __future__ import annotations

import argparse
import atexit
import os
import platform
import signal
import subprocess
import sys


# ==========================================================
# 引数（cv2/torch を import する前に解釈する必要がある）
# ==========================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="main.py を Jetson 相当の制約下で起動し Task 1〜3 を同時計測する",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=["baseline", "jetson"], default="jetson",
                   help="baseline=制限なし（基準取り） / jetson=6コア+クロック制限")
    p.add_argument("--cores", type=int, default=None,
                   help="使用論理コア数（0=制限なし）。--preset の既定を上書きする")
    p.add_argument("--cv-threads", type=int, default=None,
                   help="cv2.setNumThreads の値（-1=触らない）")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="torch.set_num_threads の値（-1=触らない）")
    p.add_argument("--omp-threads", type=int, default=None,
                   help="OMP_NUM_THREADS（cv2/torch の import 前に設定する）")
    p.add_argument("--clock-percent", type=int, default=None,
                   help="Windows: プロセッサ最大状態[%%]。31 が A78AE 1.7GHz 相当。0で無効")
    p.add_argument("--clock-mhz", type=int, default=None,
                   help="Linux: cpupower で設定する上限[MHz]。1700 が A78AE 相当。0で無効")
    p.add_argument("--no-clock", action="store_true", help="クロック制限を行わない")

    p.add_argument("--verify-scale", action="store_true",
                   help="Task 3（縮小率の比較）も同時に行う。CPUを余分に食う点に注意")
    p.add_argument("--verify-window", type=float, default=30.0,
                   help="Task 3 を計測する窓の長さ[秒]")
    p.add_argument("--verify-period", type=float, default=120.0,
                   help="Task 3 の窓の周期[秒]。既定は120秒ごとに30秒計測")
    p.add_argument("--no-profile-cpu", action="store_true", help="Task 1 の計測を行わない")
    p.add_argument("--cpu-interval", type=float, default=1.0, help="Task 1 のサンプリング間隔[秒]")

    p.add_argument("--main", default="main.py",
                   help="起動するエントリポイント（既定 main.py）")
    p.add_argument("--out-dir", default=None, help="出力先（既定 analysis/jetson_validation）")
    p.add_argument("--tag", default=None, help="出力ファイル名の接頭辞。既定は preset 名")
    p.add_argument("--dry-run", action="store_true",
                   help="制限内容を表示するだけで main.py を起動しない")
    return p


def resolve(args) -> dict:
    """preset の既定値と個別指定をマージする。"""
    if args.preset == "jetson":
        cfg = {"cores": 6, "cv_threads": 1, "torch_threads": 2, "omp_threads": 2,
               "clock_percent": 31, "clock_mhz": 1700}
    else:
        cfg = {"cores": 0, "cv_threads": -1, "torch_threads": -1, "omp_threads": 0,
               "clock_percent": 0, "clock_mhz": 0}

    for key, val in (("cores", args.cores), ("cv_threads", args.cv_threads),
                     ("torch_threads", args.torch_threads),
                     ("omp_threads", args.omp_threads),
                     ("clock_percent", args.clock_percent),
                     ("clock_mhz", args.clock_mhz)):
        if val is not None:
            cfg[key] = val
    if args.no_clock:
        cfg["clock_percent"] = 0
        cfg["clock_mhz"] = 0
    cfg["tag"] = args.tag or args.preset
    return cfg


# ==========================================================
# CPUクロックの制限と復元
# ==========================================================
class ClockLimiter:
    """OSの電源設定でCPUクロック上限を下げ、終了時に必ず戻す。

    戻し忘れると以降の学習・計測が全部遅くなるため、正常終了/例外/シグナルの
    どの経路でも restore() が呼ばれるようにしてある（強制終了は除く）。"""

    def __init__(self, percent: int = 0, mhz: int = 0):
        self.percent = percent
        self.mhz = mhz
        self.applied = False
        self.system = platform.system()
        self.restore_cmd = ""
        self.detail = ""

    # ---------- 判定 ----------
    def _is_jetson(self) -> bool:
        return os.path.exists("/etc/nvpmodel.conf")

    def _has_admin(self) -> bool:
        if self.system == "Windows":
            try:
                import ctypes
                return bool(ctypes.windll.shell32.IsUserAnAdmin())
            except Exception:
                return False
        return hasattr(os, "geteuid") and os.geteuid() == 0

    def wanted(self) -> bool:
        if self.system == "Windows":
            return self.percent > 0
        return self.mhz > 0

    # ---------- 適用 ----------
    def apply(self) -> bool:
        if not self.wanted():
            self.detail = "クロック制限なし"
            return False
        if self._is_jetson():
            self.detail = "Jetson実機のためクロック制限は行わない（nvpmodel で管理すること）"
            return False
        if not self._has_admin():
            self.detail = ("管理者/root 権限が無いためクロック制限を適用できない。"
                           "権限付きで実行し直すか、--no-clock で明示的に外すこと")
            return False

        if self.system == "Windows":
            ok = self._run_all([
                ["powercfg", "/setacvalueindex", "SCHEME_CURRENT", "SUB_PROCESSOR",
                 "PROCTHROTTLEMAX", str(self.percent)],
                ["powercfg", "/setdcvalueindex", "SCHEME_CURRENT", "SUB_PROCESSOR",
                 "PROCTHROTTLEMAX", str(self.percent)],
                ["powercfg", "/setactive", "SCHEME_CURRENT"],
            ])
            self.restore_cmd = ("powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR "
                                "PROCTHROTTLEMAX 100 && powercfg /setdcvalueindex "
                                "SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100 && "
                                "powercfg /setactive SCHEME_CURRENT")
            self.detail = f"プロセッサ最大状態 {self.percent}%"
        else:
            ok = self._run_all([["cpupower", "frequency-set", "-u", f"{self.mhz}MHz"]])
            self.restore_cmd = "sudo cpupower frequency-set -u $(cpupower frequency-info -l | tail -1 | awk '{print $2}')"
            self.detail = f"CPU上限 {self.mhz} MHz"

        self.applied = ok
        if not ok:
            self.detail += "（適用に失敗）"
        return ok

    def restore(self) -> None:
        if not self.applied:
            return
        self.applied = False        # 二重復元を防ぐ（atexit と finally の両方から呼ばれる）
        print("\n[clock] クロック設定を元に戻します...")
        if self.system == "Windows":
            ok = self._run_all([
                ["powercfg", "/setacvalueindex", "SCHEME_CURRENT", "SUB_PROCESSOR",
                 "PROCTHROTTLEMAX", "100"],
                ["powercfg", "/setdcvalueindex", "SCHEME_CURRENT", "SUB_PROCESSOR",
                 "PROCTHROTTLEMAX", "100"],
                ["powercfg", "/setactive", "SCHEME_CURRENT"],
            ])
        else:
            mhz = self._max_mhz()
            ok = self._run_all([["cpupower", "frequency-set", "-u", f"{mhz}MHz"]]) if mhz else False
        if ok:
            print("[clock] 復元しました")
        else:
            print(f"[clock] !!! 復元に失敗しました。手で戻してください:\n  {self.restore_cmd}")

    @staticmethod
    def _max_mhz() -> int:
        try:
            import psutil
            f = psutil.cpu_freq()
            return int(f.max) if f and f.max else 0
        except Exception:
            return 0

    @staticmethod
    def _run_all(cmds: list) -> bool:
        for c in cmds:
            try:
                r = subprocess.run(c, capture_output=True, text=True, timeout=20)
                if r.returncode != 0:
                    print(f"[clock] コマンド失敗: {' '.join(c)}\n  {r.stderr.strip()}")
                    return False
            except (OSError, subprocess.SubprocessError) as e:
                print(f"[clock] コマンド実行不可: {' '.join(c)} - {type(e).__name__}: {e}")
                return False
        return True


# ==========================================================
# main
# ==========================================================
def main() -> None:
    args = build_parser().parse_args()
    cfg = resolve(args)

    # --- ここが重要: cv2/torch を import する「前」に環境変数を置く ---
    #     OpenMP のスレッドプールは共有ライブラリのロード時に確定するため、
    #     import 後に setNumThreads しても OMP プール自体のサイズは変えられない。
    if cfg["omp_threads"] > 0:
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[k] = str(cfg["omp_threads"])
    if cfg["cv_threads"] >= 0:
        # OpenCV は起動時にこの環境変数を見て既定スレッド数を決める
        os.environ["OPENCV_FOR_THREADS_NUM"] = str(max(1, cfg["cv_threads"]))

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import jetson_probe_common as common          # noqa: E402  ここで初めて cv2 を読む
    import jetson_live_probe                      # noqa: E402

    common.ensure_repo_cwd()
    odir = common.out_dir(base=args.out_dir)

    clock = ClockLimiter(cfg["clock_percent"], cfg["clock_mhz"])

    print("=" * 78)
    print(f"Jetson 相当実行  preset={args.preset}  tag={cfg['tag']}")
    print("=" * 78)
    print(f"  コア数        : {cfg['cores'] or '制限なし'}")
    print(f"  cv2 スレッド  : {cfg['cv_threads'] if cfg['cv_threads'] >= 0 else '既定のまま'}")
    print(f"  torch スレッド: {cfg['torch_threads'] if cfg['torch_threads'] >= 0 else '既定のまま'}")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', '未設定')}")
    print(f"  クロック      : {'要求 ' + (str(cfg['clock_percent']) + '%' if clock.system == 'Windows' else str(cfg['clock_mhz']) + 'MHz') if clock.wanted() else '制限なし'}")
    print(f"  Task 1 計測   : {'無効' if args.no_profile_cpu else f'{args.cpu_interval}秒ごと'}")
    print(f"  Task 3 計測   : {'有効' if args.verify_scale else '無効'}")
    print(f"  出力先        : {odir}")

    if args.dry_run:
        print("\n--dry-run のため main.py は起動しません。")
        return

    # クロック制限は復元を最優先で確保してから掛ける
    atexit.register(clock.restore)

    def _on_signal(signum, _frame):
        print(f"\n[run] シグナル {signum} を受け取りました。片付けます。")
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if sig is not None:
            try:
                signal.signal(sig, _on_signal)
            except (ValueError, OSError):
                pass       # メインスレッド以外・非対応プラットフォーム

    clock.apply()
    print(f"\n  クロック状態  : {clock.detail}")
    if clock.applied:
        print(f"  !!! 強制終了した場合は手で戻すこと:\n      {clock.restore_cmd}")

    # affinity と各スレッド数（module_yolo の import 順もここで面倒を見る）
    applied = common.apply_cpu_limits(cfg["cores"], cfg["cv_threads"],
                                      cfg["torch_threads"])
    print(f"  適用結果      : {applied}")

    probes = jetson_live_probe.LiveProbeSet(
        out_dir=odir, tag=cfg["tag"],
        profile_cpu=not args.no_profile_cpu,
        verify=args.verify_scale,
        cpu_interval=args.cpu_interval,
        verify_window=args.verify_window,
        verify_period=args.verify_period,
    )

    result = None
    try:
        probes.start()
        entry = args.main if os.path.isabs(args.main) else \
            os.path.join(common.repo_root(), args.main)
        if not os.path.exists(entry):
            raise SystemExit(f"エントリポイントが見つかりません: {entry}")
        print(f"\n{os.path.basename(entry)} を起動します。"
              "計測を終えるにはGUIを閉じてください。\n")
        import runpy
        runpy.run_path(entry, run_name="__main__")
    except KeyboardInterrupt:
        print("\n[run] 中断されました。ここまでの計測をまとめます。")
    except SystemExit:
        # main.py は sys.exit(app.exec()) で終わる。正常終了として扱う
        print("\n[run] main.py が終了しました。")
    finally:
        result = probes.stop_and_report({
            "run_config": cfg,
            "applied_limits": applied,
            "clock": {"wanted": clock.wanted(), "applied": clock.applied,
                      "detail": clock.detail},
        })
        clock.restore()

    if result:
        print("\n次の手順:")
        print("  1個体あたりの実コスト（本番が logs_cycle_5goki へ書いたもの）を集計する")
        print(f"    uv run python standalone/analyze_cycle_logs.py "
              f"--run {os.path.join(odir, cfg['tag'] + '_live_summary.json')}")
        print("  基準と制限の2本が揃ったら k を出す")
        print(f"    uv run python standalone/analyze_cycle_logs.py "
              f"--ref {os.path.join(odir, 'baseline_live_summary.json')} "
              f"--proxy {os.path.join(odir, 'jetson_live_summary.json')}")


if __name__ == "__main__":
    main()
