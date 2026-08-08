"""本番（main.py）を指定の縮小率・制約下で起動し、実運転のログと実測プローブを同時に取る。

3本セットで使う想定（詳細は JETSON_VALIDATION_RUNBOOK.md）:

    # ①半解像度なし・PC制限なし（基準）
    uv run python standalone/run_as_jetson.py --scale 1.0 --preset baseline --tag s10_baseline

    # ②半解像度あり・PC制限なし
    uv run python standalone/run_as_jetson.py --scale 0.5 --preset baseline --tag s05_baseline

    # ③半解像度あり・Jetson相当の制限あり
    uv run python standalone/run_as_jetson.py --scale 0.5 --preset jetson  --tag s05_jetson

GUIを閉じる（またはCtrl-C）と計測を締めてレポートを書き、クロック設定を元に戻す。

──────────────────────────────────────────────────────────────
何を「Jetson相当」にできて、何ができないか
──────────────────────────────────────────────────────────────
できる:
  半解像度    module_yolo.HSV_DETECT_SCALE を起動前に書き換える（--scale）
  コア数      CPU affinity で6コアへ制限（Orin Nano Super は A78AE 6コア）
  クロック    Windows は powercfg で上限を下げる。実際に落ちたかは
              `% Processor Performance` カウンタを実行中ずっと直接実測して記録する
              （自己申告の clock.applied だけでは足りないことが分かっているため）
  スレッド数  cv2.setNumThreads(1) / torch.set_num_threads(2) / OMP_NUM_THREADS
              （OpenMPプールは import 時に確定するため、本スクリプトは
                cv2/torch を読む前に環境変数を設定する）

できない（数値の解釈時に必ず考慮すること）:
  ISA        AVX2(256bit) を NEON(128bit) にはできない。project_jetson.py の
             --isa-factor（既定1.4）で安全側に補正する
  GPU        RTX 4090 Laptop を Orin の iGPU にはできない。GPU側は TensorRT の
             公表実測値を使って project_jetson.py で合成する
  メモリ     8GB共有メモリの再現はしない。RSS を記録し、8GB に対する余裕を報告する
  熱         サーマルスロットリングは再現不可。実機で要確認

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
        description="main.py を指定の縮小率・制約下で起動し、実運転のログと実測プローブを取る",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale", type=float, required=True,
                   help="module_yolo.HSV_DETECT_SCALE に設定する値（1.0=半解像度なし / 0.5=半解像度あり）")
    p.add_argument("--preset", choices=["baseline", "jetson"], default="baseline",
                   help="baseline=PC制限なし / jetson=6コア+クロック制限（Jetson相当）")
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
    p.add_argument("--no-clock", action="store_true", help="クロック制限を行わない")

    p.add_argument("--no-profile-cpu", action="store_true", help="CPU実効並列度の計測を行わない")
    p.add_argument("--no-profile-clock", action="store_true", help="クロックの実測を行わない")
    p.add_argument("--cpu-interval", type=float, default=1.0, help="CPU計測のサンプリング間隔[秒]")
    p.add_argument("--clock-interval", type=float, default=2.0, help="クロック実測のサンプリング間隔[秒]")

    p.add_argument("--main", default="main.py", help="起動するエントリポイント（既定 main.py）")
    p.add_argument("--out-dir", default=None, help="出力先（既定 analysis/jetson_validation）")
    p.add_argument("--tag", required=True, help="出力ファイル名の接頭辞（例: s10_baseline）")
    p.add_argument("--dry-run", action="store_true",
                   help="制限内容を表示するだけで main.py を起動しない")
    return p


def resolve(args) -> dict:
    """preset の既定値と個別指定をマージする。"""
    if args.preset == "jetson":
        cfg = {"cores": 6, "cv_threads": 1, "torch_threads": 2, "omp_threads": 2,
               "clock_percent": 31}
    else:
        cfg = {"cores": 0, "cv_threads": -1, "torch_threads": -1, "omp_threads": 0,
               "clock_percent": 0}

    for key, val in (("cores", args.cores), ("cv_threads", args.cv_threads),
                     ("torch_threads", args.torch_threads),
                     ("omp_threads", args.omp_threads),
                     ("clock_percent", args.clock_percent)):
        if val is not None:
            cfg[key] = val
    if args.no_clock:
        cfg["clock_percent"] = 0
    cfg["scale"] = args.scale
    cfg["tag"] = args.tag
    return cfg


# ==========================================================
# CPUクロックの制限と復元
# ==========================================================
class ClockLimiter:
    """OSの電源設定でCPUクロック上限を下げ、終了時に必ず戻す。

    戻し忘れると以降の学習・計測が全部遅くなるため、正常終了/例外/シグナルの
    どの経路でも restore() が呼ばれるようにしてある（強制終了は除く）。"""

    def __init__(self, percent: int = 0):
        self.percent = percent
        self.applied = False
        self.system = platform.system()
        self.restore_cmd = ""
        self.detail = ""

    def _has_admin(self) -> bool:
        if self.system == "Windows":
            try:
                import ctypes
                return bool(ctypes.windll.shell32.IsUserAnAdmin())
            except Exception:
                return False
        return hasattr(os, "geteuid") and os.geteuid() == 0

    def wanted(self) -> bool:
        return self.system == "Windows" and self.percent > 0

    def apply(self) -> bool:
        if not self.wanted():
            self.detail = "クロック制限なし" if self.system == "Windows" else "Windows専用のため未適用"
            return False
        if not self._has_admin():
            self.detail = ("管理者権限が無いためクロック制限を適用できない。"
                           "管理者PowerShellで実行し直すか、--no-clock で明示的に外すこと")
            return False

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
        self.detail = f"要求 プロセッサ最大状態 {self.percent}%（実際に落ちたクロックは実測値を見ること）"
        self.applied = ok
        if not ok:
            self.detail += "（適用に失敗）"
        return ok

    def restore(self) -> None:
        if not self.applied:
            return
        self.applied = False        # 二重復元を防ぐ（atexit と finally の両方から呼ばれる）
        print("\n[clock] クロック設定を元に戻します...")
        ok = self._run_all([
            ["powercfg", "/setacvalueindex", "SCHEME_CURRENT", "SUB_PROCESSOR",
             "PROCTHROTTLEMAX", "100"],
            ["powercfg", "/setdcvalueindex", "SCHEME_CURRENT", "SUB_PROCESSOR",
             "PROCTHROTTLEMAX", "100"],
            ["powercfg", "/setactive", "SCHEME_CURRENT"],
        ])
        if ok:
            print("[clock] 復元しました")
        else:
            print(f"[clock] !!! 復元に失敗しました。手で戻してください:\n  {self.restore_cmd}")

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
        os.environ["OPENCV_FOR_THREADS_NUM"] = str(max(1, cfg["cv_threads"]))

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import jetson_probe_common as common          # noqa: E402  ここで初めて cv2 を読む
    import jetson_live_probe                      # noqa: E402

    common.ensure_repo_cwd()
    odir = common.out_dir(base=args.out_dir)

    clock = ClockLimiter(cfg["clock_percent"])

    print("=" * 78)
    print(f"実行  scale={cfg['scale']}  preset={args.preset}  tag={cfg['tag']}")
    print("=" * 78)
    print(f"  縮小率(HSV_DETECT_SCALE): {cfg['scale']}")
    print(f"  コア数        : {cfg['cores'] or '制限なし'}")
    print(f"  cv2 スレッド  : {cfg['cv_threads'] if cfg['cv_threads'] >= 0 else '既定のまま'}")
    print(f"  torch スレッド: {cfg['torch_threads'] if cfg['torch_threads'] >= 0 else '既定のまま'}")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', '未設定')}")
    print(f"  クロック      : {'要求 ' + str(cfg['clock_percent']) + '%' if clock.wanted() else '制限なし'}")
    print(f"  CPU計測       : {'無効' if args.no_profile_cpu else f'{args.cpu_interval}秒ごと'}")
    print(f"  クロック実測  : {'無効' if args.no_profile_clock else f'{args.clock_interval}秒ごと'}")
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
    applied = common.apply_cpu_limits(cfg["cores"], cfg["cv_threads"], cfg["torch_threads"])
    print(f"  適用結果      : {applied}")

    # 縮小率を起動前に固定する（main.py が import する module_yolo はここで既にキャッシュ済み）
    my = common.load_module_yolo()
    my.HSV_DETECT_SCALE = float(cfg["scale"])
    print(f"  HSV_DETECT_SCALE を {cfg['scale']} に設定しました")

    probes = jetson_live_probe.LiveProbeSet(
        out_dir=odir, tag=cfg["tag"],
        profile_cpu=not args.no_profile_cpu,
        profile_clock=not args.no_profile_clock,
        cpu_interval=args.cpu_interval,
        clock_interval=args.clock_interval,
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
        print("  3本揃ったら、2つの比較を行う（RUNBOOK参照）")
        print(f"    uv run python standalone/analyze_cycle_logs.py "
              f"--ref {os.path.join(odir, 's10_baseline_live_summary.json')} "
              f"--proxy {os.path.join(odir, 's05_baseline_live_summary.json')} "
              f"--tag compare_scale_effect")
        print(f"    uv run python standalone/analyze_cycle_logs.py "
              f"--ref {os.path.join(odir, 's05_baseline_live_summary.json')} "
              f"--proxy {os.path.join(odir, 's05_jetson_live_summary.json')} "
              f"--tag compare_jetson_effect")


if __name__ == "__main__":
    main()
