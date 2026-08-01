# -------------------------------------------------
# calibrate_delay.py
#   カメラ表示遅延（delay_seconds）の実測キャリブレーションツール。（Arduino版）
#
#   原理:
#     4カメラを「遅延0（生ストリーム）」で起動し、識別しやすい1個を流す。
#     各カメラが果実を「中心に捉えた瞬間（HSV赤ブロブの重心xが画面中央に最も近い時刻）」を記録し、
#     基準カメラ（遅延0側＝cam_top / cam_outside）に対する cam_under / cam_inside の
#     時間差を求める。その時間差がそのまま設定すべき delay_seconds。
#
#   使い方（運用速度に合わせて流すこと。遅延は速度依存のため）:
#     uv run python calibrate_delay.py --rounds 5 --duration 8 --speed 5
#       --rounds    : 何個流して平均（中央値）するか（既定5）
#       --duration  : 1個あたりの計測ウィンドウ秒数（既定8）
#       --speed     : 運用スピード(1-10)。回転速度に使用し、main の計算式との比較も表示
#                     （未指定時は回転速度に既定値5を使用）
#       --no-rotate : ターンテーブルの回転を始動しない（手動で果実を流す場合）
#
#   ※ main.py（Arduino版）と同じ手順でターンテーブルの回転を自動で始動/停止する
#     （起動時に 速度設定 → 回転開始、終了時に 停止）。運用と同じ流速で計測できる。
#     モーター通信は module_motor_serial 経由のArduino USBシリアル。
#     Arduinoが非常停止（ESTOP）を送ってきた場合は検知し、当該ラウンドを無効化する。
#
#   出力:
#     - json/delay_config.json に推奨遅延値を自動保存する。
#       main.py の update_camera_delays() がこのファイルを自動読込する。
#     - calibration_result_*.json に生データ（詳細）を別途保存する。
#     - HSV判定は hsv_config_{cam}.json を直接読む（カメラ別個別設定）。
# -------------------------------------------------
import sys
import os
import time
import json
import argparse
import statistics
import datetime
import threading

import cv2
import numpy as np

# このスクリプトはプロジェクト直下のモジュールを参照するため、親ディレクトリを import パスに追加
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# main.py（Arduino版）と同じカメラ/モーターモジュールを使う
import module_cameras_5goki as cam_ctr
import module_motor_serial as motor_ctr

# ==========================================================
# アーキテクチャ上の役割（main の update_camera_delays と一致させる）
# ==========================================================
REFERENCE_CAMS = ["cam_top", "cam_outside"]   # 遅延0（基準）
DELAYED_CAMS = ["cam_under", "cam_inside"]     # 遅延を掛けるカメラ（＝計測対象）
ALL_CAMS = REFERENCE_CAMS + DELAYED_CAMS

# Arduinoのシリアルポート。None で自動検出。うまくいかない場合は "COM3" 等を直接指定する
SERIAL_PORT = None
# 回転速度の既定（--speed 未指定時。main のデフォルトと揃える）
DEFAULT_ROTATE_SPEED = 6

# 中心通過の判定に使う最小ブロブ面積（小さすぎる検出＝ノイズ/画外を無視）
MIN_AREA = 10000

# main の update_camera_delays と同じ定数（--speed 指定時の計算式比較に使用）
SPEED_MAP = {
    1: 0.0010, 2: 0.0009, 3: 0.0008, 4: 0.0007,
    5: 0.0006, 6: 0.0005, 7: 0.0004, 8: 0.0003, 9: 0.0002, 10: 0.0001
}
RATIO = 1.0
MICRO_STATUS = 16   # TB6600=3200 pulse/rev ÷ 200 step/rev（module_relay と合わせる）

# main.py が現在採用している固定遅延値（実測との突き合わせ用）
CURRENT_MAIN_DELAYS = {"cam_under": 1.922, "cam_inside": 2.015}


# ==========================================================
# Arduinoの非常停止（ESTOP）監視
#   MotorSerial が受信スレッドから呼ぶコールバックでフラグを立てる。
#   計測ループ側でこのフラグを見て、当該ラウンドを無効化する。
# ==========================================================
class EstopWatcher:
    def __init__(self):
        self._event = threading.Event()

    # MotorSerial(on_estop=...) から別スレッドで呼ばれる
    def on_estop(self):
        self._event.set()
        print("\n!! 非常停止(ESTOP)を検知しました。現在のラウンドは無効化されます。")

    def on_estop_cleared(self):
        self._event.clear()
        print(">> 非常停止が解除されました。")

    def triggered(self):
        return self._event.is_set()

    def reset(self):
        self._event.clear()


# ==========================================================
# HSV赤ブロブ検出（module_yolo_csv*.ImageProcessor.get_target_info と同等。
#   ツールを軽量・独立に保つため最小版を内包。設定変更時は両者を合わせること）
# ==========================================================
_HSV_DEFAULT = (np.array([0, 60, 50]), np.array([35, 255, 255]),
                np.array([160, 60, 50]), np.array([180, 255, 255]))


def _load_single_hsv(path):
    """JSONファイルから (lower1, upper1, lower2, upper2) を読み込む。失敗時は None。"""
    try:
        with open(path, "r") as f:
            cfg = json.load(f)
        return (np.array(cfg["lower1"]), np.array(cfg["upper1"]),
                np.array(cfg["lower2"]), np.array(cfg["upper2"]))
    except Exception as e:
        print(f"!! {path} 読み込み失敗: {e}")
        return None


def load_hsv_ranges_per_cam():
    """カメラ別の hsv_config_{cam_name}.json を読み込み {cam_name: ranges} を返す。
    カメラ別ファイルが無い場合は hsv_common_config.json、それも無ければ既定値を使う。"""
    json_dir = os.path.join(_ROOT_DIR, "json")
    common_path = os.path.join(json_dir, "hsv_common_config.json")
    common_ranges = _load_single_hsv(common_path) if os.path.exists(common_path) else None
    if common_ranges is None:
        common_ranges = _HSV_DEFAULT

    ranges_per_cam = {}
    for cam in ALL_CAMS:
        per_cam_path = os.path.join(json_dir, f"hsv_config_{cam}.json")
        if os.path.exists(per_cam_path):
            r = _load_single_hsv(per_cam_path)
            if r is not None:
                ranges_per_cam[cam] = r
                print(f"  HSV: {cam} → hsv_config_{cam}.json")
                continue
        ranges_per_cam[cam] = common_ranges
        print(f"  HSV: {cam} → 共通設定（hsv_common_config.json / 既定値）")
    return ranges_per_cam


def find_center_metric(frame, ranges):
    """フレーム内の最大赤ブロブを検出し (center_dist, area) を返す。
    見つからなければ None。center_dist は重心xと画面中央xの距離（小さいほど中心）。"""
    lower1, upper1, lower2, upper2 = ranges
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2),
                            cv2.MORPH_CLOSE, kernel, iterations=2)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return None
    max_index = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    area = int(stats[max_index, cv2.CC_STAT_AREA])
    if area < MIN_AREA:
        return None
    w = frame.shape[1]
    mx = int(centroids[max_index][0])
    return abs(mx - w // 2), area


# ==========================================================
# 1回分（1個分）の計測
# ==========================================================
def run_round(controllers, ranges_per_cam, duration, estop=None):
    """duration 秒のあいだ各カメラを監視し、カメラごとの「中心通過時刻」を返す。
    返り値: {cam_name: crossing_monotonic_time or None}
    計測中にArduinoの非常停止を検知した場合は例外 RuntimeError を送出する。"""
    # フレーム重複処理を避けるため、最後に処理したフレームの識別子を保持
    last_obj = {c.name: None for c in controllers}
    # 各カメラの最良サンプル: (min_center_dist, crossing_time)
    best = {c.name: None for c in controllers}

    t_start = time.monotonic()
    next_tick = t_start + 1.0
    while True:
        now = time.monotonic()
        if now - t_start >= duration:
            break

        # 非常停止を検知したら計測を打ち切る（不正なデータを残さない）
        if estop is not None and estop.triggered():
            raise RuntimeError("非常停止(ESTOP)により計測を中断しました")

        for c in controllers:
            frame = c.get_current_frame()
            if frame is None:
                continue
            # 同一フレームの二重処理を防ぐ（latest_frame は grab ごとに新オブジェクト）
            if frame is last_obj[c.name]:
                continue
            last_obj[c.name] = frame

            metric = find_center_metric(frame, ranges_per_cam[c.name])
            if metric is None:
                continue
            center_dist, _area = metric
            cur = best[c.name]
            if cur is None or center_dist < cur[0]:
                best[c.name] = (center_dist, now)

        # 進捗表示（1秒ごと）
        if now >= next_tick:
            remaining = int(duration - (now - t_start)) + 1
            seen = [c.name for c in controllers if best[c.name] is not None]
            print(f"  計測中... 残り{remaining:>2}s  検出済み: {seen if seen else 'なし'}")
            next_tick += 1.0

        time.sleep(0.005)  # CPUを使い切らないための小休止（精度±数ms）

    return {name: (v[1] if v is not None else None) for name, v in best.items()}


# ==========================================================
# 集約と推奨遅延の算出
# ==========================================================
def summarize(rounds_times):
    """各ラウンドの {cam: crossing_time} リストから、推奨 delay_seconds を算出する。

    各ラウンドで:
      target = 基準カメラ(REFERENCE_CAMS)の中心通過時刻の平均
      offset[cam] = target - t[cam]   （正なら「targetより早く見えた」＝その分遅延すべき量）
    をカメラ別に集め、全ラウンドの中央値を推奨値とする。
    """
    per_cam_offsets = {cam: [] for cam in ALL_CAMS}

    for rt in rounds_times:
        ref_times = [rt.get(cam) for cam in REFERENCE_CAMS if rt.get(cam) is not None]
        if not ref_times:
            print("!! このラウンドは基準カメラが未検出のためスキップ")
            continue
        target = sum(ref_times) / len(ref_times)
        for cam in ALL_CAMS:
            t = rt.get(cam)
            if t is not None:
                per_cam_offsets[cam].append(target - t)

    summary = {}
    for cam in ALL_CAMS:
        vals = per_cam_offsets[cam]
        if vals:
            summary[cam] = {
                "median": statistics.median(vals),
                "mean": statistics.fmean(vals),
                "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
        else:
            summary[cam] = None
    return summary


def print_report(summary, speed):
    print("\n" + "=" * 60)
    print("  キャリブレーション結果")
    print("=" * 60)
    print(f"{'カメラ':<14}{'役割':<8}{'中央値[s]':>10}{'平均[s]':>10}{'ばらつき':>10}{'試行':>6}")
    for cam in ALL_CAMS:
        role = "基準" if cam in REFERENCE_CAMS else "遅延対象"
        s = summary[cam]
        if s is None:
            print(f"{cam:<14}{role:<8}{'未検出':>10}")
        else:
            print(f"{cam:<14}{role:<8}{s['median']:>10.3f}{s['mean']:>10.3f}{s['stdev']:>10.3f}{s['n']:>6}")

    print("\n--- 推奨 delay_seconds（main の update_camera_delays に反映）---")
    recommended = {}
    for cam in DELAYED_CAMS:
        s = summary[cam]
        if s is None:
            print(f"  {cam}: 未検出のため算出不可")
            continue
        val = s["median"]
        recommended[cam] = val
        if val < -0.01:
            print(f"  {cam}: {val:+.3f}s  !! 負値: このカメラは基準より「後」に見えています。")
            print("        現アーキテクチャ(正の遅延のみ)では合わせられません。")
            print(f"        → 基準カメラ側を {cam} に変える等の見直しが必要です。")
        else:
            print(f"  {cam}: {max(0.0, val):.3f} 秒")

    # --- main が現在使っている固定遅延値との比較 ---
    print("\n--- main.py の現行固定値との比較 ---")
    for cam in DELAYED_CAMS:
        cur = CURRENT_MAIN_DELAYS.get(cam)
        s = summary[cam]
        if cur is None:
            continue
        if s is None:
            print(f"  {cam}: 現行 {cur:.3f}s / 実測 未検出")
            continue
        meas = max(0.0, s["median"])
        diff = meas - cur
        hint = "（ほぼ一致）" if abs(diff) <= 0.05 else "（要更新の可能性）"
        print(f"  {cam}: 現行 {cur:.3f}s / 実測 {meas:.3f}s  差 {diff:+.3f}s {hint}")

    # --- 計算式との比較（--speed 指定時）---
    if speed is not None and speed in SPEED_MAP:
        delay = SPEED_MAP[speed]
        t_one_pulse = delay * 2
        step_one_rotation = RATIO * (360 / 1.8) * MICRO_STATUS
        sec = t_one_pulse * step_one_rotation * 2
        formula_delay = sec * (60 / 360)
        print(f"\n--- 現行計算式との比較（speed={speed}）---")
        print(f"  計算式が出す遅延: {formula_delay:.3f} 秒（cam_under/cam_inside 共通）")
        valid = [recommended[c] for c in DELAYED_CAMS if c in recommended and recommended[c] > 0]
        if valid and formula_delay > 0:
            meas = statistics.median(valid)
            suggested_ratio = RATIO * meas / formula_delay
            print(f"  実測（遅延対象の中央値）: {meas:.3f} 秒")
            print(f"  → 計算式を実測に合わせる補正係数 RATIO 約 {suggested_ratio:.3f}")
            print(f"     （現在 RATIO={RATIO}。main の RATIO をこの値にすると計算式が実測へ近づく）")
        if len(valid) == 2 and abs(recommended[DELAYED_CAMS[0]] - recommended[DELAYED_CAMS[1]]) > 0.05:
            print("  ※ 2台の遅延差が大きいため、共通値1つの計算式では不十分。")
            print("     カメラ別に delay_seconds を個別設定することを推奨。")

    return recommended


def save_delay_config(recommended):
    """推奨遅延値を json/delay_config.json に保存する。
    main.py の update_camera_delays() がこのファイルを読み込む。"""
    path = os.path.join(_ROOT_DIR, "json", "delay_config.json")
    data = {cam: round(max(0.0, v), 4) for cam, v in recommended.items()}
    data["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n遅延設定を保存しました: {path}")
        for cam, v in recommended.items():
            print(f"  {cam}: {max(0.0, v):.4f} 秒")
    except Exception as e:
        print(f"!! 遅延設定の保存に失敗: {e}")


def save_result(summary, recommended, rounds_times, speed):
    out = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "speed": speed,
        "summary": {cam: summary[cam] for cam in ALL_CAMS},
        "recommended_delay_seconds": recommended,
        "rounds_raw": [
            {cam: (round(rt[cam], 4) if rt.get(cam) is not None else None) for cam in ALL_CAMS}
            for rt in rounds_times
        ],
    }
    path = f"calibration_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"詳細ログを保存しました: {path}")
    except Exception as e:
        print(f"!! 詳細ログの保存に失敗: {e}")


# ==========================================================
# メイン
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description="カメラ遅延の実測キャリブレーション（Arduino版）")
    parser.add_argument("--rounds", type=int, default=5, help="計測する個数（既定5）")
    parser.add_argument("--duration", type=float, default=8.0, help="1個あたりの計測秒数（既定8）")
    parser.add_argument("--speed", type=int, default=None, help="運用スピード1-10（計算式比較用・任意）")
    parser.add_argument("--no-rotate", action="store_true",
                        help="ターンテーブルの回転を始動しない（手動で果実を流す場合）")
    args = parser.parse_args()

    print("HSV設定を読み込み中...")
    ranges_per_cam = load_hsv_ranges_per_cam()

    manager = cam_ctr.CameraManager()
    print("カメラを初期化中...")
    if not manager.init_cameras():
        print("!! カメラの初期化に失敗しました。接続を確認してください。")
        return 1

    controllers = manager.controllers
    found_names = [c.name for c in controllers]
    print(f"接続カメラ: {found_names}")
    missing = set(ALL_CAMS) - set(found_names)
    if missing:
        print(f"!! 未接続のカメラがあります: {sorted(missing)}（その分の計測は欠落します）")

    # 全カメラを遅延0（生ストリーム）に固定
    for c in controllers:
        c.delay_seconds = 0.0
    manager.start_all_get_frame()
    time.sleep(1.0)  # ストリーム安定待ち

    # --- ターンテーブルの回転を main と同じ手順で始動（速度設定 → 回転開始）---
    #   Arduinoの非常停止通知を受け取れるよう EstopWatcher を渡す
    estop = EstopWatcher()
    motor = None
    rotating = False
    if not args.no_rotate:
        rotate_speed = args.speed if args.speed is not None else DEFAULT_ROTATE_SPEED
        print(f"\nターンテーブルの回転を始動します（speed={rotate_speed}）...")
        motor = motor_ctr.MotorSerial(
            port=SERIAL_PORT,
            on_estop=estop.on_estop,
            on_estop_cleared=estop.on_estop_cleared,
        )
        if motor.init():
            motor.set_speed(rotate_speed)
            rotating = motor.rotate()
        if rotating:
            print("回転を開始しました。運用と同じ流速で果実を流してください。")
        else:
            print("!! 回転の始動に失敗しました。Arduino接続を確認するか、手動で流してください。")
    else:
        print("\n--no-rotate 指定: ターンテーブルの回転は始動しません（手動投入）。")

    rounds_times = []
    try:
        for r in range(1, args.rounds + 1):
            input(f"\n[ラウンド {r}/{args.rounds}] 識別しやすい1個を投入できたら Enter で計測開始 → ")
            if estop.triggered():
                print("!! 非常停止中です。解除してから計測してください。このラウンドはスキップします。")
                continue
            print(f"計測開始（{args.duration:.0f}秒）。果実を中央付近を通過させてください。")
            try:
                rt = run_round(controllers, ranges_per_cam, args.duration, estop=estop)
            except RuntimeError as e:
                print(f"!! {e}。このラウンドは無効化しました。")
                continue
            rel = {}
            base = min([t for t in rt.values() if t is not None], default=None)
            for cam in ALL_CAMS:
                t = rt.get(cam)
                rel[cam] = (t - base) if (t is not None and base is not None) else None
            print("  → このラウンドの中心通過（最早を0とした相対秒）:")
            for cam in ALL_CAMS:
                v = rel[cam]
                print(f"     {cam:<12}: {'未検出' if v is None else f'{v:+.3f}s'}")
            rounds_times.append(rt)
    except KeyboardInterrupt:
        print("\n中断されました。ここまでの計測で集約します。")
    finally:
        # main と同じく、終了時は必ず回転を停止する
        if rotating:
            print("ターンテーブルの回転を停止します...")
            motor.stop()
        if motor is not None:
            motor.close()
        manager.stop_all_get_frame()

    if not rounds_times:
        print("計測データがありません。終了します。")
        return 1

    summary = summarize(rounds_times)
    recommended = print_report(summary, args.speed)
    save_delay_config(recommended)
    save_result(summary, recommended, rounds_times, args.speed)
    print("\n完了。delay_config.json を自動保存しました。次回の main 起動時から自動反映されます。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
