# dcr_logger.py  —  構造化データログ（cycle / health / events / detections）
#   標準ライブラリのみで動作する。リアルタイムループ（撮影→推論→排出）を
#   絶対にブロックしないため、書き込みは専用スレッドに逃がし、呼び出し側は
#   enqueue のみ（O(1)・例外を投げない）。
#
#   コンソール出力との統合（System A + B）:
#     events（離散イベント・エラー）だけは event() の中で「即時に」Python の
#     logging へも流し、コンソールに整形表示する。cycle/health/detections は
#     データ専用で、コンソールには出さない（ファイルのみ）。
#     ※ logging のハンドラ設定は log_config.setup_logging() 側で行う。
from __future__ import annotations
import csv, json, os, queue, threading, time, logging
from datetime import datetime
from pathlib import Path

# events をコンソールへ流すためのロガー（ハンドラは log_config 側で root に付ける）
_event_logger = logging.getLogger("event")
_SEV_TO_LEVEL = {"INFO": logging.INFO, "WARN": logging.WARNING, "ERROR": logging.ERROR}

# スキーマ版（旧 1.0.0 のファイルとは混在しても schema_version 列で判別できる）。
#   全 CSV 行・events に自動付与する。
SCHEMA_VERSION = "1.0.0"

CYCLE_COLUMNS = [
    "schema_version", "timestamp", "cycle_id",
    "capture_latency[ms]", "frame_dropped[n]",                 #撮影・取得
    "hsv_flag", "hsv_mask_ratio",                           #HSVフィルタ結果
    "infer_latency[ms]", "preproc[ms]", "postproc[ms]",     #推論前後の時間
    # --- 1個体あたりの処理コスト内訳（4カメラ・全フレームの「合計」）---
    #   上の infer_latency / preproc / postproc は「1枚あたりの平均」であり、
    #   1個の処理に要した総時間ではない。目標「検出→前処理→推論が4カメラ合計で
    #   1秒/個未満」を判定できるよう、ここに合計値と枚数を別列で残す。
    #   ※ capture_latency[ms] は RetrieveResult の待ち時間（= 1/FPS ≒ 50ms）であって
    #     計算時間ではない。撮影段の実処理コストは capture_sum[ms] を見ること。
    #   集計区間は「前個体の確定 → この個体の確定」（= cycle_dur[s]）。
    "capture_sum[ms]", "capture_frames[n]",                 #撮影(RGB8→BGR変換+複製)
    "hsv_sum[ms]", "hsv_frames[n]", "visible_frames[n]",    #HSVマスク検出（帯内外を問わず全フレーム）
    "preproc_sum[ms]",                                      #crop/resize（帯内のみ）
    "infer_sum[ms]", "infer_count[n]",                      #推論（帯内のみ）
    "postproc_sum[ms]",                                     #ByteTrack+描画（帯内のみ）
    "total_per_fruit[ms]",                                  #上記4段の総計 ← 目標指標
    "visible_dur[s]", "cycle_dur[s]",                       #可視継続時間・集計区間長
    "eject_flag",                                           #推論判定（推論直後）
    "planned_eject_ts", "eject_delay[ms]",                  #リレーAPI呼び出し時刻・ソフト遅延
    "outcome_flag",                                         #リレー実行結果（排出後）
    "yolo_no_det_flag",                                     #HSV通過・YOLO無検出フラグ（1=HSV有でYOLO未検出 / 0=正常検出）
    # --- HW環境スナップショット（health 1Hz の最新キャッシュ値・per-cycle 相関用）---
    #   cycle と health は時間軸が違うため、同じ指標でも両軸に残す（ダウングレード判断）。
    "cpu_temp","cpu_util[%]",
    "gpu_temp","gpu_util[%]", "gpu_clock[mhz]",
    "gpu_mem_used[mb]", "gpu_power[w]",
]
HEALTH_COLUMNS = [
    "schema_version", "timestamp",
    # --- CPU ---
    "cpu_temp", "cpu_util[%]",                         #CPU 温度・使用率
    # --- GPU（温度/性能/消費電力・ダウングレード判断も含む）---
    "gpu_temp", "gpu_util[%]", "gpu_clock[mhz]",        #GPU 温度・使用率・クロック
    "gpu_mem_used[mb]", "gpu_power[w]",                      #VRAM 使用量・消費電力
    # --- リソース ---
    "ram_used[mb]", "proc_rss[mb]", "disk_free[gb]",          #システムRAM・自プロセスRSS・ディスク空き
    "queue_depth[n]",                                          #フレームキューの深さ
    # --- VRAMリーク検出 ---
    "torch_vram_alloc[mb]", "torch_vram_reserved[mb]",       #torch VRAM 内訳（alloc=本物のリーク）
    "cycles_total[n]",                                        #その時点の累積サイクル数（MB/サイクル算出用）
    # --- ロガー自己監視 ---
    "dropped_logs[n]",                                        #ロガーがドロップしたログ数
]
DET_COLUMNS = [
    "schema_version", "timestamp", "cycle_id",
    "class", "conf_min", "conf_max", "conf_ave", "num_detections", "final_flag",
]

# ---------- events スキーマ定義 ----------
# events は JSONL 形式で自由フィールドを持てるが、type ごとの期待フィールドを以下に示す。
# 呼び出し側はこの定義に従って logger.info() / logger.warn() / logger.error() を呼ぶ。
#
# --- システムライフサイクル ---
# type="startup"            sev=INFO   model_name, precision, gpu_name, ...（任意）
# type="shutdown"           sev=INFO   uptime_s
#
# --- GUI 状態遷移 ---
# type="gui_state"          sev=INFO   state="run"|"stop"|"pause",  msg
#
# --- 非常停止 ---
# type="estop"              sev=ERROR  source="switch"|"software",  trigger
# type="estop_cleared"      sev=INFO
#
# --- Arduino ---
# type="arduino_connect"    sev=INFO   port, firmware_ver
# type="arduino_disconnect" sev=ERROR  reason
# type="arduino_params"     sev=INFO   pulse_width_ms, interval_ms, ...（startup 時・変更時）
# type="arduino_heartbeat"  sev=INFO   status="ok"|"timeout",  latency_ms（"ok" 時のみ）


class Telemetry:
    """health スレッドが更新し、cycle が読む最新値キャッシュ。
    ホットパスで nvml / pylon を直接叩かないための仕組み。"""
    def __init__(self):
        self._d, self._lock = {}, threading.Lock()

    def update(self, **kw):
        with self._lock:
            self._d.update({k: v for k, v in kw.items() if v is not None})

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._d)


class DCRLogger:
    def __init__(self, base_dir="logs", line="line",
                 queue_size=20000, flush_interval_s=1.0):
        self.base = Path(base_dir)  # フォルダ名プレフィックス（実フォルダ: logs_{kind}_5goki）
        self.line = line
        self.q: queue.Queue = queue.Queue(maxsize=queue_size)
        self.flush_interval_s = flush_interval_s
        self.dropped = 0
        self._stop = threading.Event()
        self._t = None
        self._files = {}      # kind -> file handle
        self._writers = {}    # kind -> csv.DictWriter
        self._open_date = {}  # kind -> session_id
        # ファイル名は日次（YYYYMMDD）。同日の複数セッションは同一ファイルへ追記し、
        # SESSION START/END コメント行でセッション境界を示す。
        _now = datetime.now()
        self._session = _now.strftime("%Y%m%d")
        self._session_start_dt = _now.strftime("%Y-%m-%d %H:%M:%S")
        self._start_mono = time.monotonic()

    # ---------- public (hot path safe) ----------
    def cycle(self, **fields):
        fields.setdefault("timestamp", self._now())
        self._put(("cycle", fields))

    def health(self, **fields):
        fields.setdefault("timestamp", self._now())
        self._put(("health", fields))

    def detections(self, cycle_id, items):
        ts = self._now()
        self._put(("det", (cycle_id, ts, items)))

    def event(self, type, sev="INFO", cycle_id=None, **fields):
        rec = {"schema_version": SCHEMA_VERSION, "ts": self._now(), "sev": sev, "type": type}
        if cycle_id is not None:
            rec["cycle_id"] = cycle_id
        rec.update(fields)
        # (A+B 統合) イベントは「即時に」コンソールへも流す。失敗してもホットパスは止めない。
        self._echo_event(rec, sev)
        self._put(("event", rec))

    def info(self, type, **f):  self.event(type, "INFO", **f)
    def warn(self, type, **f):  self.event(type, "WARN", **f)
    def error(self, type, **f): self.event(type, "ERROR", **f)

    # ---------- lifecycle ----------
    def start(self, **startup_fields):
        """書き込みスレッドを起動し、startup イベントを記録する。
        モデル名・精度・GPU 静的情報などは、用意できる呼び出し側が startup_fields で渡す。
        モデルロード後に呼びたい場合は log_startup=False でスレッドだけ先に起動し、
        後から log_startup() を呼ぶ。"""
        log_startup = startup_fields.pop("log_startup", True)
        self._t = threading.Thread(target=self._run, name="dcr-logger", daemon=True)
        self._t.start()
        if log_startup:
            self.log_startup(**startup_fields)

    def log_startup(self, **fields):
        """startup イベントを記録する。schema_version は event() が自動付与する。"""
        self.info("startup", **fields)

    def start_mem_snapshot(self, interval_s=60.0):
        """tracemalloc の定期スナップショットを別タイマー（既定60秒）でイベントへ残す。
        リーク確定後に「どの行が増やしているか」を特定する足がかり。
        health(≈1Hz) より遅い周期で回し、ホットパスには触れない。標準ライブラリのみ。"""
        import tracemalloc
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        def _loop():
            while not self._stop.is_set():
                # stop に即応するため Event.wait で待つ（time.sleep だと最大 interval 遅延する）
                if self._stop.wait(interval_s):
                    break
                try:
                    cur, peak = tracemalloc.get_traced_memory()
                    top = tracemalloc.take_snapshot().statistics("lineno")[:5]
                    self.info("mem_snapshot",
                              py_current_mb=round(cur / 1e6, 1),
                              py_peak_mb=round(peak / 1e6, 1),
                              top=[f"{s.traceback[0]}: {s.size / 1e6:.1f}MB" for s in top])
                except Exception:
                    pass  # スナップショット失敗は本処理に波及させない

        threading.Thread(target=_loop, name="dcr-memsnap", daemon=True).start()

    def stop(self, timeout=5.0):
        try:
            self.event("shutdown", uptime_s=round(time.monotonic() - self._start_mono, 1))
        except Exception:
            pass
        self._stop.set()
        if self._t:
            self._t.join(timeout)
        self._write_session_end_markers()  # best-effort（クラッシュ時は書けないが正常終了時は残る）
        self._flush_all(); self._close_all()

    # ---------- internals: utilities ----------
    def _now(self) -> str:
        # ミリ秒まで記録（%f はマイクロ秒6桁なので末尾3桁を切り捨ててms精度にする）
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def _date(self) -> str:
        return self._session

    # ---------- internals: queue ----------
    def _put(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            self.dropped += 1     # never block the real-time loop

    def _run(self):
        last_flush = time.monotonic()
        while not self._stop.is_set() or not self.q.empty():
            try:
                kind, payload = self.q.get(timeout=0.2)
            except queue.Empty:
                pass
            else:
                try:
                    if kind == "cycle":
                        self._write_csv("cycle", CYCLE_COLUMNS, payload)
                    elif kind == "health":
                        payload["dropped_logs[n]"] = self.dropped
                        self._write_csv("health", HEALTH_COLUMNS, payload)
                    elif kind == "det":
                        self._write_det(*payload)
                    elif kind == "event":
                        self._write_jsonl(payload)
                except Exception as e:
                    # 書き込み失敗はホットパスに波及させず記録だけ残す
                    self._safe_event_error("logger_write_error", str(e))
            if time.monotonic() - last_flush >= self.flush_interval_s:
                self._flush_all(); last_flush = time.monotonic()

    # ---------- internals: echo ----------
    def _echo_event(self, rec, sev):
        """events を logging 経由でコンソールへ整形表示する。
        rec に人間可読の "msg" があればそれを優先表示し、無ければ "type key=val" を出す。
        ("msg" 自体は JSONL にもそのまま残る)"""
        try:
            level = _SEV_TO_LEVEL.get(sev, logging.INFO)
            if rec.get("msg"):
                text = rec["msg"]
            else:
                extras = " ".join(f"{k}={v}" for k, v in rec.items()
                                  if k not in ("ts", "sev", "type", "msg", "schema_version"))
                text = rec["type"] if not extras else f"{rec['type']} {extras}"
            _event_logger.log(level, text)
        except Exception:
            pass  # コンソール出力の失敗を実処理に波及させない

    # ---------- internals: file I/O ----------
    def _csv_writer(self, kind, columns):
        d = self._date()
        if kind in self._writers and self._open_date.get(kind) == d:
            return self._writers[kind]
        if kind in self._files:                     # 日付替わり→開き直し
            self._files[kind].flush(); self._files[kind].close()
        kind_dir = Path(f"{self.base}_{kind}_5goki")
        kind_dir.mkdir(parents=True, exist_ok=True)
        path = kind_dir / f"{kind}_{d}.csv"
        new = not path.exists()
        fh = open(path, "a", newline="", encoding="utf-8")
        if not new:
            # 既存ファイルへの追記 = 同日に再起動した証拠。セッション境界を可視化する。
            # pandas では read_csv(..., comment='#') で読み飛ばせる。
            fh.write(f"# === SESSION START {self._session_start_dt} ===\n")
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if new:
            w.writeheader()
        self._files[kind] = fh; self._writers[kind] = w; self._open_date[kind] = d
        return w

    def _write_csv(self, kind, columns, fields):
        row = {c: "" for c in columns}
        if "schema_version" in row:
            row["schema_version"] = SCHEMA_VERSION
        # None は欠損として空文字に統一する（"None" 文字列を混ぜない）
        row.update({k: ("" if v is None else v) for k, v in fields.items() if k in columns})
        self._csv_writer(kind, columns).writerow(row)

    def _write_det(self, cycle_id, ts, items):
        w = self._csv_writer("detections", DET_COLUMNS)
        for it in items:
            row = {c: "" for c in DET_COLUMNS}
            row.update(schema_version=SCHEMA_VERSION, timestamp=ts, cycle_id=cycle_id)
            row.update({k: ("" if v is None else v) for k, v in it.items() if k in DET_COLUMNS})
            w.writerow(row)

    def _jsonl_handle(self):
        kind, d = "events", self._date()
        if kind in self._files and self._open_date.get(kind) == d:
            return self._files[kind]
        if kind in self._files:
            self._files[kind].flush(); self._files[kind].close()
        kind_dir = Path(f"{self.base}_events_5goki")
        kind_dir.mkdir(parents=True, exist_ok=True)
        path = kind_dir / f"{kind}_{d}.jsonl"
        fh = open(path, "a", encoding="utf-8")
        self._files[kind] = fh; self._open_date[kind] = d
        return fh

    def _write_jsonl(self, rec):
        self._jsonl_handle().write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------- internals: housekeeping ----------
    def _write_session_end_markers(self):
        """正常終了時にCSVへ SESSION END コメントを書き込む（best-effort）。
        クラッシュ時は書けないが、次回起動の SESSION START で区切りは把握できる。"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for kind, fh in self._files.items():
            if kind == "events":   # JSONL は shutdown イベントで十分
                continue
            try:
                fh.write(f"# === SESSION END   {ts} ===\n")
            except Exception:
                pass

    def _safe_event_error(self, type, msg):
        try:
            self._write_jsonl({"ts": self._now(), "sev": "ERROR", "type": type, "message": msg[:300]})
        except Exception:
            pass

    def _flush_all(self):
        for fh in self._files.values():
            try:
                fh.flush(); os.fsync(fh.fileno())
            except Exception:
                pass

    def _close_all(self):
        for fh in self._files.values():
            try:
                fh.close()
            except Exception:
                pass
