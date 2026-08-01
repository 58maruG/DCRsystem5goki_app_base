# Jetson Orin Nano Super 移行調査メモ

現行環境（Core i9 + RTX 4090 Laptop / Windows）で動作している本システムを
**Jetson Orin Nano Super（6コア Cortex-A78AE / 統合メモリ8GB）** へ移行できるかを
コード解析と実測で検証した記録。

作成日: 2026-07-22

---

## 0. この文書の読み方（重要）

数値には **実測値** と **推定値** が混在している。取り違えると判断を誤るので明示する。

| 印 | 意味 |
|---|---|
| **[実測]** | 実際にコードを走らせて計測した値 |
| **[計算]** | 設定ファイルの値から算術的に導いた値 |
| **[推定]** | アーキテクチャからの推論。実機未検証 |

### 計測環境の制約

計測は **本番機（i9 + RTX4090）ではなく、解析用の別PC** で行った。

- CPU: AMD Ryzen (Family 25 Model 117 = Zen4 世代) / **16論理コア**
- torch: **2.12.1+cpu（CUDA無し）** → GPU推論そのものは計測できていない
- OS: Windows 11

したがって以下は**計測できていない**:

- GPU推論のレイテンシ（TensorRT化後のFPS）
- ARM Cortex-A78AE の絶対性能
- 実機のUSB3ホストコントローラ挙動

「Jetson換算 ×3」と書いた箇所は、NEON(128bit) vs AVX2(256bit) のSIMD幅差と
クロック差からの **[推定]** であり、実測ではない。実機入手後に `bench/` の
スクリプトをそのまま走らせれば係数が確定する。

---

## 1. 結論サマリ

| 項目 | 判定 | 根拠 |
|---|---|---|
| メモリ 8GB | **問題なし** | 実測+計算で約2.3〜3.2GB。4.8GB以上余る |
| USB帯域 | **問題なし** | pfs実解像度から83MB/s。USB3.0の実効限界に対し余裕大 |
| **CPU 6コア** | **要対策** | 現状構成のままでは締切超過。下記の設定変更が必須 |
| GPU推論 | **未検証** | TensorRT変換後の実測が必要。最大の未確定要素 |

**最重要**: ボトルネックは GPU ではなく **GUIスレッド（CPU）** になる公算が高い。
6コアに制限した実測で、現状構成は既に **p95=51ms / 50ms締切超過5.1%** と限界にある。
しかもこれは Jetson より速い x86 コアでの結果。

**必須の対策2つ**（詳細は §9）:
1. `cv2.setNumThreads(1)` — 入れないとネスト並列でHSV処理が3.6倍悪化する
2. HSV処理を既存の4カメラスレッドへ移す — 逐次21ms → 並列9.5ms

---

## 2. 設定ファイル棚卸し

リポジトリ内の設定ファイルを漏れなく確認した結果。

| ファイル | 内容 | 見積もりへの影響 |
|---|---|---|
| `cam_pfs/cam_top_25453227.pfs` | **640×640** RGB8 / 帯域上限163MB/s(On) | 実解像度が判明 |
| `cam_pfs/cam_under_25453229.pfs` | **640×640** RGB8 / 帯域上限163MB/s(On) | 同上 |
| `cam_pfs/cam_inside_25308967.pfs` | **560×560** RGB8 / 帯域上限モード **Off** | 同上＋不整合発見 |
| `cam_pfs/cam_outside_25308968.pfs` | **500×500** RGB8 / 帯域上限163MB/s(On) | 同上 |
| `json/delay_config.json` | cam_under=2.023s / cam_inside=2.015s | 遅延キューのメモリ量が確定 |
| `json/relay_config.json` | 排出角度(remove 67.5° / transport 112.5°) | 影響なし |
| `json/hsv_config_cam_*.json` ×4 | HSV閾値 | 影響なし（数十バイト） |
| `pyproject.toml` | `requires-python = ">=3.12"` | JetPack互換性リスク（§11） |
| `Arduino/turntable_control.ino` | `Serial.begin(115200)` | 別MCU。Jetson資源に無関係 |
| ~~`ripeness_classifier/json/*.json`~~ | HSV閾値 | 2026-07-23 に本プロジェクト外へ移動（§2.1） |

全カメラ共通: `PixelFormat=RGB8` / `AcquisitionFramePeriodRaw=50000`(=20fps) /
`SensorShutterMode=Global`

### 2.1 ripeness_classifier について

**2026-07-23、`C:\Users\kotan\gohara\cherry_yolo\ripeness_classifier` へ移動した。
本プロジェクトには存在しない。**

学習データ整備用の別GUIツールであり、稼働中のメモリ・スレッド数には影響しない
（内部に `QThread` を3種持つが main 実行時には起動しない）。

ただし移動前の本メモには「`main_5goki_JP_v3.py` から一切 import されていない」と
記載していたが、これは**誤解を招く記述だった**。main から直接 import されていないだけで、
`module_yolo.py` を経由して `ripeness_classifier.common.mask_utils` に
**推移的に依存**していた（果実検出1段目のHSVマスク生成＝ホットパス）。

分離にあたり、本番が使う3関数（`mask_from_hsv` / `remove_stem` /
`remove_reflection_sat`）はプロジェクト直下の `hsv_mask_utils.py` へ移した。
`module_yolo.py` と `standalone/hsv_calibration.py` が同ファイルを共有する。
移動先の `ripeness_classifier/common/mask_utils.py` にも同一実装が残っているため、
マスク生成アルゴリズムを変更する際は両方を揃えること（閾値はJSON側なので影響しない）。

---

## 3. スレッド構成

### 3.1 コードが明示的に起動するスレッド（固定10本）

| 発生元 | 数 | 内容 |
|---|---|---|
| メインスレッド | 1 | Qtイベントループ |
| `module_cameras_5goki.py:119` | **4** | カメラ別キャプチャ（`_capture_loop`） |
| `module_yolo.py:388` | 1 | YOLO推論ワーカー（4カメラ共有・キュー経由） |
| `module_motor_serial.py:106` | 1 | Arduino受信（`_reader_loop`） |
| `module_motor_serial.py:112` | 1 | Arduinoハートビート |
| `dcr_logger.py:150` | 1 | ログ書き込み（`dcr-logger`） |
| `dcr_logger.py:182` | 1 | tracemallocスナップショット（`dcr-memsnap`） |
| **合計** | **10** | |

加えて `main_5goki_JP_v3.py:249` の `QThreadPool` がリレー／パトライト操作を
イベント駆動で実行（上限=論理コア数、実際の同時実行は1〜2本）。

### 3.2 「6コアに10スレッドは足りない」は誤解

**スレッドの数ではなく、同時に実行可能な数がコアを消費する。**
定常状態では大半がブロック中で、コアを消費しない。

| スレッド | 定常状態 | コア消費 |
|---|---|---|
| カメラ×4 | `RetrieveResult` でUSBフレーム待ち | ほぼ0 |
| 推論×1 | `queue.get(timeout=0.1)` 待ち | ほぼ0 |
| Arduino×2 | シリアル読み待ち・sleep | ほぼ0 |
| ロガー×2 | `queue.get` / `Event.wait` 待ち | ほぼ0 |
| GUI×1 | イベントループ待ち | ほぼ0 |

20fpsなら各カメラスレッドは50msに1回しか起きない。

### 3.3 本当の危険はライブラリ内部スレッド **[実測]**

アプリの10本より、import しただけで増えるライブラリ内部スレッドの方が多い。

```
baseline                    :  4 OSスレッド
+ torch import              : 22 (+18)   MKL/OpenMPプール
+ cv2 import                : 22
行列演算の実行後            : 29 (+7)    intra-opプールが実体化

torch.get_num_threads()     : 8   ← 既定でコア数
torch.get_num_interop_threads(): 8
cv2.getNumThreads()         : 16  ← 既定でコア数
```

Jetson では両方が **6** を要求する。これが §8 のオーバーサブスクリプション問題の原因。

---

## 4. メモリ見積もり

### 4.1 ライブラリのRSS実測 **[実測]**

```
baseline                : 17.6 MB
+ numpy                 : 30.2 MB
+ opencv-python         : 37.4 MB
+ torch (CPU版)         : 211.1 MB
+ PySide6               : 221.2 MB
+ ultralytics           : 234.4 MB
```

### 4.2 カメラ関連メモリ **[計算]**

pfsの実解像度と `MaxNumBuffer=20`（`module_cameras_5goki.py:92`）から算出。

Pylon内部バッファ:

| カメラ | 解像度 | 1フレーム | ×20バッファ |
|---|---|---|---|
| cam_top | 640×640×3 | 1.229 MB | 24.58 MB |
| cam_under | 640×640×3 | 1.229 MB | 24.58 MB |
| cam_inside | 560×560×3 | 0.941 MB | 18.82 MB |
| cam_outside | 500×500×3 | 0.750 MB | 15.00 MB |
| **計** | | | **82.97 MB** |

遅延キュー（`FPS=20.0`、`delay_frames = int(delay_seconds × 20)`）:

| カメラ | 遅延秒 | フレーム数 | 容量 |
|---|---|---|---|
| cam_under | 2.023s | 40 | 49.15 MB |
| cam_inside | 2.015s | 40 | 37.63 MB |
| **計** | | | **86.78 MB** |

**カメラ関連合計: 約170 MB**

> 注: 会話中に口頭で「79MB / 83MB」と述べたが、再計算した上記が正しい（83MB / 87MB）。

### 4.3 総計

| 項目 | 見積もり | 種別 |
|---|---|---|
| ライブラリ本体 | 234 MB | [実測] |
| カメラ関連（バッファ＋遅延キュー） | 170 MB | [計算] |
| CUDA context + TensorRTランタイム | 300〜600 MB | [推定] |
| モデル重み（v4_11s.pt / YOLO11s） | 20〜50 MB | [推定] |
| Qt GUI描画バッファ | 20〜40 MB | [推定] |
| dcr_loggerキュー（`queue_size=20000`） | 最大10〜20 MB | [計算] |
| OS + デスクトップGUI(Ubuntu L4T) | 1.5〜2 GB | [推定] |
| **合計** | **約2.3〜3.2 GB** | |

→ 8GB統合メモリに対し **4.8〜5.7GB の余裕**。メモリは制約にならない。

**注意点**: 遅延キューは `delay_seconds` に比例する。キャリブレーションで
遅延秒数を伸ばすとメモリも増える（例: 4秒 → 遅延キューだけで約174MB）。

---

## 5. USB帯域 **[計算]**

pfsの実解像度 × 20fps:

```
cam_top    640×640×3 × 20fps = 24.58 MB/s
cam_under  640×640×3 × 20fps = 24.58 MB/s
cam_inside 560×560×3 × 20fps = 18.82 MB/s
cam_outside 500×500×3 × 20fps = 15.00 MB/s
------------------------------------------
合計                          = 82.97 MB/s
```

`module_cameras_5goki.py:94` のコメントは「4台で400MB/s = USB3.0の実効限界付近」
としているが、**実際は約83MB/s** で桁が違う。Jetsonの共有USBコントローラでも
問題になりにくい。

---

## 6. ホットパス実測 **[実測]**

`update_video_feeds`（QTimer 50ms周期）がGUIスレッドで抱える処理。
pfsの実解像度と `hsv_config_*.json` の実値を使用。

### 6.1 HSV判定の内訳（cv2内部スレッド=16）

| カメラ | HSV全体 | cvtColor | morphology×4 | connectedComponents |
|---|---|---|---|---|
| cam_top (640) | 5.00 ms | 0.49 | 1.19 | 1.19 (labels 1.64MB確保) |
| cam_under (640) | 4.40 ms | 0.43 | 0.98 | 1.06 (1.64MB) |
| cam_inside (560) | 3.30 ms | 0.22 | 0.77 | 0.90 (1.25MB) |
| cam_outside (500) | 2.78 ms | 0.22 | 0.79 | 0.42 (1.00MB) |
| **4台合計** | **15.47 ms** | | | |

### 6.2 GUIスレッドの総コスト

| 処理 | 4台合計 |
|---|---|
| HSV判定 | 15.5 ms |
| `.copy()` 群 + resize | 約5.5 ms |
| Qt表示（cvtColor + QPixmap + Smooth縮小） | 5.3 ms |
| **合計** | **約26 ms / 50ms予算** |

x86(Zen4)で既に予算の **52%** を消費。
Jetson換算(×2.5〜3.5) **[推定]** で **65〜90ms** となり **50ms締切を超過**する。

### 6.3 cv2内部スレッド数の影響 **[実測]**

| cv2スレッド | 現状実装 | 最適化(kernel事前生成+morph 1回) | 半解像度HSV |
|---|---|---|---|
| 16 | 15.18 ms | 12.73 ms | 4.85 ms |
| 6 | 14.64 ms | 12.12 ms | 4.03 ms |
| 2 | 18.60 ms | 16.10 ms | 4.99 ms |
| 1 | 17.84 ms | 16.05 ms | 4.68 ms |

**この解像度では cv2 の内部並列がほとんど効いていない**（16スレッド→1スレッドで
17%しか悪化しない）。つまり `setNumThreads(1)` の代償は小さく、コアを他へ回せる。

半解像度HSVは **3.8倍高速**（17.84 → 4.68ms）で単体最大の効果。

### 6.4 tracemalloc は無害だった **[実測]**

`main_5goki_JP_v3.py:338` の `start_mem_snapshot(60.0)` が `tracemalloc.start()` を
呼び常時有効になっている点を疑って計測したが:

```
tracemalloc OFF : 21.50 ms/tick
tracemalloc ON  : 20.59 ms/tick
tracemalloc OFF : 19.65 ms/tick (再測定)
オーバーヘッド  : +0.1%
take_snapshot() : 0.2 ms
```

**オーバーヘッドは誤差範囲**。numpy/OpenCVのC層アロケーションはtracemallocの
フックを通らないため。当初の仮説は誤りで、急いで削除する必要はない。

---

## 7. GUI: PySide6 vs Tkinter **[実測]**

表示経路のコスト（4台合計・1ティック・480×360へ縮小表示を想定）:

| 経路 | 実測 |
|---|---|
| **PySide6** `QPixmap.scaled(Smooth)` = 現状 | **5.32 ms** |
| PySide6 `FastTransformation` | 3.74 ms |
| **PySide6 `cv2.resize` → QImage化** | **2.40 ms** ← 最速 |
| **Tkinter** PIL resize + ImageTk | **25.68 ms** ← 現状の約5倍遅い |
| Tkinter `cv2.resize` + ImageTk | 7.29 ms |

import時のRSS:

```
tkinter + PIL : +6.3 MB
PySide6       : +16.7 MB
```

### 結論: Tkinterへの変更は逆効果

メモリは10MBしか変わらない。一方でメモリは4.8GB以上余っており（§4）、
**CPUこそが制約**（§8）。「10MB節約するためにCPUを3〜5倍使う」のは
最も避けるべきトレード。**GUIはPySide6のまま据え置く。**

---

## 8. 4スレッド並列とオーバーサブスクリプション

### 8.1 前提: スレッドは増えない

「HSVを4スレッドに分ける」提案は、**新規スレッドの追加ではなく、
`module_cameras_5goki.py:119` に既に存在する4本のキャプチャスレッドへ
GUIスレッドの仕事を移す**もの。スレッド総数は10本のまま。

### 8.2 GILは障害にならない **[実測]**

cv2内部スレッド=1に固定し、4台のHSVを逐次 vs 4スレッド並列で比較:

```
逐次 (現状のGUIスレッドforループ) : 21.05 ms
4スレッド並列                      :  9.49 ms
高速化率                           : 2.22x
```

OpenCVは処理中にGILを解放するため、Pythonスレッドでも実際に並列化が効く。

### 8.3 6コア制限下での構成比較 **[実測]**

CPU affinityを6コアに固定（Jetsonのコア数を再現）。
カメラ20fps・推論CPU負荷スレッド常駐・6秒計測。

| 構成 | OSスレッド数 | HSV 1台1フレーム p50 | GUIティック p50/p95 | 50ms締切超過 |
|---|---|---|---|---|
| **[A] 現状**（GUIで4台逐次, cv2=6） | 26 | 5.31 ms | 24.4 / **51.0 ms** | **5.1%** |
| **[B] 4スレッド並列, cv2=6のまま** | 30 | **19.03 ms** | 3.9 / 7.7 ms | 0% |
| **[C] 4スレッド並列, cv2=1に固定** | 30 | 6.45 ms | 4.3 / 6.2 ms | 0% |

**[B] がオーバーサブスクリプションの実例**: OpenCVは既定でコア数分（Jetsonなら6）の
内部スレッドを使うため、4本のワーカーが同時にOpenCVを呼ぶと **4×6=24本** が
6コアを奪い合い、1フレームあたりが 5.31→19.03ms へ **3.6倍悪化**した。

**[C] のように `cv2.setNumThreads(1)` を入れるだけで 6.45ms に回復**（Bの2.9倍改善）。

### 8.4 CPU時間で見た必要コア数

| 構成 | CPU時間/50ms | 必要コア数 [実測] | Jetson換算(×3) [推定] |
|---|---|---|---|
| [A] 現状 | 21.2 ms | 0.42 | **GUIティック73ms → 締切破綻** |
| [B] ネスト並列 | 76.1 ms | 1.52 | **4.6コア / 6** → 飽和 |
| [C] 推奨 | 25.8 ms | 0.52 | **1.6コア / 6** → 余裕あり |

**結論: 4スレッド化は6コアでも問題ない。ただし `cv2.setNumThreads(1)` が必須条件。**

なお **[A]の現状構成が6コアでは既に限界**（p95=51ms・超過5.1%）である点に注意。
x86の速いコアでこれなので、無対策でJetsonへ移すのが最も危険。

---

## 9. 適用すべき設定（最優先）

```python
# main の起動時、カメラ・推論スレッド開始より前に実行する
import cv2, torch
cv2.setNumThreads(1)        # ネスト並列の防止（最重要）
torch.set_num_threads(2)    # 推論CPU側。6コア全部を要求させない
```

`torch` も既定で `get_num_threads() == コア数`（実測で8）なので同じ罠を持つ。
OpenMPプールは import 時に確定するため、より確実にするならプロセス起動前に
環境変数を設定する:

```bash
export OMP_NUM_THREADS=2
```

注: `cv2.setNumThreads()` はプロセス全体に効く設定で、スレッドごとの指定はできない。
GUIの `cv2.resize` にも影響するが §6.3 の通り実測上ほぼ無害。

---

## 10. 改善案（効果の大きい順）

| # | 変更 | 実測効果 | リスク |
|---|---|---|---|
| 1 | `cv2.setNumThreads(1)` / `torch.set_num_threads(2)` | §8.3 [B]→[C] で2.9倍 | ほぼ無し。**最初に入れるべき** |
| 2 | HSV/前処理を4カメラスレッドへ移動 | 21.0→9.5 ms | 状態共有部の切り分けが必要（§10.1） |
| 3 | HSV判定を半解像度で実行（YOLO入力は原寸維持） | 17.8→4.7 ms (3.8倍) | `min_area=500` と `BAND_HALF_PX` の再調整が必須 |
| 4 | Qt表示を `cv2.resize` → QImage化 | 5.3→2.4 ms | ほぼ無し |
| 5 | 画像収集の停止 | コピー2.4MB/フレーム削減 | §10.2 参照 |
| 6 | `np.array`/`getStructuringElement` 事前生成、morph反復2→1 | 17.8→16.1 ms | 反復削減はマスク品質に影響 |
| 7 | `module_yolo.py:780` の `.copy()` 削除 | 1.2MB/フレーム削減 | 無し（下記） |

### 10.1 #2 実装時の制約

並列化できる／できない処理の切り分け:

| 対象 | 並列化 | 理由 |
|---|---|---|
| HSV判定・クロップ・リサイズ | **可** | カメラ独立 |
| `self.trackers[cam_name]` | **可** | カメラ別辞書 |
| `self.obj_detections` / `obj_active` / `obj_first_seen` 等 | **不可** | 4カメラ共有の個体状態 |
| `self.current_cherry_id` / `last_seen_time` / `_obj_generation` | **不可** | 同上 |

→ ByteTracker適用と状態更新（`_apply_inference`）は**単一スレッドに残す**。

推論本体は、4スレッドから同時にGPUを叩いても Jetson の iGPU では結局直列化するため、
**4枚を `batch=4` で1回の `predict` にまとめる**方がGPU効率が上がる。

### 10.2 #5 の背景

`module_yolo.py:212` と `:251` を見ると `cv2.imwrite` は既にコメントアウトされ、
**画像保存は停止中**。ところが供給側は全速力で動いており:

- `_update_cam_tile` の `frame.copy()`（1.2MB）
- `entry['frame'] = img.copy()`（1.2MB）

が検出のたびに実行され、`obj_cam_tile` / `obj_cam_train` に溜めては捨てている。
保存を当面再開しないなら、この収集自体をフラグで止めるのが確実な削減になる。

### 10.3 #7 の根拠

`module_yolo.py:764` で `input_img_resized = cv2.resize(...)` が
新規配列を返しており、`:780` の `.copy()` は冗長。ローカル変数で毎回作り直されるため
他から変更される恐れがない。

---

## 11. 移行時に修正が必要な箇所

| # | 箇所 | 問題 | 対応 |
|---|---|---|---|
| 1 | `telemetry_sources.py:10` `gpu_stats()` | `pynvml`(NVML)前提。**JetsonのGPUはNVMLで見えない** | `jtop`(jetson-stats) か `tegrastats` へ差し替え。放置すると `gpu_temp`/`gpu_util`/`gpu_power` が常に空 |
| 2 | `module_motor_serial.py:84` `_detect_port()` | Windowsの `COM3` 系を前提 | Linuxの `/dev/ttyACM*` `/dev/ttyUSB*` に対応させる |
| 3 | ~~`pyproject.toml:6`~~ | ~~`requires-python = ">=3.12"`~~ | **2026-07-22 対応済み [実測]**。`requires-python = ">=3.10,<3.11"` へ変更し `uv sync` で本PCのvenvもPython 3.10.19へ切替済み（§14.10参照） |
| 4 | `module_cameras_5goki.py:97` → `:99-104` | コードで `DeviceLinkThroughputLimit=100MB/s` を設定した**直後**に `load_pfs_custom()` がpfsを読み込み上書き。pfs側は163MB/s、`cam_inside` に至っては制限モード自体がOff → **コードの意図した制限が無効化されている** | 実害は小さい（実必要帯域83MB/s）が、意図と実態の食い違いは移行前に解消すべき |
| 5 | 依存パッケージ全般 | JetPackのバージョンに固定されたNVIDIA提供ARM64版 torch / TensorRT を使う必要がある | ultralytics のバージョンとの組合せ検証。Jetson用ビルドは本家より更新が遅れがち |
| 6 | pypylon | ARM64 Linux版の対応確認が必要 | Basler公式の対応状況を要確認 |

---

## 12. 実機で確認すべきこと（未確定事項）

優先度順。①〜③が「動くか動かないか」を左右する。

1. **TensorRT変換後の実推論FPS**
   4カメラ×20fps = 80fps分の推論要求に対し、現状の `.pt`(PyTorch) のままでは非現実的。
   TensorRTエンジン（FP16/INT8）への変換が前提。変換後の実測が必須。
   - エンジンビルド自体がメモリを圧迫する可能性 → デスクトップGUIを閉じた状態でビルドし、
     生成済み `.engine` だけをアプリ実行時に読み込む運用にする

2. **4カメラ同時USB3グラブの安定性**
   キャリアボードのUSB3ホストコントローラが独立か共有か。実機で4台同時グラブして
   ドロップ率を確認（`module_cameras_5goki.py` の `[CamStats]` ログで追える）

3. **6コアでのスレッド競合**
   `bench/bench_oversub.py` をJetson実機で実行し、§8.3 の表を実測値で埋める

4. **メモリ実測**: `tegrastats` でRAM/GPU使用率のピークを記録
5. **サーマル**: TDPモード(7/15/25W)切替とSuper Mode持続動作時のスロットリング有無
6. **排出タイミング精度**: 推論レイテンシ増大が `delay_seconds` ベースの排出計算
   （`main_5goki_JP_v3.py` の `_relay_and_log`）に影響しないか

---

## 13. ベンチマークの再実行方法

`bench/` に計測スクリプトを配置済み。リポジトリルートからの相対パスで動くよう
修正してあるので、別PCやJetson実機でもそのまま実行できる。

```bash
# 依存: numpy, opencv-python, psutil, PySide6, Pillow(tkinter比較用)
python bench/bench_hotpath.py    # HSVパイプラインの内訳（§6.1）
python bench/bench_opt.py        # cv2スレッド数別＋最適化案の比較（§6.3）
python bench/bench_gui.py        # PySide6 vs Tkinter 表示経路（§7）
python bench/bench_threads.py    # 逐次 vs 4スレッド並列（§8.2）
python bench/bench_oversub.py    # 6コア制限下の構成比較（§8.3）※約20秒
```

- `bench_oversub.py` は `os.cpu_count() > 6` のとき自動でaffinityを6コアに絞る。
  Jetson実機（6コア）ではそのまま全コアで走る。
- 擬似フレーム（赤い円）を使うため実際のサクランボ画像は不要。
  ただし `json/hsv_config_*.json` と pfs の解像度定義に依存するので、
  設定を変えたら数値も変わる。

---

## 14. 本PC（現行環境そのもの）での再実測 [実測]

追記日: 2026-07-22

§1〜13 は解析用の別PC（AMD Zen4 / torch CPU版）での実測だった。今回、
**本メモ冒頭でいう「現行環境」＝ Core i9 + RTX 4090 Laptop（本機）** で
`bench/` のスクリプト一式を新規作成し、同じ計測を実機で行った。

### 14.0 本PCの計測環境

```
CPU        : Intel Core i9 (32論理コア)
GPU        : NVIDIA GeForce RTX 4090 Laptop GPU（17.2GB, torch 2.12.1+cu126 で認識・CUDA利用可）
OS         : Windows 11
Python     : 3.12.12
opencv-python: 4.13.0
```

別PCとの違い: 別PCは16論理コア/torch CPU版だったため GPU推論のレイテンシは
一切測れなかった。本PCは CUDA が有効な torch を認識しているため、GPU推論
（TensorRT変換後のFPS、§12-1）を計測できる条件は揃っている。ただし今回は
**§13のbenchスクリプト（CPU側のホットパス計測）のみを再実行**しており、
TensorRT変換・GPU推論のベンチマークは未実施（次のステップとして別途着手可能）。

### 14.1 新たな発見: ultralytics importの副作用

`import ultralytics` した瞬間に `cv2.getNumThreads()` が **自動的に1へ変更される**
ことを確認した（本PCで `cv2.setNumThreads()` を呼ばず素の状態で確認済み）。

```
ultralytics import前: cv2.getNumThreads() = 32（論理コア数）
ultralytics import後: cv2.getNumThreads() = 1
```

つまり `module_yolo.py` が `ultralytics` を import する現行コードでは、
**§9で提案している `cv2.setNumThreads(1)` は実質的に import 時点で既に効いている**
可能性が高い（明示的に呼んでいなくても）。とはいえ ultralytics のバージョンに
依存する非公式の副作用であり、将来のバージョンアップで挙動が変わり得るため、
§9の推奨（明示的に `cv2.setNumThreads(1)` を呼ぶ）は引き続き有効。

### 14.2 §6.1 HSV判定の内訳（本PC・cv2スレッド=32へ明示的に戻して計測）

```
python bench/bench_hotpath.py
```

| カメラ | HSV全体(中央値) | cvtColor | morphology×4 | connectedComponents | refine |
|---|---|---|---|---|---|
| cam_top (640) | 7.40 ms | 0.39 | 0.50 | 1.43 | on |
| cam_under (640) | 6.86 ms | 0.39 | 0.48 | 1.44 | on |
| cam_inside (560) | 5.07 ms | 0.24 | 0.41 | 1.30 | on |
| cam_outside (500) | 4.06 ms | 0.24 | 0.38 | 0.49 | on |
| **4台合計** | **23.40 ms** | | | | |

別PC(§6.1)の4台合計 15.47ms より**遅い**。理由は cv2スレッド数の既定値の違い
（別PCは16、本PCは32）。この解像度(500〜640px)では内部並列がほとんど効かず、
スレッド数を増やすほどスレッドプールの同期オーバーヘッドが相対的に効いてくる
（§14.3で裏付け）。**i9の方が1コアあたりは速いはずだが、既定スレッド数が
多すぎることでその優位性が相殺されている** — 本メモの主張（スレッド数の
明示的な制御が重要）を裏付ける結果になった。

### 14.3 §6.3 cv2スレッド数別＋最適化案の比較

```
python bench/bench_opt.py
```

| cv2スレッド | 実装 | 4カメラ合計(中央値) |
|---|---|---|
| 32 | 現状実装 | 23.24 ms |
| 32 | 最適化(kernel事前生成+morph1回) | 20.73 ms |
| 32 | 半解像度HSV | 9.13 ms |
| **6** | **現状実装** | **16.05 ms** |
| 6 | 最適化(kernel事前生成+morph1回) | 14.33 ms |
| 6 | 半解像度HSV | 5.83 ms |
| 2 | 現状実装 | 17.34 ms |
| 2 | 最適化(kernel事前生成+morph1回) | 15.94 ms |
| 2 | 半解像度HSV | 5.62 ms |
| 1 | 現状実装 | 18.61 ms |
| 1 | 最適化(kernel事前生成+morph1回) | 16.90 ms |
| 1 | 半解像度HSV | 5.66 ms |

cv2スレッド=6が最速（32→6で 23.24→16.05ms、約1.4倍改善）。1・2スレッドでも
6スレッドより若干遅い程度で大差はなく、別PCの結論「この解像度ではcv2の内部
並列がほとんど効かない」は本PCでも成立する。**cv2スレッド=6での現状実装
16.05msは、別PCがcv2=16で測った15.47msとほぼ一致**しており、適切なスレッド数
さえ選べば別PCと同等の性能が出ることが分かる。半解像度HSVは6スレッドで
16.05→5.83ms（2.75倍）と単体最大の効果である点も別PCと同じ傾向。

### 14.4 §7 PySide6 vs Tkinter 表示経路

```
python bench/bench_gui.py
```

| 経路（4台合計・1ティック） | 実測(中央値) |
|---|---|
| PySide6 `QPixmap.scaled(Smooth)` = 現状 | 3.16 ms |
| PySide6 `FastTransformation` | 2.74 ms |
| PySide6 `cv2.resize` → QImage化 | **1.58 ms** ← 最速 |
| Tkinter PIL resize + ImageTk | 14.78 ms ← 現状の約4.7倍遅い |
| Tkinter cv2.resize + ImageTk | 3.92 ms |

別PC(§7)と傾向は完全に一致（Tkinter PILが最も遅く、PySide6 cv2.resizeが最速）。
絶対値は本PCの方が高速（クロック差）。**結論は変わらず: GUIはPySide6のまま
据え置くべき。**

### 14.5 §8.2 逐次 vs 4スレッド並列（cv2スレッド=1固定）

```
python bench/bench_threads.py
```

| 構成 | 4カメラ合計(中央値) |
|---|---|
| 逐次（現状のGUIスレッドforループ） | 18.88 ms |
| 4スレッド並列 | 7.88 ms |

高速化率 **2.40x**（別PCは2.22x）。OpenCVが処理中にGILを解放し、Pythonスレッド
でも並列化が効くという結論は本PCでも再現した。

### 14.6 §8.3 6コア制限下での構成比較 — 別PCと異なる結果が出た点に注意

```
python bench/bench_oversub.py
```

6コアへaffinity制限し、1コア分のCPU負荷スレッド（SHA256計算でGILを解放しつつ
常時1コアを使用）を背景に走らせた状態で6秒計測（4カメラ分の処理を「逐次」
または「4スレッド並列＋join」で同期的に実行し、そのティック時間を測定）。

| 構成 | OSスレッド数 | GUIティック p50 | p95 | 50ms締切超過 | 必要コア数(実測) |
|---|---|---|---|---|---|
| [A] 現状（GUIで4台逐次, cv2=6） | 38 | 28.10 ms | 33.64 ms | 0.0% | 2.46 |
| [B] 4スレッド並列, cv2=6のまま | 43 | 17.22 ms | 18.69 ms | 0.0% | 3.43 |
| [C] 4スレッド並列, cv2=1に固定 | 43 | 16.89 ms | 18.77 ms | 0.0% | 2.71 |

**別PCの結論（[B]で3.6倍悪化するオーバーサブスクリプション）が本PCでは
再現しなかった。** [A]→[B]はむしろ速くなっている（28.10→17.22ms）。加えて
50ms締切超過は全構成で0%（別PCの[A]は5.1%超過だった）。

考えられる理由:
- 本PCのOpenCV(4.13.0)は別PC計測時と異なるビルド・スレッディングバックエンド
  の可能性があり、ネスト並列時の挙動が違う
- i9の1コアあたりの性能が高く、6コアに絞っても余裕がある
- affinity制限の実装（Windows `SetProcessAffinityMask` 経由）とAMD機の挙動の違い

**実務上の含意**: `cv2.setNumThreads(1)` を入れる判断そのものは変えなくてよい
（§14.3の通りコストはほぼ無い一方、下振れリスクを消せる保険になる）。ただし
**「オーバーサブスクリプションで3.6倍悪化する」という具体的な悪化率はこの
2台のx86機の間でも一致しない**、つまり **Jetson実機（ARM Cortex-A78AE・
NEON・6コア固定）で同じ現象が起きる保証はない**。§12-3「6コアでのスレッド
競合」は実機実測が必須という結論は変わらない、むしろ強まった。

### 14.7 総括

| 項目 | 別PC(AMD Zen4/CPU torch) | 本PC(i9+RTX4090/CUDA torch) | 一致度 |
|---|---|---|---|
| HSV内訳の絶対値 | 15.47ms(cv2=16) | 23.40ms(cv2=32) / 16.05ms(cv2=6) | スレッド数を揃えれば近い |
| cv2内部並列がこの解像度でほぼ効かない | ○ | ○ | 一致 |
| 半解像度HSVが最大の効果 | ○(3.8倍) | ○(2.75倍) | 一致（傾向） |
| GUI: Tkinter PILが最遅・PySide6 cv2.resizeが最速 | ○ | ○ | 一致 |
| 4スレッド並列でGIL越しに2倍前後高速化 | ○(2.22x) | ○(2.40x) | 一致 |
| **オーバーサブスクリプションで3.6倍悪化** | ○ | **×（再現せず、むしろ改善）** | **不一致** |
| 50ms締切超過（6コア制限・現状構成） | 5.1% | 0.0% | 不一致（本PCの方が余裕） |

CPU側のホットパス最適化に関する結論（§9・§10の改善案の効果と優先順位）は
両PCで一致しており、対策の妥当性は高い。一方でオーバーサブスクリプションの
悪化率と締切超過率は機種依存であることが分かったため、**Jetson実機での
§12の実測（特に3番・6コアでのスレッド競合）を省略してはいけない**。

未検証のまま残っている項目（§12と同じ）:
1. TensorRT変換後の実推論FPS — §14.8で素のPyTorch分は計測済み。TensorRT化は未実施
2. 4カメラ同時USB3グラブの安定性 — 実カメラ接続が必要
3. 6コアでのスレッド競合 — 上記の通りJetson実機でしか確定しない
4〜6. メモリ実測・サーマル・排出タイミング精度 — 実機が必要

### 14.8 GPU推論レイテンシ（PyTorch, TensorRT未適用）[実測]

`tensorrt`/`onnx`/`onnxruntime` が本PCに未インストールのため、TensorRT変換は
未実施。まずは追加インストール不要な**素のPyTorch(.pt)でのGPU推論**を計測した
（`Trained_Models/v4_11s.pt` をそのまま `model.to("cuda:0")` して実行）。

```
python bench/bench_gpu_infer.py
```

現状実装（`module_yolo.py:785`）と同じ「1枚ずつ `model.predict`」呼び出しと、
§10.1で提案されているバッチ推論（4枚まとめて1回の`predict`）を比較:

| 構成 | 1枚あたり(中央値) | 4枚分のコスト | スループット |
|---|---|---|---|
| [A] 現状実装（1枚/callを4回） | 8.69 ms/枚 | 34.77 ms | **115.0 fps** |
| [B] バッチ推論（4枚/call） | 10.53 ms/枚 | 42.11 ms | 95.0 fps |

要求スループット目安 80fps（4カメラ×20fps）に対し、**[A][B]とも本PC(RTX4090
Laptop)では余裕で満たす**。

**§10.1の想定と逆の結果が出た点に注意**: メモでは「4枚をbatch=4で1回の
predictにまとめる方がGPU効率が上がる」としていたが、本PCでは**バッチ化した
方がむしろ1枚あたり21%遅い**（8.69→10.53ms/枚）。RTX4090 Laptopは1枚だけの
呼び出しでも既にGPUを十分に使い切れる性能があり、バッチ化によるPython側の
前処理（スタック生成等）オーバーヘッドの方が上回ったと考えられる。

この結果は**Jetsonには適用できない**: §10.1のバッチ化提案はJetsonの非力な
iGPUが「4スレッドから同時に叩いても結局直列化する」ことを根拠にしていた
[推定]であり、GPUが速く並列度に余裕のある本PCで否定されても、Jetsonでの
成否とは無関係（むしろJetson実機ではバッチ化が効く可能性がまだ残っている）。
**バッチ化すべきかどうかはJetson実機で計測するまで判断できない**、というのが
今回追加でわかった論点。

### 14.9 TensorRT変換の実施結果 [実測]

`uv add tensorrt onnx onnxruntime-gpu onnxslim` で依存関係を追加し
（`pyproject.toml`/`uv.lock` 更新済み）、`model.export(format="engine")` で
`Trained_Models/v4_11s.pt` をTensorRTエンジン化した。

**FP16化はブロックされた**: インストールされた `tensorrt==11.1.0.106`（TensorRT
11系）は「strongly-typed」設計のため、FP16/INT8の指定はビルダーフラグではなく
`nvidia-modelopt` によるONNXグラフへの事前量子化焼き込みが必須になっている。
`nvidia-modelopt[onnx]` を追加しようとしたところ、依存の `onnxruntime-extensions`
にWindows(`win_amd64`)向けwheelが存在せず解決不能だった:

```
error: Distribution `onnxruntime-extensions==0.15.2` can't be installed because
it doesn't have a source distribution or wheel for the current platform
hint: only has wheels for manylinux/macos platforms
```

→ **これはWindows固有の制約**。JetsonはLinux(JetPack)なので、この特定の
ブロッカーは実機では起きない可能性が高い（ただしJetPack同梱のTensorRTの
バージョン・modelopt要否は別途確認が必要）。

**FP32エンジンでの計測**（`bench/bench_tensorrt.py`）:

| モデル形式 | 1枚あたり(中央値) | スループット |
|---|---|---|
| PyTorch (.pt, FP32) | 8.40 ms | 119.1 fps |
| TensorRT (.engine, FP32) | 9.16 ms | 109.1 fps |

**TensorRT化(FP32)はむしろ0.92倍（9%遅い）** という結果になった。RTX4090
LaptopはPyTorch/cuDNNの標準カーネルだけで既に十分速く、TensorRTのグラフ
最適化の恩恵よりラッパー呼び出し（バインディング設定等）のオーバーヘッドが
上回ったとみられる。TensorRTの本来の強みはFP16/INT8量子化とカーネル融合に
あり、それを試せなかった今回のFP32比較では効果を測れていない。

**結論**: §12-1(TensorRT変換後の実推論FPS)は**依然「未検証」のまま**。
本PCで分かったのは (1) 変換パイプライン自体はONNX経由で動く、
(2) FP16化はTensorRT11+Windowsの組み合わせでは追加対応が必要、
(3) FP32のままではTensorRT化のメリットが出ない、という3点。Jetsonでの
実測ではJetPack同梱のTensorRT（多くの場合10.x系）でFP16/INT8が素直に
効く可能性があり、本PCの結果からは楽観・悲観どちらの結論も出せない。

### 14.10 Python 3.10への切り替え（§11-3対応）[実測]

追記日: 2026-07-22

Jetson移行を見据え、開発環境を JetPack 6系の標準Python（3.10）に合わせた。
`git`管理下に無いプロジェクトのため、切り替え前に `pyproject.toml` /
`uv.lock` / `.python-version` を `.backup_py312_migration/` へバックアップ
した上で作業（何か問題があれば3ファイルを元に戻して `uv sync` すれば
Python 3.12環境へロールバック可能）。

**変更内容**:
```diff
- requires-python = ">=3.12"
+ requires-python = ">=3.10,<3.11"
```
```diff
- numpy>=2.5.0
+ numpy>=2.2.6,<2.3
```
```diff
- onnxruntime-gpu>=1.27.0
+ onnxruntime-gpu>=1.23.2,<1.24
```
`.python-version` も `3.12.12` → `3.10.19` に変更。

**Python 3.10非対応で引っかかった依存関係**（`uv sync`のエラーから判明）:

| パッケージ | 問題 | 対応 |
|---|---|---|
| `numpy` | 2.3.0以降 `cp310` wheelなし | `>=2.2.6,<2.3` に固定（2.2.6が3.10対応の最終版） |
| `onnxruntime-gpu` | 1.24.0以降 `cp310` wheelなし | `>=1.23.2,<1.24` に固定 |

それ以外（`torch==2.12.1+cu126`, `torchvision==0.27.1+cu126`, `opencv-python`,
`pyside6`, `pypylon`, `onnx`, `onnxslim`, `tensorrt`/`tensorrt-cu13`系, `ultralytics`,
`lap`, `duckdb`, `hidapi`, `pyserial`）は無修正で `uv sync` が通った
（`tensorrt-cu13-bindings` は事前に `cp310` wheelの存在をPyPIで確認済み）。

**動作確認**（Python 3.10.19のvenv上）:
- `module_cameras_5goki.py` / `module_motor_serial.py` / `module_relay.py` /
  `module_patlite.py` / `telemetry_sources.py` / `dcr_logger.py` /
  `module_yolo.py` の import が全て成功
- `torch.cuda.is_available() == True`（CUDA版torchが維持されている。
  `pyproject.toml`内のコメントで警告されているCPU版への意図しない後退は
  起きていない）
- `bench/bench_hotpath.py` が正常に完走（4台合計28.34ms、cv2=32相当）
- `YOLO('Trained_Models/v4_11s.pt').to('cuda:0')` での実推論が成功

**まだ確認していないこと**: `PySide6` を使ったGUIの実起動（`main_5goki_JP_v3.py`
本体）、実カメラ(pypylon)接続時の動作、Arduino/シリアル通信。次にGUIを
実際に起動して一通りの画面遷移を確認するのが次のステップ候補。

---

## 15. 実稼働ログによる裏付け（2026-07-22 実機データ）[実測]

§14 までは bench スクリプトによる**合成負荷**の計測だった。2026-07-22 に
**実カメラ・実サクランボで 2 セッション（計 388 秒 / 275 個体）を稼働**させ、
`logs_cycle_5goki` / `logs_health_5goki` を採取した。解析結果は
[analysis/report_20260722_jetson.md](analysis/report_20260722_jetson.md) にまとめてある。

本メモの結論に対する影響:

| §    | 従来の記述 | 実稼働データによる更新 |
|---|---|---|
| §4.3 | 総計 2.3〜3.2 GB（ライブラリ 234 MB 前提） | **プロセス RSS 実測 1965 MB**。総計は 4.0〜4.7 GB へ上方修正。8 GB には収まるが余裕は 3.3〜4.0 GB |
| §5 | USB 帯域 83 MB/s [計算] | **4台 20.0 fps・エラー 0・ドロップ 0 で 300 秒連続稼働**。実稼働で裏付け |
| §3・§6・§8・§10-2 | 「HSVはGUIスレッドの50ms QTimer(`update_video_feeds`)が抱える」 | **前提が誤り。既に解消済み**。`module_yolo.py:848` `_camera_worker` がカメラごとに1本立ち、HSV→前処理→推論→後処理→描画までワーカーで実行して `frame_ready` でGUIへ渡す。GUIスレッドに残るのは QImage 化と `setPixmap` のみ（4台で3.16 ms）。**§10-2 の改善案は実装済み**。なお `module_main_window_JP.py:384` の `QTimer` は `timeout` 未接続の死んだコード |
| §6.2 | GUIスレッド 26 ms / 50 ms 予算 | 締切の正体は GUI ではなく**カメラのフレーム周期（FPS=20 → 50 ms）**。カメラ別ワーカー1本が1フレームを50 ms以内に処理する必要がある。**帯内フレームは実測 44.42 ms = 予算の 89 %**（HSV 6.24 / 前処理 0.62 / 推論 36.08 / 後処理 1.48）。締切超過はクラッシュせず「推論枚数が減る」形で現れる（`get_next_frame` が最新フレームのみ返すため） |
| §10 の優先順位 | HSV最適化（#2 #3）が最優先 | **組み替えが必要**。帯内フレームのうち HSV は 6.24 ms（14 %）で、推論まわりが 38.2 ms（86 %）。加えて `module_yolo.py:944` の `_infer_lock` が4カメラの推論を直列化しており、システム全体の推論能力は 27.7 回/秒に制限されている |
| §12-1 | TensorRT 化が最大の未確定要素 | **優先度が下がった**。GPU 使用率は実稼働で 11.5 %（最大 37 %）・40 W・55 ℃ しかなく、FP32 のままで GPU は律速していない |
| §14.8 | 純GPU推論 8.4〜8.7 ms/枚 | **実アプリの `infer_latency` は 36.08 ms/回**。差の 27.7 ms（77 %）は Python/ultralytics 側のオーバーヘッドで、**TensorRT では消えない**。§10 に「推論呼び出しのPython側オーバーヘッド削減」を追加すべき |
| §14.10 | Python 3.10 への切替は「動作確認済み」 | **要検証**。同一モデル・同一GPUで `infer_latency` が 06-23 の 14.52 ms → 07-22 は 36.08 ms（**2.5倍**）に悪化している。Python 3.10 化 / tensorrt等の追加 / 検出数の増加（4.6→12.5個）のどれが効いているか未切り分け。Jetson へ移す前に決着させること |
| §11-1 | `gpu_stats()` の NVML 依存を要修正 | **`cpu_temp_c()` も壊れている**（545 ℃ / 110.9 ℃ / 室温以下の値）。CPU 側テレメトリも同時に作り直す |

**結論の方向は変わらない**（ボトルネックは GPU ではなく CPU）が、**場所が違った**。
GUIスレッドではなく**カメラ別ワーカーの推論パス**である。
§10 の改善案 #2（4スレッド並列）は既に実装済みで、#3（半解像度HSV）だけでは
帯内フレーム 44.42 ms → 40.45 ms にしかならない（×3換算で 103 ms、予算の 2 倍）。

**新しい最優先事項**:
1. §4.2 の推論2.5倍悪化を解消する（0623 の 14.52 ms に戻れば帯内フレームは 22.86 ms＝予算の 46 %）
2. 推論の Python 側オーバーヘッド 27.68 ms を削る
3. `_infer_lock` による4カメラ推論の直列化を解く

---

## 付録: 主要な参照先

| 内容 | ファイル:行 |
|---|---|
| カメラキャプチャスレッド | `module_cameras_5goki.py:119` |
| カメラのFPS定数 / MaxNumBuffer | `module_cameras_5goki.py:22` / `:92` |
| 帯域制限の設定（pfsに上書きされる） | `module_cameras_5goki.py:94-104` |
| 遅延キューの実装 | `module_cameras_5goki.py:164-174` |
| YOLO推論ワーカー | `module_yolo.py:388` |
| HSV判定 `get_target_info` | `module_yolo.py:259` |
| 推論結果の適用（単一スレッド必須） | `module_yolo.py:624` |
| ホットパス `evaluate_frame` | `module_yolo.py:699` |
| 画像保存（コメントアウト中） | `module_yolo.py:212` / `:251` |
| GUI更新ループ | `main_5goki_JP_v3.py:415` |
| QThreadPool | `main_5goki_JP_v3.py:249` |
| ヘルスログ（バックグラウンド実行） | `main_5goki_JP_v3.py:485` |
| tracemalloc起動 | `main_5goki_JP_v3.py:338` |
| 遅延設定の読み込み | `main_5goki_JP_v3.py:387` |
| GPUテレメトリ（NVML依存） | `telemetry_sources.py:6` |
