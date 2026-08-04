# Jetson Orin Nano Super 購入判断 検証ランブック

実験しながらこのファイルを開いて、上から順にチェックを付けていく手順書。
コマンドはすべて**リポジトリルートで**実行する。

- 検証プラン（何を測るかの理論）: `jetson_purchase_validation_plan.md`
- 既存の調査記録: [JETSON_MIGRATION_NOTES.md](JETSON_MIGRATION_NOTES.md)
- スクリプト本体: `standalone/` 配下（各ファイル先頭の docstring に詳細）

---

## 0. 30秒で分かる全体像

| Phase | 何をするか | ライン運転 | 所要時間 | 出力 |
|---|---|---|---|---|
| **0** | 事前準備・環境確認 | 不要 | 15分 | — |
| **1** | 静的確認（並列度・スレッド構成） | 不要 | 5分 | `task1_report.md` |
| **2** | **Task 3** 半解像度の精度検証 | **必要** | 8分運転＋確認5分 | `task3_*_live_summary.json` |
| **3** | **Task 1/2** 基準取り（制限なし） | **必要** | 10分運転 | `baseline_live_summary.json` |
| **4** | **Task 1/2** Jetson相当（制限あり） | **必要** | 10分運転 | `jetson_live_summary.json` |
| **5** | 換算・GO/NO GO 判定 | 不要 | 10分 | `task2_projection.json` |

**合計 約1.5時間**（ライン運転は合計28分）。サクランボを流し続ける人員が必要。

> **なぜこの順番か**: Task 3 が不合格だと半解像度化が使えず、`1個体=287ms` という投影の前提が丸ごと崩れる。
> Task 1/2 を先にやっても無駄になるので、**必ず Task 3 から**。

---

## 1. Phase 0: 事前準備

### 1-1. チェックリスト

- [ ] リポジトリルートに居る（`main.py` が見えること）
- [ ] `uv sync` 済み
- [ ] Basler カメラ4台が接続され、`cam_pfs/*.pfs` と一致している
- [ ] **他のカメラ使用アプリを全部閉じた**（`hsv_calibration.py` / 本番の別インスタンス / Pylon Viewer）
- [ ] サクランボを連続で流せる状態（Task 3 だけで100個体以上必要）
- [ ] 電源をACに接続、ブラウザ等の余計なアプリを閉じた
- [ ] `psutil` が入っている → `uv run python -c "import psutil; print(psutil.__version__)"`
  - 入っていなければ `uv add psutil`（ultralytics 経由で入っていることが多い）

### 1-2. クロック制限の権限を確認する（Phase 4 で必要）

**Windows**: 管理者権限の PowerShell を開いておく。以下が通ればOK。

```powershell
powercfg /getactivescheme
```

**Linux**: `cpupower` が使えること。

```bash
sudo cpupower frequency-info | head -5
```

> 権限が無い場合でも Phase 4 は実行できるが、**コア数制限だけ**になりクロックは下がらない。
> その場合 k が過小評価になるので、レポートの `clock.applied` を必ず確認すること（§6-2 参照）。

### 1-3. 出力先

すべて `analysis/jetson_validation/` に出る。`analysis/` は `.gitignore` 済みなのでリポジトリは汚れない。

```bash
mkdir -p analysis/jetson_validation
```

---

## 2. Phase 1: 静的確認（ライン不要・5分）

```bash
uv run python standalone/profile_cpu_usage.py --static-only --tag task1_static
```

**確認すること**: `analysis/jetson_validation/task1_static_report.md` を開いて

- [ ] `explicit_setNumThreads_in_code` の値を記録する
  - **`False` なら要注意**。notes §9 が「最優先の必須対策」とする `cv2.setNumThreads(1)` が本番コードに入っていない
  - Phase 4 の `run_as_jetson.py --preset jetson` はこれを実行時に強制するので、検証自体は進められる
- [ ] `ultralytics_import_side_effect` を記録する（`True` なら import だけでスレッド数が1になっている＝notes §14.1 の再現）
- [ ] 「OpenCV ビルド構成」の表を記録する
  - **Jetson実機と比較する際に必須**。JetPack 同梱の OpenCV は CUDA 無効ビルドで並列フレームワークも違う。
    両方の出力を残しておかないと、後で「どっちの差なのか」が判別できなくなる

---

## 3. Phase 2: Task 3 — 半解像度の精度検証（運転8分）

### 3-1. なぜ `--preset baseline` で走らせるのか

**検出結果はCPUの速さに依存しない**（同じフレームに同じ演算をするだけ）ので、制限をかける意味がない。
むしろ制限をかけると**検証スレッドがフレーム流入に追いつかず、窓が全部除外される**。

- 流入: 4カメラ × 20fps = 80フレーム/秒
- 検証1フレームの全解像度HSV: 制限なしで約8ms → 0.64コア分（追いつく）
- Jetson相当の制限下では約22ms → 1.8コア分（**単一スレッドでは追いつかない**）

### 3-2. 実行

```bash
uv run python standalone/run_as_jetson.py --preset baseline --verify-scale \
    --verify-window 30 --verify-period 45 --tag task3
```

- [ ] コンソールに `[probe] Task 3 検証を有効化` が出た
- [ ] **GUIのトグルスイッチをONにする**（ラベルが「停止中」→「動作中」）
  - > **これを忘れると何も測れない**。停止中はカメラ映像を表示するだけで、
  > `get_target_info` が一度も呼ばれない（`module_yolo._camera_worker`）
- [ ] サクランボを**止めずに**流し続ける
- [ ] **8分**運転する（下記の根拠を参照）
- [ ] GUIを閉じる → 自動で集計してレポートが出る

### 3-3. 運転時間の根拠

実稼働ログ（notes §15）の実測から **1個体あたり約1.4秒**（388秒で275個体）。

| 項目 | 計算 |
|---|---|
| 検証窓1つ(30秒)で取れる個体数 | 30 ÷ 1.4 ≒ **21個体/カメラ** |
| 合格基準に必要な個体数 | **100個体**（プラン §Task 3） |
| 必要な有効窓数 | 100 ÷ 21 ≒ **5窓** |
| 45秒周期での所要時間 | 5 × 45 = 225秒 |
| キュー溢れで除外される窓を見込んだ余裕 | **8分**（約10窓） |

> 個体数は**サクランボを流す速度に完全に依存する**。流れが遅ければそのぶん延長すること。
> 最後に出る `個体=` の数字が100未満なら、時間を延ばして測り直す。

### 3-4. 結果の見方

コンソール末尾（と `analysis/jetson_validation/task3_live_summary.json`）:

```
[Task 3] 縮小率の比較（基準1.0 vs 本番）
          cam_top      合格  一致率=0.9987  個体=112 取りこぼし=0 遅延中央値=0.0
```

- [ ] 4カメラすべて **合格** か
- [ ] `個体=` が **100以上** か（未満なら統計的に無意味。延長して再測定）
- [ ] `除外した窓` が出ていないか（出ていれば、その窓はキュー溢れで捨てられている）

| 判定 | 次の行動 |
|---|---|
| 全カメラ合格 | **Phase 3 へ進む** |
| 不合格あり | **ここで止まる。** Task 1/2 をやっても投影の前提が崩れているので無意味。まず面積閾値の再チューニング（§7-2）へ |
| 判定不能 | 基準側の検出が0個体。トグルON忘れ・果実が流れていない・カメラ名の取り違えを疑う |

### 3-5. この実行の Task 1/2 の数値は使わないこと

同じ実行で Task 1/2 の値も出るが、**検証スレッドがCPUを食っているので信用できない**。
プローブのCPU占有率が10%を超えると、レポート末尾に自動で警告が出る。

```
[!] 計測プローブがCPU時間の 45% を使っている。...
```

Task 1/2 の数値は Phase 3/4 の実行から取る。

---

## 4. Phase 3: Task 1/2 の基準取り（制限なし・運転10分）

```bash
uv run python standalone/run_as_jetson.py --preset baseline --tag baseline
```

- [ ] `--verify-scale` を**付けない**（Task 3 の負荷を混ぜないため）
- [ ] GUIのトグルをONにする
- [ ] **10分**運転する（最低5分）
- [ ] Phase 4 と**同じ流速・同じ運転時間**にする ← k の比較が成立する条件
- [ ] GUIを閉じる

### 運転時間の根拠

| 運転時間 | 取れる個体数 | 用途 |
|---|---|---|
| 5分 | 約215個体 | 最低ライン |
| **10分** | **約430個体** | **推奨**。p50が安定し、流速のムラも平均化される |

### 確認

```bash
uv run python standalone/analyze_cycle_logs.py --run analysis/jetson_validation/baseline_live_summary.json
```

- [ ] `個体数` が200以上ある
- [ ] `1個体あたり合計` の p50 を記録シート（§8）に書く
- [ ] `CPU側合計(p50)` を記録する ← **これが Jetson 換算で k を掛ける対象**

---

## 5. Phase 4: Task 1/2 の Jetson相当実行（運転10分）

### 5-1. 実行（Windowsは管理者PowerShellで）

```bash
uv run python standalone/run_as_jetson.py --preset jetson --tag jetson
```

適用される制限:

| 項目 | 値 | 根拠 |
|---|---|---|
| コア数 | 6 | Orin Nano Super は Cortex-A78AE 6コア |
| クロック | 31% (Win) / 1700MHz (Linux) | A78AE 約1.7GHz ÷ i9ブースト約5.4GHz |
| `cv2.setNumThreads` | 1 | notes §9 の必須対策 |
| `torch.set_num_threads` | 2 | 同上 |
| `OMP_NUM_THREADS` | 2 | OpenMPプールは import 時に確定するため事前に設定 |

- [ ] 起動直後の表示で **`クロック状態 : プロセッサ最大状態 31%`**（または `CPU上限 1700 MHz`）を確認
  - `管理者/root 権限が無いため...` と出たら、権限付きで起動し直す
- [ ] **復元コマンドの表示をメモ**（強制終了した場合に手で戻すため）
- [ ] GUIのトグルをONにする
- [ ] **Phase 3 と同じ10分**運転する
- [ ] GUIを閉じる → `[clock] 復元しました` を確認

### 5-2. 終了後に必ず確認

- [ ] コンソールに `[clock] 復元しました` が出た
- [ ] 出ていない場合、表示された復元コマンドを手で実行する

**Windows の復元コマンド**（管理者PowerShell）:
```powershell
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100
powercfg /setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100
powercfg /setactive SCHEME_CURRENT
```

**Linux の復元コマンド**:
```bash
sudo cpupower frequency-set -u 5400MHz   # 実機の最大クロックに読み替える
```

> **戻し忘れると以降の学習・計測が全部遅くなる。** 実験の最後に
> `Get-Counter '\Processor Information(_Total)\% Processor Performance'`（Win）や
> `cpupower frequency-info`（Linux）でクロックが戻っていることを確認する。

---

## 6. Phase 5: 換算と GO/NO GO 判定（10分）

### 6-1. 工程別のスケール係数を出す

```bash
uv run python standalone/analyze_cycle_logs.py \
    --ref  analysis/jetson_validation/baseline_live_summary.json \
    --proxy analysis/jetson_validation/jetson_live_summary.json
```

出力例:

```
  工程                         基準[ms]     制限[ms]       k  種別
  HSV検出                      159.66     433.00   2.712  CPU
  推論                         431.68     430.12   0.996  GPU
  CPU側合計の k = 2.710（226.4 → 613.5 ms）
  全体の k       = 1.585（GPUが薄めるので、判定にはCPU側の k を使うこと）
```

### 6-2. 数値の妥当性チェック（ここを飛ばすと誤った結論が出る）

- [ ] **推論の k が 1.0 付近か**
  - 1.0付近 → 制限が意図通りCPUだけに効いている。正常
  - 1.5超 → 推論のPython/ultralytics側オーバーヘッドが支配的（notes §15 と同傾向）。
    この場合、推論もCPU側として扱う必要があるため、報告に明記する
- [ ] **CPU側の k が 1.15 未満になっていないか**
  - なっていたら**制限が効いていない**。クロック設定の適用失敗を疑う。
    `jetson_live_summary.json` の `clock.applied` と `env.cpu_freq_mhz` を両実行で比較する
- [ ] 両実行の**個体数が同程度**か（片方だけ極端に少ないと比較にならない）
- [ ] **全体の k を使っていないか**（GPUが薄めるので必ず CPU側の k を使う）

### 6-3. Jetson換算と最終判定

前ステップの出力末尾に、そのまま貼れるコマンドが表示される。

```bash
uv run python standalone/project_jetson.py --k-measured <CPU側のk> --cpu-ms-pc <基準のCPU側ms>
```

判定基準:

| k_estimated（ISA補正後） | 判定 | アクション |
|---|---|---|
| ≤ 4.0 | **GO** | 安全マージン25%以上。購入可 |
| 4.0 〜 5.4 | **条件付きGO** | VPIによるGPUオフロード（プラン Appendix A）を前提に購入 |
| > 5.4 | **NO GO** | 現構成では目標未達。再設計 or Orin NX 検討 |

- [ ] `[注意] ISA補正の係数で判定が変わる` が出た場合、**判定を幅で報告する**
  - 例:「×1.0なら GO、×1.5なら NO GO」。ISA補正1.4自体が推定値なので、片方だけ書くと誤解を招く

---

## 7. 気をつけること（実験中に迷ったら）

### 7-1. 落とし穴トップ7

| # | 落とし穴 | 症状 | 対処 |
|---|---|---|---|
| 1 | **GUIのトグルをONにし忘れる** | 個体数0・HSV呼び出し0回 | ラベルが「動作中」か確認。停止中は `get_target_info` が一度も呼ばれない |
| 2 | **Task 3 と Task 1/2 を同じ実行で取る** | Task 1/2 の値が悪化して見える | 別々に実行する。プローブ占有率10%超で自動警告が出る |
| 3 | **Task 3 を `--preset jetson` で走らせる** | 窓が全部 `excluded_windows` になる | Task 3 は必ず `--preset baseline` |
| 4 | **クロック制限の権限不足に気づかない** | k が 1.2 程度にしかならない | `clock.applied` を確認。管理者権限で起動し直す |
| 5 | **クロックを戻し忘れる** | 以降の学習が全部遅い | 正常終了なら自動復元。強制終了時は §5-2 のコマンドを手で実行 |
| 6 | **カメラの排他アクセス** | `カメラを初期化できません` | 本番の別インスタンス・`hsv_calibration.py`・Pylon Viewer を閉じる |
| 7 | **個体数が100未満のまま判定する** | 取りこぼし率1%の判定が無意味 | 運転時間を延ばす。100個体＝約2.5分ぶんの連続した流し |

### 7-2. Task 3 が不合格だったら

半解像度化が使えないので、**Task 1/2 に進んではいけない**。次の順で潰す。

1. 原因を見る: レポートの「不一致フレームの原因」表
   - `edge` が多い → `module_yolo.py:66` の既知事項（OPEN/CLOSEカーネルが5x5固定で
     scale連動しないため、縮小画像では端接触判定が逆転しうる）
2. 面積閾値を再チューニングする（録画した動画に対して）

   ```bash
   uv run python standalone/verify_hsv_scale_video.py --video <動画> --cam cam_top \
       --sweep-area 0.6,0.8,1.0,1.2,1.5
   ```
   F1が最大の倍率を `json/hsv_config_{cam}.json` の `min_blob_area` に反映して再検証する
3. それでも駄目なら縮小率を上げる（0.5 → 0.7）。ただしCPU削減効果は落ちる

### 7-3. 「Jetson相当」にできないもの（結論の解釈に必ず添える）

| 項目 | 再現可否 | 扱い |
|---|---|---|
| コア数・クロック・スレッド数 | **できる** | 実測 k に反映済み |
| ISA（AVX2 256bit → NEON 128bit） | できない | `--isa-factor`（既定1.4）で安全側に補正。**この係数自体が推定値** |
| GPU | できない | アーキテクチャもSM数も別物。公表実測値 7.17ms/回 で合成 |
| メモリ8GB共有 | 再現しない | 絞ってもOOMで落ちるだけ。RSSを記録して余裕を報告（プラン Appendix B） |
| サーマルスロットリング | できない | 実機でのみ確認可。25W連続・夏季30℃超の条件（プラン Appendix B） |

---

## 8. 記録シート（実験しながら埋める）

**実施日**: ____________  **実施者**: ____________

### Phase 1 静的確認

| 項目 | 値 |
|---|---|
| `explicit_setNumThreads_in_code` | |
| `ultralytics_import_side_effect` | |
| OpenCV Parallel framework | |
| 論理コア数 | |

### Phase 2 Task 3

| カメラ | 判定 | 一致率 | 個体数 | 取りこぼし | 遅延中央値 | 遅延P95 |
|---|---|---|---|---|---|---|
| cam_top | | | | | | |
| cam_under | | | | | | |
| cam_inside | | | | | | |
| cam_outside | | | | | | |

除外された窓: ____________  → **総合判定: 合格 / 不合格**

### Phase 3・4 Task 1/2

| 工程 | 基準[ms] | 制限[ms] | k |
|---|---|---|---|
| 撮影 | | | |
| HSV検出 | | | |
| 前処理 | | | |
| 推論(GPU) | | | |
| 後処理 | | | |
| **CPU側合計** | | | |
| 1個体あたり合計 | | | |

| 項目 | 基準 | 制限 |
|---|---|---|
| 個体数 | | |
| 運転時間 | | |
| CPU使用率 p50 | | |
| RSS ピーク[MB] | | |
| クロック実測[MHz] | | |
| `clock.applied` | — | |

### Phase 5 判定

| 項目 | 値 |
|---|---|
| k_measured（CPU側） | |
| k_estimated（×1.4） | |
| Nano Super 換算[ms] | |
| マージン[%] | |
| **判定** | **GO / 条件付きGO / NO GO** |
| 感度（×1.0 / ×1.3 / ×1.5） | / / |

---

## 9. Plan B: ラインを動かせない場合

実機を運転できないときは、マイクロベンチと録画動画で代替する。
**実アプリの競合が入らないぶん楽観的な値が出る**ので、結論には必ずその旨を添える。

```bash
# Task 1: 実行中プロセスを外から監視（本番を別ターミナルで起動しておく）
uv run python standalone/profile_cpu_usage.py --duration 60

# Task 2: HSV単体のマイクロベンチ（基準）
uv run python standalone/bench_hsv_jetson.py --tag ref_full --breakdown
#   ↓ ここでクロック制限を掛ける（§5-1 と同じ）
uv run python standalone/bench_hsv_jetson.py --cores 6 --cv-threads 1 --tag jetson_proxy
uv run python standalone/project_jetson.py \
    --ref  analysis/jetson_validation/bench_hsv_ref_full.json \
    --proxy analysis/jetson_validation/bench_hsv_jetson_proxy.json

# Task 3: 録画動画で検証
uv run python standalone/verify_hsv_scale_video.py --video <動画> --cam cam_top
```

所要時間: 全部で約15分（クロック制限の切替を含む）。

実画像がある場合はベンチに渡すこと（無ければ擬似フレームに自動フォールバックする）:

```bash
uv run python standalone/bench_hsv_jetson.py --image-dir training_images/healthy --limit 50 --tag ref_full
```

---

## 10. 判定フロー

```
Phase 2: Task 3
    │
    ├─ 不合格 ──→ §7-2 で閾値再チューニング ──→ 再検証（ここを抜けるまで先に進まない）
    │
    └─ 合格
         ↓
Phase 3/4: Task 1/2（基準 → 制限）
         ↓
    §6-2 の妥当性チェック
         │
         ├─ 推論の k > 1.5   ──→ 推論もCPU側として扱い、報告に明記
         ├─ CPU側 k < 1.15   ──→ 制限が効いていない。クロック設定を直して再測定
         │
         └─ 正常
              ↓
Phase 5: project_jetson.py
              ↓
    ┌─────────┼─────────┐
  k≤4.0    4.0<k≤5.4    k>5.4
   GO      条件付きGO    NO GO
    │          │           │
  購入可   VPIオフロード  再設計 or
           前提で購入     Orin NX 検討
```

---

## 11. 参照

| 内容 | ファイル |
|---|---|
| 検証プラン（理論・判定式の導出） | `jetson_purchase_validation_plan.md` |
| 既存の調査記録（実測値の蓄積） | `JETSON_MIGRATION_NOTES.md` |
| HSV検出の本体 | `module_yolo.py:269` `ImageProcessor.get_target_info` |
| 縮小率の設定と採用根拠 | `module_yolo.py:73` `HSV_DETECT_SCALE` |
| HSVマスクの共通実装 | `hsv_mask_utils.py` |
| 1個体あたりコストのログ定義 | `dcr_logger.py:24` `CYCLE_COLUMNS` |
| カメラワーカー（運転中のみ動く） | `module_yolo.py:696` `_camera_worker` |
