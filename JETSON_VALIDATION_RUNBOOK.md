# Jetson購入判断 検証ランブック（3本比較版）

実験しながらこのファイルを開いて、上から順にチェックを付けていく手順書。
コマンドはすべて**リポジトリルートで**実行する。

前回（0804/0805）までの実験は情報量が多くなりすぎたため整理した。
今回は目的を2つの比較に絞る。

- **比較①** 半解像度の有無による処理速度の違い（PC制限なしで実行）
- **比較②** 半解像度あり＋制限なしの結果と、半解像度あり＋Jetson相当の制限をかけた結果

半解像度化そのものの検出精度（ゲート通過率）は前回の実験で合格が確認済みのため、
今回は再検証しない。

---

## 0. 30秒で分かる全体像

3本とも「本番アプリを実際に運転してログを取る」実験。1本あたり運転8〜10分。

| # | tag | 縮小率 | PC制限 | 何と比べるか |
|---|---|---|---|---|
| ① | `s10_baseline` | 1.0（半解像度なし） | なし | ②との比較で「半解像度の効果」 |
| ② | `s05_baseline` | 0.5（半解像度あり） | なし | ①・③の両方と比較する基準点 |
| ③ | `s05_jetson`   | 0.5（半解像度あり） | Jetson相当（6コア+クロック31%） | ②との比較で「PC制限の効果」 |

**合計 約30分**（運転時間のみ。前後の準備・集計は別途）。

---

## 1. 事前準備

- [ ] リポジトリルートに居る（`main.py` が見えること）
- [ ] `uv sync` 済み
- [ ] Basler カメラ4台が接続され、`cam_pfs/*.pfs` と一致している
- [ ] 他のカメラ使用アプリを全部閉じた（`hsv_calibration.py` / 本番の別インスタンス / Pylon Viewer）
- [ ] サクランボを連続で流せる状態（1本あたり最低200個体、目安8〜10分）
- [ ] `psutil` が入っている → `uv run python -c "import psutil; print(psutil.__version__)"`
- [ ] `module_yolo.py` の `HSV_DETECT_SCALE` が **0.5**（本番採用値）になっていることを確認
      → 各実行は `run_as_jetson.py --scale` が起動前に上書きするので、ファイル上の値そのものは
        気にしなくてよいが、他の作業と混ざっていないことの確認として見ておく
- [ ] ③（Jetson相当）は管理者権限が必要。管理者PowerShellを用意しておく
      （`powercfg /getactivescheme` が通ればOK）

```powershell
mkdir -p analysis/jetson_validation   # 無ければ作成（.gitignore済みでリポジトリは汚れない）
```

---

## 2. 実行

### 2-1. ①半解像度なし・PC制限なし（基準）

```bash
uv run python standalone/run_as_jetson.py --scale 1.0 --preset baseline --tag s10_baseline
```

- [ ] コンソールに `HSV_DETECT_SCALE を 1.0 に設定しました` が出た
- [ ] **GUIのトグルスイッチをONにする**（ラベルが「停止中」→「動作中」）
  > これを忘れると何も測れない。停止中はカメラ映像を表示するだけで `get_target_info` が呼ばれない
- [ ] サクランボを止めずに **8〜10分** 流し続ける
- [ ] GUIを閉じる → 自動で集計してレポートが出る

### 2-2. ②半解像度あり・PC制限なし

```bash
uv run python standalone/run_as_jetson.py --scale 0.5 --preset baseline --tag s05_baseline
```

- [ ] ①と**同じ流速・同じ運転時間**にする（比較が成立する条件）
- [ ] GUIのトグルをON → 8〜10分運転 → GUIを閉じる

### 2-3. ③半解像度あり・Jetson相当の制限あり（管理者PowerShellで）

```bash
uv run python standalone/run_as_jetson.py --scale 0.5 --preset jetson --tag s05_jetson
```

- [ ] 起動直後の表示で `クロック状態` を確認
  - `管理者権限が無いため...` と出たら、管理者権限で起動し直す
- [ ] **復元コマンドの表示をメモ**（強制終了した場合に手で戻すため）
- [ ] ②と**同じ運転時間**（8〜10分）
- [ ] GUIを閉じる → `[clock] 復元しました` を確認
  - 出ていない場合、表示された復元コマンドを手で実行する

```powershell
# 手動復元コマンド（強制終了時のみ使う）
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100
powercfg /setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100
powercfg /setactive SCHEME_CURRENT
```

### 2-4. 各実行の直後に確認すること

`analysis/jetson_validation/{tag}_live_summary.json` を開いて、

- [ ] `hsv_timing.by_cam` の4カメラすべてに `calls` が入っている（0ならトグルON忘れ）
- [ ] ③のみ: `clock_measured.effective_ghz_p50` を記録する
  - **自己申告の `clock.applied` だけでは足りない**。目標は Orin Nano Super 相当の
    約1.7GHz。前回の実測では `PROCTHROTTLEMAX 31%` を適用しても実効クロックは
    3.2GHz程度までしか落ちなかった（Turbo Boost/Speed Shift搭載CPUの既知の限界）。
    今回どこまで落ちたかを必ず見ること
- [ ] ③のみ: `cpu_usage.cores_equiv_p95` が 6 コア以内に収まっているか

---

## 3. 集計・比較

### 3-1. 比較① 半解像度の効果（s10_baseline vs s05_baseline）

```bash
uv run python standalone/analyze_cycle_logs.py \
    --ref  analysis/jetson_validation/s10_baseline_live_summary.json \
    --proxy analysis/jetson_validation/s05_baseline_live_summary.json \
    --tag compare_scale_effect
```

- [ ] `k_cpu_side` が 1 未満（半解像度の方が速い）になっているか確認
- [ ] 推論(GPU)の `k` が 1.0 近傍か確認（半解像度はHSVだけに効くはずなので、
      推論側が動いていたら別要因が混ざっている）

### 3-2. 比較② PC制限の効果（s05_baseline vs s05_jetson）

```bash
uv run python standalone/analyze_cycle_logs.py \
    --ref  analysis/jetson_validation/s05_baseline_live_summary.json \
    --proxy analysis/jetson_validation/s05_jetson_live_summary.json \
    --tag compare_jetson_effect
```

- [ ] **CPU側の k が 1.15 未満なら要注意**（制限が実質的に効いていない可能性）。
      `s05_jetson_live_summary.json` の `clock_measured.effective_ghz_p50` と比較すること
- [ ] 推論(GPU)の `k` が 1.0 近傍か確認（CPU制限がGPU側に漏れていないか）

### 3-3.（任意）Jetson実機への換算

比較②の `k_cpu_side` を実機換算まで進めたい場合:

```bash
uv run python standalone/project_jetson.py \
    --k-measured <比較②のk_cpu_side> \
    --cpu-ms-pc  <比較②のcpu_side_ref_ms>
```

判定基準・ISA補正の考え方は `standalone/project_jetson.py` の docstring を参照。

---

## 4. 気をつけること

| # | 落とし穴 | 症状 | 対処 |
|---|---|---|---|
| 1 | GUIのトグルをONにし忘れる | 個体数0・HSV呼び出し0回 | ラベルが「動作中」か確認 |
| 2 | 3本の運転時間・流速が揃っていない | kの比較が成立しない | 同じ時間・同じ流速で流す |
| 3 | ③を管理者権限なしで実行 | クロックが落ちない | 管理者PowerShellで実行し直す |
| 4 | クロック実測を見ずに `clock.applied` だけ信じる | 実は3割しか制限が効いていないのに気づかない | 必ず `clock_measured.effective_ghz_p50` を見る |
| 5 | クロックを戻し忘れる | 以降の学習が全部遅い | 正常終了なら自動復元。強制終了時は§2-3のコマンドを手で実行 |
| 6 | カメラの排他アクセス | `カメラを初期化できません` | 本番の別インスタンス・`hsv_calibration.py`・Pylon Viewer を閉じる |

---

## 5. 記録シート

**実施日**: ____________  **実施者**: ____________

| # | tag | 運転時間 | 個体数 | 1個体あたりp50[ms] | CPU側k | 推論k | 実効クロック(p50) |
|---|---|---|---|---|---|---|---|
| ① | s10_baseline | | | | — | — | — |
| ② | s05_baseline | | | | | | — |
| ③ | s05_jetson | | | | | | |

**比較①（半解像度の効果）**: ____________

**比較②（PC制限の効果）**: ____________

---

## 6. 参照

| 内容 | ファイル |
|---|---|
| HSV検出の本体 | `module_yolo.py` `ImageProcessor.get_target_info` |
| 縮小率の設定と採用根拠 | `module_yolo.py:73` `HSV_DETECT_SCALE`（本番採用値 0.5） |
| 実行スクリプト | `standalone/run_as_jetson.py` |
| 実測プローブ（CPU・クロック・HSV時間） | `standalone/jetson_live_probe.py` |
| 共有ヘルパー | `standalone/jetson_probe_common.py` |
| cycle ログの集計・比較 | `standalone/analyze_cycle_logs.py` |
| Jetson実機への換算 | `standalone/project_jetson.py` |
| 1個体あたりコストのログ定義 | `dcr_logger.py` `CYCLE_COLUMNS` |
