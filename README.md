# DCRsystem5goki_app_base

サクランボ選別ライン向けの外観検査・自動仕分けシステム。4台のカメラでさくらんぼを撮影し、
YOLO による物体検出で健全 / 障害を二値判定、判定結果に応じてリレー経由で仕分け弁を駆動する。

## システム構成

```
main.py                      実行専用エントリポイント（GUI起動のみ）
module_main_window_JP.py     ウィンドウ・制御ロジック本体（main.py から分離）
module_gui_JP.py             GUI デザイン定義（レイアウト・カスタムウィジェット）
module_cameras_5goki.py      Basler カメラ（pypylon）制御
module_yolo.py               YOLO 推論・状態機械（個体追跡〜健全/障害の二値判定）
module_motor_serial.py       Arduino とのシリアル通信（ターンテーブル制御）
module_relay.py              リレーボード制御（判定結果に応じた仕分け弁の駆動）
module_patlite.py            パトライト制御（システム状態の表示）
hsv_mask_utils.py            HSV マスク生成・整形の共通処理（本番・校正ツール間で共有）
dcr_logger.py                構造化データログ（cycle / health / events / detections）
log_config.py                コンソール・ファイルログの統一設定
telemetry_sources.py         health スレッド向けテレメトリ取得ヘルパー
```

### 判定の流れ

1. `module_cameras_5goki.py` が4カメラ（top / under / inside / outside）から映像を取得
2. `module_yolo.py` が HSV マスクで果実領域を検出し、中心窓に入ったフレームのみ YOLO 推論
3. ByteTracker でフレーム間の個体を追跡し、個体が確定した時点で健全/障害を二値判定
   （障害系クラスは1カメラでも検出があれば即座に障害、健全は複数カメラでの一致を要求）
4. `module_relay.py` が判定結果に応じて仕分け弁（運搬弁 / 除去弁）を駆動
5. `dcr_logger.py` が cycle・detections 等をログへ記録

### standalone/ 配下（校正・検証ツール）

本番とは独立して実行できる GUI ツール群。

| ファイル | 用途 |
|---|---|
| `hsv_calibration.py` | 果実検出（HSV存在検出＋推論ゲート）の閾値をスライダーで調整し `json/hsv_config_{cam}.json` へ保存 |
| `relay_calibration.py` | リレー（仕分け弁）の開弁タイミングをスライダーで実機に合わせ込む |
| `delay_calibration.py` | カメラ表示遅延（delay_seconds）の実測キャリブレーション |
| `hsv_ripeness_classifier.py` | 赤色占有率スコアによる healthy/unripe 振り分けGUI（学習データ整備用） |
| `classification_gui_demo.py` | 実機なしで個体確定〜健全/障害判定の流れを確認するデモ |

## セットアップ

Python 3.10 系、パッケージ管理は [uv](https://docs.astral.sh/uv/) を使用する。

```bash
uv sync
```

torch / torchvision は CUDA (cu126) ビルドを明示的に指定しているため、`uv sync` だけで
GPU 対応版が入る（CPU 版が入ると YOLO 推論が大幅に遅くなるため要注意）。

## 実行

```bash
uv run main.py
```

## 補足ドキュメント

- [JETSON_MIGRATION_NOTES.md](JETSON_MIGRATION_NOTES.md) — Jetson Orin Nano Super への移行検証メモ
