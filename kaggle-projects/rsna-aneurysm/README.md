# RSNA Intracranial Aneurysm Detection 🧠

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Kaggle RSNA 2025 脳動脈瘤検出コンペティション向けの最先端機械学習ソリューション**

## 🎯 プロジェクト概要

本プロジェクトは、医療画像（DICOM）から脳動脈瘤を検出するための包括的なMLパイプラインです。Google Colab GPU環境での実行を前提とした、再現性の高い実験管理システムを提供します。

### 主な特徴

- 🏗️ **階層型実験管理**: 1実験1ディレクトリ方式で完全な分離と再現性を確保
- 🔬 **医療画像特化**: DICOM処理、医療特有の前処理、窓幅/窓位設定対応
- ⚡ **GPU最適化**: Mixed Precision、TTA、マルチスケール学習をサポート
- 📊 **W&B統合**: 実験追跡、モデル比較、アーティファクト管理
- 🎛️ **柔軟な設定管理**: YAML設定によるハイパーパラメータと実験の完全管理
- 📈 **高品質CV**: 患者レベル分割、リーク検出、CV-LB相関監視

## 🏗️ アーキテクチャ

```
kaggle-projects/rsna-aneurysm/
├── configs/                   # 共通ベース設定
│   ├── data.yaml             # データ前処理設定
│   ├── model.yaml            # モデルアーキテクチャ設定
│   ├── train.yaml            # 学習パラメータ設定
│   ├── cv.yaml               # 交差検証設定
│   └── augmentation.yaml     # データ拡張設定
├── experiments/               # 実験管理
│   └── exp0001/              # ベースライン実験
│       ├── training.ipynb    # 学習ノートブック
│       ├── evaluation.ipynb  # 評価・分析ノートブック
│       ├── inference.ipynb   # 推論・提出ノートブック
│       ├── config.yaml       # 実験固有設定スナップショット
│       ├── model/            # 学習済みモデル
│       ├── submissions/      # Kaggle提出ファイル
│       └── env/              # 実験環境（requirements.lock）
├── data/                      # データ管理（DVC）
│   ├── raw/                  # 元データ（DICOM等）
│   ├── processed/            # 前処理済みデータ
│   └── external/             # 外部データ
├── scripts/                   # 共通モジュール
│   ├── dataset.py            # PyTorch Dataset
│   ├── model.py              # モデルアーキテクチャ
│   ├── transforms.py         # 画像変換・拡張
│   ├── utils.py              # ユーティリティ関数
│   ├── metrics.py            # 評価指標
│   └── dicom_utils.py        # DICOM処理
├── docs/                      # ドキュメント
│   ├── colab_setup.md        # Google Colab設定ガイド
│   └── experiment_workflow.md # 実験ワークフロー
└── experiments.csv            # 実験台帳
```

## 🚀 クイックスタート

### 1. Google Colab セットアップ

```python
# Google Colab で実行
!git clone https://github.com/YOUR_USERNAME/rsna-aneurysm-project.git
%cd rsna-aneurysm-project/experiments/exp0001

# 環境構築
!pip install -r env/requirements.lock

# GPU確認
import torch
print(f"GPU available: {torch.cuda.is_available()}")
```

### 2. データ準備

```bash
# Kaggleデータダウンロード
kaggle competitions download -c rsna-intracranial-aneurysm-detection -p ../../data/raw --unzip

# DICOMデータ前処理
python -m scripts.dicom_utils \
  --input ../../data/raw \
  --output ../../data/processed \
  --window-center 40 \
  --window-width 80 \
  --target-size 512 512
```

### 3. ベースライン実験実行

```python
# 実験ディレクトリに移動
%cd experiments/exp0001

# 順次実行
# 1. training.ipynb   - モデル学習
# 2. evaluation.ipynb - OOF評価・分析  
# 3. inference.ipynb  - テスト推論・提出
```

## 📊 実験管理ワークフロー

### 新実験の開始

1. **仮説・変更点の明確化**
2. **実験ディレクトリ作成** (`experiments/exp0002/`)  
3. **設定ファイル調整** (`config.yaml`)
4. **学習・評価・推論の実行**
5. **結果分析・次実験計画**

### 実験品質保証

- ✅ **再現性**: Git SHA、固定シード、環境ロック
- ✅ **トレーサビリティ**: W&B統合、実験台帳管理
- ✅ **品質監視**: CV-LB相関、リーク検出、Fold一貫性

## 🛠️ 主要機能

### 医療画像処理
- DICOM読み込み・前処理
- 窓幅/窓位（Window/Level）調整
- Hounsfield Unit 正規化
- 医療特化データ拡張

### モデルアーキテクチャ
- ResNet、EfficientNet、Vision Transformer対応
- 事前学習モデル活用
- Mixed Precision Training
- マルチスケール・アンサンブル学習

### 実験管理
- 設定スナップショット（完全再現性）
- W&B実験追跡
- CV-LB相関監視
- 自動閾値最適化

## 📋 要件

### システム要件
- Python 3.8+
- CUDA 11.7+ (GPU使用時)
- Google Colab Pro推奨（長時間実験用）

### 主要依存関係
- PyTorch 2.1+
- Albumentations 1.3+
- PyDICOM 2.4+
- Weights & Biases 0.16+
- scikit-learn 1.3+

詳細は `experiments/exp0001/env/requirements.lock` を参照。

## 🎯 パフォーマンス最適化

### GPU活用
```python
# Mixed Precision Training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
```

### データローディング最適化
```python
# 最適化されたDataLoader
train_loader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
```

## 📈 評価・検証

### クロスバリデーション
- **Stratified Group K-Fold**: 患者単位でのデータ分離
- **リーク防止**: 同一患者の画像が複数Foldに分散しない設計
- **一貫性監視**: Fold間スコア標準偏差によるCV品質評価

### 評価指標
- **ROC AUC** (主要指標)
- **Average Precision**  
- **Sensitivity/Specificity**
- **Calibration Error**

## 🔧 トラブルシューティング

### よくある問題

**GPU Memory不足**
```python
# バッチサイズ削減 or 勾配累積
batch_size = 8
accumulate_grad_batches = 2
```

**DICOM読み込みエラー**
```python
# エラーハンドリング付きDICOM処理
from scripts.dicom_utils import DICOMProcessor
processor = DICOMProcessor(default_window_center=40, default_window_width=80)
```

**Colab切断対策**
```python
# Google Driveへのバックアップ
from google.colab import drive
drive.mount('/content/drive')
# 重要ファイルを定期的にDriveに保存
```

詳細は [docs/colab_setup.md](docs/colab_setup.md) を参照。

## 📚 ドキュメント

- [Google Colab セットアップガイド](docs/colab_setup.md)
- [実験ワークフロー](docs/experiment_workflow.md)
- [API リファレンス](scripts/)

## 🤝 コントリビューション

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Create Pull Request

## 📄 ライセンス

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 実験結果例

| Experiment | Model | CV AUC | LB AUC | Key Changes |
|------------|-------|---------|---------|-------------|
| exp0001 | ResNet50 | 0.8732 ± 0.0055 | 0.8701 | Baseline |
| exp0002 | ResNet50 | 0.8810 ± 0.0048 | 0.8756 | Enhanced Augmentation |
| exp0003 | EfficientNet-B2 | 0.8891 ± 0.0042 | 0.8834 | Architecture Change |

## 🔗 関連リンク

- [Kaggle Competition](https://www.kaggle.com/c/rsna-intracranial-aneurysm-detection)
- [RSNA Challenge](https://www.rsna.org/en/education/ai-resources-and-training/ai-image-challenge)
- [PyTorch Medical Imaging](https://pytorch.org/blog/accelerating-medical-imaging-with-pytorch/)

---

**🧠 医療AIで脳動脈瘤検出の精度向上を目指しましょう！**

For questions and support, please create an issue or join our discussion forum.