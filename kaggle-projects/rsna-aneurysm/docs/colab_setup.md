# Google Colab セットアップガイド - RSNA Aneurysm Detection

## 🚀 基本セットアップ

### 1. ランタイムの設定

1. **ランタイム** → **ランタイムのタイプを変更**
2. **ハードウェアアクセラレータ**: GPU
3. **GPU の種類**: T4 または A100 (利用可能な場合)
4. **高RAM**: 有効化（推奨）
5. **保存**をクリック

### 2. リポジトリのクローン

```python
# リポジトリクローン（初回のみ）
!git clone https://github.com/YOUR_USERNAME/rsna-aneurysm-project.git
%cd rsna-aneurysm-project/experiments/exp0001
```

### 3. 依存関係のインストール

```python
# 固定環境のインストール
!pip install -r env/requirements.lock

# GPUサポートの確認
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## 🔐 APIキーの設定

### Colab Secrets（推奨）

1. 🔑 **Secrets** パネルを開く
2. 以下のシークレットを追加：
   - `WANDB_API_KEY`: W&B API Key
   - `KAGGLE_USERNAME`: Kaggle ユーザー名
   - `KAGGLE_KEY`: Kaggle API Key

```python
from google.colab import userdata

# APIキー設定
import os
os.environ["WANDB_API_KEY"] = userdata.get('WANDB_API_KEY')
os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME') 
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
```

### 直接設定（非推奨）

```python
# セキュリティリスクあり - 本番では使用しない
import os
os.environ["KAGGLE_USERNAME"] = "YOUR_KAGGLE_USERNAME"
os.environ["KAGGLE_KEY"] = "YOUR_KAGGLE_KEY"
os.environ["WANDB_API_KEY"] = "YOUR_WANDB_KEY"
```

## 📁 ファイル構造の確認

```python
# プロジェクト構造確認
!ls -la
!ls experiments/exp0001/

# 設定ファイルの確認
!cat config.yaml | head -20
```

## 💾 Google Drive 連携による永続化

### Drive マウント

```python
from google.colab import drive
drive.mount('/content/drive')

# バックアップディレクトリ作成
backup_dir = "/content/drive/MyDrive/rsna-aneurysm"
!mkdir -p "{backup_dir}"
```

### 自動バックアップ設定

```python
import shutil
from pathlib import Path

def backup_to_drive(experiment_id, files_to_backup):
    """実験成果物をGoogle Driveにバックアップ"""
    backup_path = f"/content/drive/MyDrive/rsna-aneurysm/{experiment_id}"
    Path(backup_path).mkdir(parents=True, exist_ok=True)
    
    for file_path in files_to_backup:
        if Path(file_path).exists():
            if Path(file_path).is_dir():
                shutil.copytree(file_path, f"{backup_path}/{Path(file_path).name}", dirs_exist_ok=True)
            else:
                shutil.copy2(file_path, backup_path)
            print(f"Backed up: {file_path}")
    
    print(f"Backup completed: {backup_path}")

# 使用例
backup_files = [
    "model/",
    "oof_predictions.csv", 
    "metrics.json",
    "wandb_run.txt",
    "submissions/"
]
backup_to_drive("exp0001", backup_files)
```

## ⚠️ GPU・メモリ管理

### GPU使用状況監視

```python
# GPU使用量確認
!nvidia-smi

# PyTorchメモリ使用量
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### メモリクリア

```python
# GPUメモリクリア
import gc
gc.collect()
torch.cuda.empty_cache()

# プロセスメモリ確認
!free -h
```

### GPU/CPU自動切替

```python
def get_device():
    """利用可能な最適デバイスを取得"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

device = get_device()
```

## 🔄 実験実行フロー

### 完全版（初回実行）

```python
# 1. セットアップ
!git clone https://github.com/YOUR_USERNAME/rsna-aneurysm-project.git
%cd rsna-aneurysm-project

# 2. データ準備
!kaggle competitions download -c rsna-intracranial-aneurysm-detection -p data/raw --unzip
!python -m scripts.dicom_utils --input data/raw --output data/processed

# 3. 実験実行
%cd experiments/exp0001
# training.ipynb → evaluation.ipynb → inference.ipynb 順に実行
```

### 高速版（データ準備済み）

```python
# セットアップのみ
!git clone https://github.com/YOUR_USERNAME/rsna-aneurysm-project.git
%cd rsna-aneurysm-project/experiments/exp0001
!pip install -r env/requirements.lock

# APIキー設定
from google.colab import userdata
import os
for key in ['WANDB_API_KEY', 'KAGGLE_USERNAME', 'KAGGLE_KEY']:
    os.environ[key] = userdata.get(key)

# Google Drive マウント
from google.colab import drive
drive.mount('/content/drive')
```

## 📊 モニタリング・デバッグ

### W&B ログ監視

```python
import wandb

# W&B初期化確認
run = wandb.init(project="rsna-aneurysm-detection", name="test-run")
print(f"W&B run URL: {run.get_url()}")
wandb.finish()
```

### 学習進捗監視

```python
# リアルタイム学習監視用
def plot_training_progress(metrics_history):
    """学習進捗をプロット"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics_history['train_loss'], label='Train Loss')
    plt.plot(metrics_history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress - Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics_history['val_auc'], label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Training Progress - AUC')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
```

## 🎯 パフォーマンス最適化

### データローディング最適化

```python
# 最適なワーカー数の決定
import multiprocessing
num_workers = min(4, multiprocessing.cpu_count())
print(f"Using {num_workers} workers for data loading")

# DataLoaderの最適化
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=16,  # GPU メモリに応じて調整
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True if num_workers > 0 else False
)
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

# Scalerの初期化
scaler = GradScaler()

# 学習ループでの使用例
for images, labels in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 🔧 トラブルシューティング

### 一般的なエラーと対処法

#### 1. `ModuleNotFoundError`
```python
# 不足ライブラリの個別インストール
!pip install missing_package_name
```

#### 2. `CUDA out of memory`
```python
# バッチサイズを削減
batch_size = 8  # 16から8に削減

# または勾配累積を使用
accumulate_grad_batches = 2
effective_batch_size = batch_size * accumulate_grad_batches
```

#### 3. `RuntimeError: DataLoader worker (pid xxx) is killed by signal`
```python
# ワーカー数を削減
num_workers = 0  # または 1-2 に削減
```

#### 4. Kaggle API認証エラー
```python
# 認証情報の確認
import os
print("KAGGLE_USERNAME:", os.environ.get("KAGGLE_USERNAME"))
print("KAGGLE_KEY:", "***" if os.environ.get("KAGGLE_KEY") else "Not set")

# .kaggle/kaggle.json の代替設定
!mkdir -p ~/.kaggle
!echo '{"username":"YOUR_USERNAME","key":"YOUR_KEY"}' > ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
```

### ログファイル確認

```python
# エラーログの確認
!tail -50 /var/log/messages

# システム情報確認
!cat /proc/meminfo | head -10
!lscpu | head -20
```

## 📋 実験チェックリスト

### 実験開始前
- [ ] GPU ランタイムが有効
- [ ] 必要なライブラリがインストール済み
- [ ] API キーが設定済み
- [ ] Google Drive がマウント済み
- [ ] データが準備済み

### 学習中
- [ ] GPU 使用率が適切（80%以上）
- [ ] メモリ使用量が安全範囲内
- [ ] W&B ログが正常に記録
- [ ] 定期的にバックアップ実行

### 実験終了時
- [ ] モデルファイルが保存済み
- [ ] OOF 予測が保存済み
- [ ] 評価指標が記録済み
- [ ] 重要ファイルをDriveにバックアップ
- [ ] W&B ランが正常終了

## 🔄 セッション復旧

### セッション切断からの復旧

```python
# 1. 環境再構築
!git clone https://github.com/YOUR_USERNAME/rsna-aneurysm-project.git
%cd rsna-aneurysm-project/experiments/exp0001
!pip install -r env/requirements.lock

# 2. Drive マウントとバックアップからの復旧
from google.colab import drive
drive.mount('/content/drive')

# 3. 前回の状態を復元
!cp -r "/content/drive/MyDrive/rsna-aneurysm/exp0001/*" ./
```

### チェックポイントからの学習再開

```python
# チェックポイントの確認
import torch
checkpoint_files = !ls model/*.pth
print("Available checkpoints:", checkpoint_files)

# 学習再開
if checkpoint_files:
    latest_checkpoint = checkpoint_files[-1]  # 最新のチェックポイント
    checkpoint = torch.load(latest_checkpoint)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    
    print(f"Resuming from epoch {start_epoch}")
```

## 💡 ベストプラクティス

### 1. リソース効率的な使用
- バッチサイズはGPUメモリの80%程度を目安に設定
- 未使用のモデルは明示的に削除（`del model; torch.cuda.empty_cache()`）
- 長時間実験では定期的にメモリクリア

### 2. セキュリティ
- APIキーは必ずSecretsに保存
- 共有ノートブックでは認証情報を削除
- 公開前にセンシティブな情報を確認

### 3. 再現性
- 実験前に必ずGit commitを実行
- 環境は requirements.lock で固定
- ランダムシードを統一

### 4. 効率的な開発
- 小さなデータセットでプロトタイプを先に作成
- デバッグモードで動作確認してから本実験
- 重要な中間結果は都度保存

---

このガイドに従って、Google Colab環境でのRSNA Aneurysmプロジェクトを効率的に実行してください。問題が発生した場合は、トラブルシューティングセクションを参照してください。