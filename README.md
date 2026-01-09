# 次元削減とクラスタリングの可視化

PCA、t-SNE、UMAPを用いた次元削減手法の比較と、BERTopicを使用したテキストクラスタリングのデモンストレーションプロジェクトです。

## 概要

このプロジェクトでは、以下の2つの主要なタスクを実装しています:

1. **MNIST手書き数字の次元削減可視化**: 784次元のMNISTデータを2次元に削減し、各次元削減手法の特性を比較
2. **ライブドアニュースコーパスのクラスタリング**: Google Embedding APIで生成した埋め込みベクトルに対してBERTopicとKMeansを用いてクラスタリングを実行し、ARI/NMIで精度評価

## 実行環境

- Python 3.12.8

## セットアップ

### 1. リポジトリのクローン

```bash
git clone <your-repository-url>
cd 次元削減
```

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 3. 環境変数の設定（**Optional**）

⚠️ **通常は不要です。**

本リポジトリには、すでに生成済みの埋め込みベクトル  `data/embeddings_cache.npy` が含まれているため、  **APIキーを設定しなくてもそのまま実行できます。**

---

#### 以下の場合のみ設定してください
- 埋め込みベクトルを再生成したい場合
- データを変更した場合
- Google Embedding API を直接使いたい場合

```bash
export GOOGLE_API_KEY='your-api-key-here'
```

または、`.env`ファイルを作成して以下を記述してください:

```
GOOGLE_API_KEY=your-api-key-here
```

## 使用方法

### 1. データ例の表示

MNISTの各数字(0-9)の例を画像として表示します。

```bash
python data_examples.py
```

**出力:**
- `images/mnist_examples.png`: 0-9の数字の例を表示した画像

### 2. MNIST手書き数字の次元削減可視化

784次元のMNISTデータをPCA、t-SNE、UMAPで2次元に削減し、可視化します。

```bash
python mnist_visualization.py
```

**出力:**
- `images/mnist_dimensionality_reduction.png`: 3つの次元削減手法による可視化結果
- 各手法の実行時間がコンソールに表示されます

**パラメータ:**
- デフォルトでは10,000サンプルを使用します
- サンプル数を変更する場合は、`main()`関数内の`load_mnist_data(n_samples=10000)`を編集してください

### 3. ライブドアニュースコーパスのクラスタリング

Google Embedding APIで埋め込みベクトルを生成し、次元削減を適用してからBERTopicでクラスタリングを実行します。

```bash
python livedoor_clustering.py
```

**出力:**
- `images/livedoor_clustering_results.png`: クラスタリング結果の可視化
- `data/embeddings_cache.npy`: 埋め込みベクトルのキャッシュファイル (2回目以降は再利用されます)
- ARI/NMIスコアと実行時間がコンソールに表示されます

**実行される次元削減手法:**
- 次元削減なし (768次元のまま)
- PCA (50次元に削減)
- t-SNE (3次元に削減)
- UMAP (10次元に削減)

## プロジェクト構成

```
次元削減/
├── README.md                           # このファイル
├── requirements.txt                    # 依存パッケージ一覧
├── .env.example                        # 環境変数設定の例
├── data_examples.py                    # データ例の表示
├── mnist_visualization.py              # MNIST次元削減可視化
├── livedoor_clustering.py              # ライブドアニュースクラスタリング
├── images/                             # 出力画像ディレクトリ
│   ├── mnist_examples.png
│   ├── mnist_dimensionality_reduction.png
│   └── livedoor_clustering_results.png
└── data/                               # データディレクトリ (自動生成)
    ├── text/                           # ライブドアニュースコーパス
    └── embeddings_cache.npy            # 埋め込みベクトルキャッシュ
```

## 次元削減手法の特徴

### PCA (主成分分析)
- **特徴**: 線形変換、高速
- **適用場面**: データの大まかな構造把握、前処理

### t-SNE (t-分布確率的近傍埋め込み)
- **特徴**: 局所構造を保持、非線形
- **適用場面**: クラスタの可視化、類似データのグルーピング

### UMAP (一様多様体近似と射影)
- **特徴**: 局所・大域構造のバランス、高速
- **適用場面**: 大規模データの可視化、クラスタリング前処理

## 評価指標

### ARI (Adjusted Rand Index)
- -1 ~ 1の範囲 (1に近いほど良い)
- ランダムなクラスタリングに対して補正されたスコア

### NMI (Normalized Mutual Information)
- 0 ~ 1の範囲 (1に近いほど良い)
- クラスタと正解ラベルの相互情報量

## ライセンス

MIT License

## 謝辞

- **MNISTデータセット**: Yann LeCun et al.
- **ライブドアニュースコーパス**: 株式会社ロンウイット
- **Google Embedding API**: Google AI (埋め込みベクトル生成に使用)

## 参考文献

- van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine Learning Research, 9, 2579-2605.
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794.
