"""
ライブドアニュースコーパスのBERTopicクラスタリング

Google Embedding API(gemini-embedding-001)で埋め込みベクトルを生成し、
BERTopic + 次元削減（PCA/t-SNE/UMAP）+ KMeansでクラスタリングを行う。
クラスタリング精度をARI、NMIで評価する。
"""

import os
import re
import tarfile
import urllib.request
import time
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from google import genai
from google.genai import types
from tqdm import tqdm

# ライブドアニュースコーパスのURL
LIVEDOOR_URL = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
DATA_DIR = Path("./data")
CORPUS_DIR = DATA_DIR / "text"

# カテゴリ名とラベルのマッピング
CATEGORIES = [
    "dokujo-tsushin",
    "it-life-hack", 
    "kaden-channel",
    "livedoor-homme",
    "movie-enter",
    "peachy",
    "smax",
    "sports-watch",
    "topic-news"
]


def download_livedoor_corpus() -> None:
    """ライブドアニュースコーパスをダウンロードして展開"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = DATA_DIR / "ldcc-20140209.tar.gz"
    
    if CORPUS_DIR.exists():
        print("コーパスは既にダウンロード済みです")
        return
    
    print("ライブドアニュースコーパスをダウンロード中...")
    urllib.request.urlretrieve(LIVEDOOR_URL, tar_path)
    
    print("展開中...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    
    tar_path.unlink()
    print("ダウンロード完了")


def load_livedoor_corpus(max_docs_per_category: Optional[int] = None) -> tuple[list[str], list[int], list[str]]:
    """
    ライブドアニュースコーパスを読み込む
    
    Args:
        max_docs_per_category: カテゴリごとの最大記事数（Noneで全件）
    
    Returns:
        documents: 記事本文のリスト
        labels: カテゴリラベル（0-8）のリスト
        category_names: カテゴリ名のリスト
    """
    download_livedoor_corpus()
    
    documents = []
    labels = []
    
    print("記事を読み込み中...")
    for label_id, category in enumerate(tqdm(CATEGORIES)):
        category_dir = CORPUS_DIR / category
        files = sorted(category_dir.glob("*.txt"))
        
        # LICENSE.txt を除外
        files = [f for f in files if f.name != "LICENSE.txt"]
        
        if max_docs_per_category:
            files = files[:max_docs_per_category]
        
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # 最初の3行はメタデータ（URL、日付、タイトル）なのでスキップ
                if len(lines) > 3:
                    text = "".join(lines[3:]).strip()
                    # 空白や改行を正規化
                    text = re.sub(r"\s+", " ", text)
                    if text:
                        documents.append(text)
                        labels.append(label_id)
    
    print(f"読み込み完了: {len(documents)}記事, {len(CATEGORIES)}カテゴリ")
    return documents, labels, CATEGORIES


def get_embeddings_google(
    documents: list[str],
    model_name: str = "gemini-embedding-001",
    batch_size: int = 100,
    cache_file: str = "embeddings_cache.npy"
) -> np.ndarray:
    """
    Google Embedding APIで埋め込みベクトルを生成(キャッシュ機能付き)
    
    Args:
        documents: テキストのリスト
        model_name: 使用するモデル名
        batch_size: バッチサイズ
        cache_file: キャッシュファイル名
    
    Returns:
        embeddings: 埋め込みベクトル (n_docs, embedding_dim)
    """
    cache_path = DATA_DIR / cache_file
    
    # キャッシュが存在すれば読み込む
    if cache_path.exists():
        print(f"キャッシュから埋め込みを読み込み中: {cache_path}")
        embeddings = np.load(cache_path)
        print(f"埋め込み読み込み完了: {embeddings.shape}")
        return embeddings
    
    # API キーの設定
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "環境変数 GOOGLE_API_KEY が設定されていません。\n"
            "export GOOGLE_API_KEY='your-api-key' を実行してください。"
        )
    
    client = genai.Client(api_key=api_key)
    
    print(f"Google Embedding API ({model_name}) で埋め込みを生成中...")
    embeddings = []
    
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i + batch_size]
        # テキストが長すぎる場合は切り詰め(APIの制限対応)
        batch = [doc[:2000] for doc in batch]
        
        result = client.models.embed_content(
            model=model_name,
            contents=batch,
            config=types.EmbedContentConfig(
                output_dimensionality=768,
                task_type="CLUSTERING"
            )
        )
        embeddings.extend([emb.values for emb in result.embeddings])
    
    embeddings = np.array(embeddings)
    print(f"埋め込み完了: {embeddings.shape}")
    
    # キャッシュに保存
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"埋め込みをキャッシュに保存: {cache_path}")
    
    return embeddings


class IdentityWrapper(BaseDimensionalityReduction):
    """次元削減なしのラッパー"""
    def __init__(self):
        pass
    
    def fit(self, X, y=None, **kwargs):
        return self
    
    def transform(self, X, **kwargs):
        return X
    
    def fit_transform(self, X, y=None, **kwargs):
        return X


class TSNEWrapper(BaseDimensionalityReduction):
    """BERTopic用のt-SNEラッパー（PCA前処理なし）"""
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.tsne = TSNE(
            n_components=n_components,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            random_state=42,
            n_jobs=-1
        )
    
    def fit(self, X, y=None, **kwargs):
        # t-SNEはfitをサポートしないが、インターフェース互換性のため定義
        return self
    
    def transform(self, X, **kwargs):
        # t-SNEはtransformをサポートしないので、fit_transformを使う
        return self.tsne.fit_transform(X)
    
    def fit_transform(self, X, y=None, **kwargs):
        return self.tsne.fit_transform(X)


def run_bertopic_with_dim_reduction(
    documents: list[str],
    embeddings: np.ndarray,
    dim_reduction_model: BaseDimensionalityReduction,
    method_name: str
) -> tuple[np.ndarray, BERTopic]:
    """
    指定した次元削減手法でBERTopicを実行する
    """
    print(f"\n{method_name}でBERTopicを実行中...")
    
    cluster_model = KMeans(n_clusters=9, random_state=42)
    
    topic_model = BERTopic(
        umap_model=dim_reduction_model,
        hdbscan_model=cluster_model,
        calculate_probabilities=True,
        verbose=False
    )
    
    topics, _ = topic_model.fit_transform(documents, embeddings=embeddings)
    
    return np.array(topics), topic_model


def evaluate_clustering(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    method_name: str
) -> dict[str, float]:
    """
    クラスタリング精度を評価
    
    Args:
        true_labels: 正解ラベル
        pred_labels: 予測ラベル
        method_name: 手法名
    
    Returns:
        metrics: ARI, NMIのスコア
    """
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    
    return {"ARI": ari, "NMI": nmi}


def plot_results(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    results: dict[str, tuple[np.ndarray, dict]],
    save_path: str = "images/livedoor_clustering_results.png"
) -> None:
    """
    クラスタリング結果の可視化
    
    Args:
        embeddings: 埋め込みベクトル
        true_labels: 正解ラベル
        results: 各手法の結果（予測ラベル, 評価指標）
        save_path: 保存先パス
    """
    # 保存先ディレクトリを作成
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 可視化用に2次元に削減（UMAP使用）
    print("\n可視化用に2次元に削減中...")
    reducer = umap.UMAP(n_components=2, metric="cosine", min_dist=0.0, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # 5つのプロット（正解ラベル + 4手法）
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # カラーマップ
    cmap = plt.cm.get_cmap("tab10")
    
    # 正解ラベルでの可視化
    ax = axes[0]
    for label_id in range(len(CATEGORIES)):
        mask = true_labels == label_id
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[cmap(label_id % 10)],
            label=CATEGORIES[label_id],
            alpha=0.6,
            s=10
        )
    ax.set_title("正解ラベル", fontsize=14)
    ax.legend(loc="best", fontsize=8)
    
    # 各手法の結果（4手法）
    for idx, (method_name, (pred_labels, metrics)) in enumerate(results.items(), start=1):
        ax = axes[idx]
        
        unique_labels = np.unique(pred_labels)
        for label in unique_labels:
            mask = pred_labels == label
            color = cmap(label % 10)
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[color],
                alpha=0.6,
                s=10
            )
        
        title = f"{method_name}\nARI={metrics['ARI']:.3f}, NMI={metrics['NMI']:.3f}"
        ax.set_title(title, fontsize=14)
        if len(unique_labels) <= 15:
            ax.legend(loc="best", fontsize=8)
    
    # 最後のサブプロット（6番目）を非表示にする
    axes[5].axis('off')
    
    plt.suptitle("ライブドアニュースのクラスタリング結果", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"図を保存しました: {save_path}")
    plt.show()


def print_summary(results: dict[str, tuple[np.ndarray, dict]]) -> None:
    """結果のサマリーを表示"""
    print("\n" + "=" * 60)
    print("クラスタリング精度比較")
    print("=" * 60)
    print(f"{'手法':<15} {'ARI':>10} {'NMI':>10}")
    print("-" * 60)
    
    for method_name, (_, metrics) in results.items():
        print(f"{method_name:<15} {metrics['ARI']:>10.4f} {metrics['NMI']:>10.4f}")
    
    print("=" * 60)


def main():
    """メイン処理"""
    # データ読み込み（テスト用に記事数を制限可能）
    documents, labels, category_names = load_livedoor_corpus(max_docs_per_category=None)
    labels = np.array(labels)
    
    # Google Embedding APIで埋め込み生成
    embeddings = get_embeddings_google(documents)
    
    # 各次元削減手法でBERTopicを実行
    dim_reduction_models = {
        "次元削減なし": IdentityWrapper(),
        "PCA": PCA(n_components=50, random_state=42),
        "t-SNE": TSNEWrapper(n_components=3),
        "UMAP": umap.UMAP(n_neighbors=50,n_components=10, min_dist=0, metric="cosine", random_state=42),
    }
    
    results = {}
    for method_name, model in dim_reduction_models.items():
        start_time = time.time()
        topics, topic_model = run_bertopic_with_dim_reduction(documents, embeddings, model, method_name)
        clustering_time = time.time() - start_time
        metrics = evaluate_clustering(labels, topics, method_name)
        results[method_name] = (topics, metrics)
        print(f"{method_name}: ARI={metrics['ARI']:.4f}, NMI={metrics['NMI']:.4f}")
        print(f"{method_name} execution time: {clustering_time:.2f} seconds")
    
    # 結果の可視化
    plot_results(embeddings, labels, results)
    
    # サマリー表示
    print_summary(results)
    
    print("\n完了しました！")


if __name__ == "__main__":
    main()
