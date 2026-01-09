"""
MNIST手書き数字データの次元削減可視化

PCA, t-SNE, UMAPを使用して784次元のMNISTデータを2次元に削減し、
ラベル（0-9）ごとに色分けした散布図を作成する。
"""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import time


def load_mnist_data(n_samples: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    """
    MNISTデータセットを読み込む (28x28 = 784次元)
    
    Args:
        n_samples: 使用するサンプル数
    
    Returns:
        X: 画像データ (n_samples, 784)
        y: ラベル (n_samples,)
    """
    print("MNIST 784を読み込み中...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data, mnist.target.astype(int)
    
    # サンプリング
    if n_samples < len(X):
        np.random.seed(42)
        indices = np.random.choice(len(X), n_samples, replace=False)
        X, y = X[indices], y[indices]
    
    # 正規化
    X = X / 255.0
    
    print(f"データサイズ: {X.shape[0]}サンプル, {X.shape[1]}次元")
    return X, y


def apply_pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """PCAで次元削減"""
    print("PCAを実行中...")
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X)


def apply_tsne(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """t-SNEで次元削減"""
    print("t-SNEを実行中...")
    tsne = TSNE(
        n_components=n_components,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42
    )
    return tsne.fit_transform(X)


def apply_umap(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """UMAPで次元削減"""
    print("UMAPを実行中...")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )
    return reducer.fit_transform(X)


def plot_comparison(
    X_pca: np.ndarray,
    X_tsne: np.ndarray,
    X_umap: np.ndarray,
    y: np.ndarray,
    save_path: str = "images/mnist_dimensionality_reduction.png"
) -> None:
    """
    3手法の比較散布図を作成
    
    Args:
        X_pca: PCA結果 (n_samples, 2)
        X_tsne: t-SNE結果 (n_samples, 2)
        X_umap: UMAP結果 (n_samples, 2)
        y: ラベル (n_samples,)
        save_path: 保存先パス
    """
    # 保存先ディレクトリを作成
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # カラーマップ（10クラス用）
    cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(i) for i in range(10)]
    
    data_list = [
        (X_pca, "PCA"),
        (X_tsne, "t-SNE"),
        (X_umap, "UMAP")
    ]
    
    for ax, (X_reduced, title) in zip(axes, data_list):
        for digit in range(10):
            mask = y == digit
            ax.scatter(
                X_reduced[mask, 0],
                X_reduced[mask, 1],
                c=[colors[digit]],
                label=str(digit),
                alpha=0.6,
                s=10
            )
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("成分1")
        ax.set_ylabel("成分2")
        ax.legend(title="数字", loc="best", markerscale=2)
    
    plt.suptitle("MNIST手書き数字データの次元削減による可視化", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"図を保存しました: {save_path}")
    plt.show()


def main():
    """メイン処理"""
    # データ読み込み
    X, y = load_mnist_data()
    
    # 次元削減
    start_time = time.time()
    X_pca = apply_pca(X)
    pca_time = time.time() - start_time
    print(f"PCA execution time: {pca_time:.2f} seconds")
    
    start_time = time.time()
    X_tsne = apply_tsne(X)
    tsne_time = time.time() - start_time
    print(f"t-SNE execution time: {tsne_time:.2f} seconds")
    
    start_time = time.time()
    X_umap = apply_umap(X)
    umap_time = time.time() - start_time
    print(f"UMAP execution time: {umap_time:.2f} seconds")
    
    # 可視化
    plot_comparison(X_pca, X_tsne, X_umap, y)
    
    print("\n完了しました！")


if __name__ == "__main__":
    main()
