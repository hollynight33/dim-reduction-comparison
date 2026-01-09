"""
MNISTとlivedoorニュースコーパスのデータ例を表示

MNISTの0~9の各数字について1つずつサンプルを画像として表示し、
livedoorの各カテゴリから1つずつテキストサンプルを表示する。
"""

import os
import re
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import fetch_openml
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


def load_mnist_examples() -> tuple[np.ndarray, np.ndarray]:
    """
    MNISTデータセットから各数字(0-9)の最初の例を取得
    
    Returns:
        images: 画像データ (10, 784)
        labels: ラベル (10,)
    """
    print("\nMNIST 784を読み込み中...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data, mnist.target.astype(int)
    
    # 各数字の最初の例を取得
    examples = []
    example_labels = []
    
    print("各数字の最初の例を取得中...")
    for digit in tqdm(range(10)):
        # 該当する数字のインデックスを取得
        indices = np.where(y == digit)[0]
        if len(indices) > 0:
            # 最初の例を取得
            examples.append(X[indices[0]])
            example_labels.append(digit)
    
    return np.array(examples), np.array(example_labels)


def load_livedoor_examples() -> list[tuple[str, str, str]]:
    """
    ライブドアニュースコーパスから各カテゴリの最初の例を取得
    
    Returns:
        examples: (カテゴリ名, タイトル, 本文)のリスト
    """
    download_livedoor_corpus()
    
    examples = []
    
    print("\n各カテゴリの最初の記事を読み込み中...")
    for category in tqdm(CATEGORIES):
        category_dir = CORPUS_DIR / category
        files = sorted(category_dir.glob("*.txt"))
        
        # LICENSE.txt を除外
        files = [f for f in files if f.name != "LICENSE.txt"]
        
        if len(files) > 0:
            # 最初のファイルを読み込む
            with open(files[0], "r", encoding="utf-8") as f:
                lines = f.readlines()
                
                # 最初の3行はメタデータ（URL、日付、タイトル）
                if len(lines) > 3:
                    title = lines[2].strip()
                    text = "".join(lines[3:]).strip()
                    # 空白や改行を正規化
                    text = re.sub(r"\s+", " ", text)
                    # 長すぎる場合は切り詰め
                    if len(text) > 500:
                        text = text[:500] + "..."
                    
                    examples.append((category, title, text))
    
    return examples


def plot_mnist_examples(images: np.ndarray, labels: np.ndarray, save_path: str = "images/mnist_examples.png") -> None:
    """
    MNISTの各数字の例を画像として表示
    
    Args:
        images: 画像データ (10, 784)
        labels: ラベル (10,)
        save_path: 保存先パス
    """
    # 保存先ディレクトリを作成
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()
    
    for idx, (image, label) in enumerate(zip(images, labels)):
        # 784次元を28x28に変換
        image_2d = image.reshape(28, 28)
        
        axes[idx].imshow(image_2d, cmap="gray")
        axes[idx].axis("off")
    
    plt.suptitle("MNIST手書き数字データの例 (0-9)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nMNIST例の図を保存しました: {save_path}")
    plt.show()


def print_livedoor_examples(examples: list[tuple[str, str, str]]) -> None:
    """
    livedoorニュースの各カテゴリの例をテキストで表示
    
    Args:
        examples: (カテゴリ名, タイトル, 本文)のリスト
    """
    print("\n" + "=" * 80)
    print("ライブドアニュースコーパスのデータ例")
    print("=" * 80)
    
    for idx, (category, title, text) in enumerate(examples, 1):
        print(f"\n【カテゴリ {idx}: {category}】")
        print(f"タイトル: {title}")
        print(f"本文: {text}")
        print("-" * 80)


def main():
    """メイン処理"""
    print("=" * 80)
    print("MNISTとlivedoorニュースコーパスのデータ例を表示")
    print("=" * 80)
    
    # MNISTの例を取得して表示
    mnist_images, mnist_labels = load_mnist_examples()
    plot_mnist_examples(mnist_images, mnist_labels)
    
    # livedoorの例を取得して表示
    livedoor_examples = load_livedoor_examples()
    print_livedoor_examples(livedoor_examples)
    
    print("\n完了しました！")


if __name__ == "__main__":
    main()
