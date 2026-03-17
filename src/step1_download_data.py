"""
Bước 1a: Download datasets
- International Football Results from GitHub (martj42 - same as Kaggle dataset)
- FIFA World Ranking from GitHub mirror
"""
import os
import requests

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)


def download_file(url, filepath):
    """Download a file from URL to local path."""
    print(f"  Downloading: {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(filepath, 'wb') as f:
        f.write(resp.content)
    size = os.path.getsize(filepath) / 1024 / 1024
    print(f"  -> Saved: {os.path.basename(filepath)} ({size:.2f} MB)")


def download_datasets():
    # 1. International Football Results (GitHub - martj42, same data as Kaggle)
    print("=== [1/2] International Football Results ===")
    base_url = "https://raw.githubusercontent.com/martj42/international_results/master"
    for filename in ["results.csv", "goalscorers.csv", "shootouts.csv"]:
        download_file(f"{base_url}/{filename}", os.path.join(RAW_DIR, filename))

    # 2. FIFA World Ranking
    print("\n=== [2/2] FIFA World Ranking ===")
    ranking_url = "https://raw.githubusercontent.com/cnvrg/fifa-world-ranking/master/fifa_ranking.csv"
    try:
        download_file(ranking_url, os.path.join(RAW_DIR, "fifa_ranking.csv"))
    except Exception as e:
        print(f"  Warning: Primary source failed: {e}")
        # Fallback: try alternative source
        alt_url = "https://raw.githubusercontent.com/mahinmukul/Fifa-World-Cup-Prediction/main/datasets/fifa_ranking-2024-06-20.csv"
        try:
            download_file(alt_url, os.path.join(RAW_DIR, "fifa_ranking.csv"))
        except Exception as e2:
            print(f"  Warning: Fallback also failed: {e2}")
            print("  -> Will use Elo system as ranking proxy in preprocessing")

    print("\n=== Download Complete ===")
    print("Files in raw directory:")
    for f in sorted(os.listdir(RAW_DIR)):
        size = os.path.getsize(os.path.join(RAW_DIR, f)) / 1024 / 1024
        print(f"  {f} ({size:.2f} MB)")


if __name__ == '__main__':
    download_datasets()
