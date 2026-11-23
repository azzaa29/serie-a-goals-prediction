"""Build data/processed/matches.csv from the raw files.

    python src/build_dataset.py
"""

from config import DATASET_PATH, SEASONS
from features import build_dataset

if __name__ == "__main__":
    df = build_dataset(SEASONS)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Saved {len(df)} matches, {df.shape[1]} columns -> {DATASET_PATH}")
