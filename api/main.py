"""FastAPI API backed by processed Parquet (Notebook 2).

Endpoints read ``data/amazon_clean.parquet/clean_data.parquet``. Sentiment uses the same
deterministic keyword heuristic as ``app.py`` (no Spark JVM here).

Run: ``uvicorn fastapi_app:app --reload`` or ``python fastapi_app.py``
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import uvicorn
import mlflow
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Try loading the trained model artifacts
try:
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    # model = mlflow.pyfunc.load_model("models/best_model")
    MODEL_LOADED = True
except Exception:
    MODEL_LOADED = False


from src.utils.config_loader import load_config
config, _ = load_config()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PARQUET = PROJECT_ROOT / config["data"]["clean_path"]
RAW_INGESTED_PARQUET = PROJECT_ROOT / config["data"]["raw_path"]

POSITIVE_WORDS = frozenset(
    {
        "amazing",
        "awesome",
        "best",
        "excellent",
        "good",
        "great",
        "love",
        "nice",
        "perfect",
        "recommend",
        "satisfied",
        "thanks",
        "wonderful",
    }
)

NEGATIVE_WORDS = frozenset(
    {
        "awful",
        "bad",
        "broken",
        "disappoint",
        "garbage",
        "hate",
        "horrible",
        "junk",
        "mediocre",
        "pathetic",
        "poor",
        "refund",
        "rubbish",
        "terrible",
        "trash",
        "useless",
        "waste",
        "worst",
    }
)

NEGATION_CUES = frozenset({"not", "no", "never", "none", "hardly", "barely", "scarcely"})


def _token_polarity_hits(text_lower: str) -> tuple[int, int]:
    """Count positive/negative hits with simple negation handling."""
    tokens = re.findall(r"[a-z]+(?:n't)?|[.!?]", text_lower)
    hp, hn = 0, 0
    neg_window = 0
    for tok in tokens:
        if tok in {".", "!", "?"}:
            neg_window = 0
            continue
        if tok in NEGATION_CUES or tok.endswith("n't"):
            neg_window = 3
            continue

        pos_hit = any(tok.startswith(w) for w in POSITIVE_WORDS)
        neg_hit = any(tok.startswith(w) for w in NEGATIVE_WORDS)
        if pos_hit:
            if neg_window > 0:
                hn += 1
            else:
                hp += 1
        if neg_hit:
            if neg_window > 0:
                hp += 1
            else:
                hn += 1

        if neg_window > 0:
            neg_window -= 1
    return hp, hn


def _confidence_from_positive_strength(n: int) -> float:
    if n <= 0:
        return 0.50
    conf = 0.82 + 0.17 * min(1.0, max(0, n - 1) / 2.0)
    return float(round(min(0.99, conf), 4))


def _confidence_from_negative_strength(n: int) -> float:
    if n <= 0:
        return 0.50
    conf = 0.62 + 0.28 * min(1.0, max(0, n - 1) / 2.0)
    return float(round(min(0.93, conf), 4))


def _classify_sentiment(text_lower: str) -> tuple[str, float]:
    hp, hn = _token_polarity_hits(text_lower)
    net = hp - hn
    if net > 0:
        return "Positive", _confidence_from_positive_strength(net)
    if net < 0:
        return "Negative", _confidence_from_negative_strength(-net)
    if hp == 0 and hn == 0:
        return "Neutral", 0.48
    return "Neutral", 0.52

app = FastAPI(
    title="Amazon Review Intelligence API",
    version="1.1.0",
    description="Stats and user lookups from cleaned Parquet; sentiment is a lightweight heuristic.",
)


class ReviewRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float


class TopProductRow(BaseModel):
    parent_asin: str
    rating: float = Field(description="Observed star rating (1–5) in clean data")


class TopProductsResponse(BaseModel):
    user_id: str
    source: str = Field(
        default="amazon_clean.parquet",
        description="Rows filtered by user_id (not ALS predictions).",
    )
    items: list[TopProductRow]


def _clean_dataset_available() -> bool:
    return CLEAN_PARQUET.is_dir() or CLEAN_PARQUET.is_file()


def _folder_size_bytes(root: Path) -> int:
    total = 0
    if root.is_file():
        return root.stat().st_size
    if root.is_dir():
        for dirpath, _, files in os.walk(root):
            for f in files:
                total += os.path.getsize(Path(dirpath) / f)
    return total


def _raw_data_file_sizes() -> dict:
    """Notebook 1 Parquet folder + optional original JSONL downloads in ``data/``."""
    out: dict = {}
    if RAW_INGESTED_PARQUET.exists():
        out["approx_raw_ingested_parquet_gb"] = round(
            _folder_size_bytes(RAW_INGESTED_PARQUET) / (1024**3), 3
        )
        out["raw_ingested_parquet_path"] = str(RAW_INGESTED_PARQUET)
    jsonl = sorted(PROJECT_ROOT.glob("data/*.jsonl"))
    if jsonl:
        total_b = sum(f.stat().st_size for f in jsonl)
        out["raw_jsonl_total_gb"] = round(total_b / (1024**3), 3)
        out["raw_jsonl_files"] = [f.name for f in jsonl]
    return out


@lru_cache(maxsize=1)
def _cached_dataset_stats() -> dict | None:
    if not _clean_dataset_available():
        return None
    try:
        dataset = ds.dataset(str(CLEAN_PARQUET), format="parquet")
        n = int(dataset.count_rows())
        tbl = dataset.scanner(columns=["rating"]).to_table()
        ratings = tbl.column(0).to_numpy(zero_copy_only=False)
        mean_rating = float(np.mean(ratings))
        unique, counts = np.unique(ratings, return_counts=True)
        rating_distribution = {int(u): int(c) for u, c in zip(unique, counts)}
        amazon_clean_root = CLEAN_PARQUET.parent
        size_gb = _folder_size_bytes(amazon_clean_root) / (1024**3)
        return {
            "review_count": n,
            "mean_rating": mean_rating,
            "rating_distribution": rating_distribution,
            "approx_amazon_clean_gb": round(size_gb, 3),
            "parquet_path": str(CLEAN_PARQUET),
        }
    except Exception:
        return None


@app.get("/stats")
def dataset_stats():
    """Aggregates + rating histogram from processed clean Parquet (aligned with Streamlit dashboard)."""
    stats = _cached_dataset_stats()
    raw_sizes = _raw_data_file_sizes()
    if stats is None:
        return {
            "available": False,
            "message": f"No dataset at {CLEAN_PARQUET}",
            **raw_sizes,
        }
    return {"available": True, **stats, **raw_sizes}


@app.get("/health")
def health():
    return {"status": "ok", "dataset_configured": _clean_dataset_available()}


@app.get("/users/{user_id}/top-products", response_model=TopProductsResponse)
def top_products_for_user(
    user_id: str,
    limit: int = Query(10, ge=1, le=100),
):
    """Highest star ratings this user gave in the cleaned dataset (same logic as Streamlit)."""
    if not _clean_dataset_available():
        raise HTTPException(status_code=503, detail="Clean Parquet dataset not found.")
    uid = user_id.strip()
    if not uid:
        raise HTTPException(status_code=400, detail="user_id is empty.")
    try:
        df = pd.read_parquet(
            str(CLEAN_PARQUET),
            columns=["user_id", "parent_asin", "rating"],
            filters=[("user_id", "==", uid)],
            engine="pyarrow",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Parquet read failed: {exc}") from exc
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="No rows for this user_id in clean_data.parquet.",
        )
    df = (
        df.sort_values("rating", ascending=False)
        .drop_duplicates(subset=["parent_asin"])
        .head(limit)
    )
    items = [
        TopProductRow(parent_asin=str(row.parent_asin), rating=float(row.rating))
        for row in df.itertuples(index=False)
    ]
    return TopProductsResponse(user_id=uid, items=items)


@app.post("/predict_sentiment", response_model=SentimentResponse)
def predict_sentiment(request: ReviewRequest):
    """Deterministic keyword heuristic (matches ``app.py`` fallback)."""
    sentiment, confidence = _classify_sentiment(request.text.lower())
    return SentimentResponse(sentiment=sentiment, confidence=confidence)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
