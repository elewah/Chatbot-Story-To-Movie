"""One-time script to regenerate embeddings.csv with fastembed (384-dim) instead of OpenAI (1536-dim)."""

import pandas as pd
from fastembed import TextEmbedding

# Load current embeddings, drop old embeddings column
df = pd.read_csv("embeddings.csv", index_col=0)
df.drop(columns=["embeddings"], inplace=True)

# Generate new embeddings using fastembed
model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
overviews = df["Overview"].fillna("").tolist()
embeddings = list(model.embed(overviews))

# Store as stringified lists (same format as before)
df["embeddings"] = [emb.tolist() for emb in embeddings]

df.to_csv("embeddings.csv")
print(f"Done. Shape: {df.shape}, embedding dim: {len(embeddings[0])}")
