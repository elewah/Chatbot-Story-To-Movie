"""Generate embedding CSVs for both models from the IMDB Top 1000 dataset."""

import pandas as pd
import kagglehub
from fastembed import TextEmbedding

# Download dataset via kagglehub
path = kagglehub.dataset_download("harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")
df = pd.read_csv(f"{path}/imdb_top_1000.csv")

overviews = df["Overview"].fillna("").tolist()

# nomic-embed requires "search_document:" prefix on stored texts for best retrieval quality
models = {
    "sentence-transformers/all-MiniLM-L6-v2": ("embeddings_minilm.csv", ""),
    "BAAI/bge-small-en-v1.5": ("embeddings_bge.csv", ""),
    "nomic-ai/nomic-embed-text-v1.5": ("embeddings_nomic.csv", "search_document: "),
}

for model_name, (output_file, doc_prefix) in models.items():
    print(f"Generating embeddings with {model_name}...")
    model = TextEmbedding(model_name)
    texts = [doc_prefix + t for t in overviews]
    embeddings = list(model.embed(texts))
    df_out = df.copy()
    df_out["embeddings"] = [emb.tolist() for emb in embeddings]
    df_out.to_csv(output_file, index=False)
    print(f"  Saved {output_file} — shape: {df_out.shape}, dim: {len(embeddings[0])}")

print("Done.")
