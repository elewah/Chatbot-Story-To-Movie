"""Generate embedding CSVs for both models from the IMDB Top 1000 dataset."""

import pandas as pd
import kagglehub
from fastembed import TextEmbedding

# Download dataset via kagglehub
path = kagglehub.dataset_download("harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")
df = pd.read_csv(f"{path}/imdb_top_1000.csv")

overviews = df["Overview"].fillna("").tolist()

models = {
    "sentence-transformers/all-MiniLM-L6-v2": "embeddings_minilm.csv",
    "BAAI/bge-small-en-v1.5": "embeddings_bge.csv",
}

for model_name, output_file in models.items():
    print(f"Generating embeddings with {model_name}...")
    model = TextEmbedding(model_name)
    embeddings = list(model.embed(overviews))
    df_out = df.copy()
    df_out["embeddings"] = [emb.tolist() for emb in embeddings]
    df_out.to_csv(output_file, index=False)
    print(f"  Saved {output_file} — shape: {df_out.shape}, dim: {len(embeddings[0])}")

print("Done.")
