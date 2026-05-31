
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_distances


@st.cache_data
def load_embeddings(embeddings_file):
    df = pd.read_csv(embeddings_file)
    df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)
    return df


def get_relevant_movies(user_question, model, embeddings_file, query_prefix=""):
    df = load_embeddings(embeddings_file)

    question_embedding = list(model.embed([query_prefix + user_question]))[0].tolist()
    distances = cosine_distances([question_embedding], list(df["embeddings"]))[0]

    # Work on a copy to avoid mutating cached data
    df_copy = df.copy()
    df_copy["distances"] = distances
    df_copy.sort_values(by="distances", ascending=True, inplace=True)

    return df_copy.head(10)[["Series_Title", "Overview"]].to_dict(orient="records")
