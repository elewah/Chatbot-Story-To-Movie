
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_distances


@st.cache_data
def load_embeddings():
    df = pd.read_csv("embeddings.csv", index_col=0)
    df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)
    return df


def get_relevant_movies(user_question, model):
    """
    Given a user question, returns a list of the top 3 most relevant movies with their titles and overviews.

    Parameters:
        user_question (str): The user's question or description.
        model: A fastembed TextEmbedding model instance.

    Returns:
        list: A list of dicts containing movie titles and overviews.
    """
    df = load_embeddings()

    # Generate the embedding for the user question
    question_embedding = list(model.embed([user_question]))[0].tolist()

    # Calculate distances
    distances = cosine_distances([question_embedding], list(df["embeddings"]))[0]

    # Work on a copy to avoid mutating cached data
    df_copy = df.copy()
    df_copy["distances"] = distances
    df_copy.sort_values(by="distances", ascending=True, inplace=True)

    # Get the top 3 most relevant movies
    df_3movies = df_copy.head(5)
    return df_3movies[["Series_Title", "Overview"]].to_dict(orient="records")
