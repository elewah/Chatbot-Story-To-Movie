
import numpy as np
import pandas as pd
import json
import os
#load dotenv
# !pip install python-dotenv
from dotenv import load_dotenv
load_dotenv()
import openai

# Load environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_base = "https://openai.vocareum.com/v1"
openai.api_key = OPENAI_API_KEY
df = pd.read_csv("embeddings.csv", index_col=0)
df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)

def get_relevant_movies(user_question, embedding_model_name="text-embedding-ada-002"):
    """
    Given a user question, returns a list of the top 3 most relevant movies with their titles and overviews.

    Parameters:
        user_question (str): The user's question or description.
        df (pd.DataFrame): The dataframe containing movie data and their embeddings.
        embedding_model_name (str): The name of the embedding model to use.

    Returns:
        list: A list of JSON objects containing movie titles and overviews.
    """
    # Generate the embedding for the user question
    response = openai.Embedding.create(
        input=user_question,
        engine=embedding_model_name
    )

    # Extract the embeddings from the response
    question_embeddings = response["data"][0]["embedding"]

    # Calculate distances between the question embedding and dataset embeddings
    from sklearn.metrics.pairwise import cosine_distances

    def distances_from_embeddings(query_embedding, dataset_embeddings, distance_metric="cosine"):
        if distance_metric == "cosine":
            return cosine_distances([query_embedding], list(dataset_embeddings))[0]
        else:
            raise ValueError("Unsupported distance metric")

    distances = distances_from_embeddings(
        question_embeddings,
        df["embeddings"],
        distance_metric="cosine"
    )

    # Add distances to the dataframe and sort by relevance
    df["distances"] = distances
    df.sort_values(by="distances", ascending=True, inplace=True)

    # Get the top 3 most relevant movies
    df_3movies = df.head(3)
    df_3movies_list = df_3movies[["Series_Title", "Overview"]].to_dict(orient="records")

    return df_3movies_list

# Example usage
# USER_QUESTION = "A man is stuck in a dream world where time moves differently, and he has to plant an idea in someone's mind."
# df = pd.read_csv("embeddings.csv", index_col=0)
# df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)

# # Call the function
# relevant_movies = get_relevant_movies(USER_QUESTION, df)
# print("relevant_movies")
# # Print the result
# print(json.dumps(relevant_movies, indent=2))