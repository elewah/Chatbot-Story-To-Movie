import streamlit as st
from dotenv import load_dotenv
import os
from movie_search import get_relevant_movies
from fastembed import TextEmbedding

load_dotenv()
from groq import Groq

MODEL_OPTIONS = {
    "Nomic-embed-text-v1.5": {
        "model_name": "nomic-ai/nomic-embed-text-v1.5",
        "embeddings_file": "embeddings_nomic.csv",
        "query_prefix": "search_query: ",
    },
    "BGE-small-en-v1.5": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "embeddings_file": "embeddings_bge.csv",
        "query_prefix": "",
    },
    "MiniLM-L6-v2": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "embeddings_file": "embeddings_minilm.csv",
        "query_prefix": "",
    },
}


@st.cache_resource
def load_embedding_model(model_name):
    return TextEmbedding(model_name)


with st.sidebar:
    "⚠️ Note: This is a demo showcasing the application of the RAG (Retrieval-Augmented Generation) approach in a real-world example. Built to highlight the developer's skills."
    "[![View the source code](https://img.shields.io/badge/Source%20Code-GitHub-blue?logo=github&logoColor=white)](https://github.com/elewah/Chatbot-Story-To-Movie)"

    selected_model_label = st.radio(
        "Embedding Model",
        options=list(MODEL_OPTIONS.keys()),
        index=1,
    )

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            min-width: 400px;
            max-width: 450px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Example Usage:
        **User Prompt:** "A movie where a young person who became rich in a short time by making fake cheques"

        **Response:**
        One movie that comes to mind based on your description is "Catch Me If You Can" (2002) directed by Steven Spielberg. The film is based on the true story of Frank Abagnale Jr., played by Leonardo DiCaprio, who becomes a millionaire by writing fake checks and impersonating a pilot, doctor, and lawyer, all before the age of 19.

        Another movie that might fit your description is "The Wolf of Wall Street" (2013), also based on a true story. The film, directed by Martin Scorsese, tells the story of Jordan Belfort, played by Leonardo DiCaprio, who becomes a wealthy stockbroker by engaging in fraudulent activities, including writing fake checks.

        Additionally, "Boiler Room" (2000) and "Wall Street" (1987) also involve themes of young characters getting rich quickly through questionable means, but they might not specifically involve fake checks.
        """
    )

    st.markdown(
        """
        [![GitHub UI Template](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)
        """
    )

selected_config = MODEL_OPTIONS[selected_model_label]
embedding_model = load_embedding_model(selected_config["model_name"])

st.title("💬 Story To Movie Chatbot")



if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi! I am a chatbot that recommends movies based on your description. How can I assist you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("Please set the GROQ_API_KEY environment variable.")
    st.stop()
client = Groq(api_key=groq_api_key)

if user_prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.chat_message("user").write(user_prompt)

    with st.spinner("Finding relevant movies..."):
        top_movies = get_relevant_movies(
            user_prompt,
            model=embedding_model,
            embeddings_file=selected_config["embeddings_file"],
            query_prefix=selected_config["query_prefix"],
        )

    prompt_template = f"""You are a movie recommendation system. You will be given a list of movies and their descriptions. Based on the descriptions, you will recommend the best movie that matches the user's question.

    The user question is: {user_prompt}

    The list of movies and their descriptions is:
    {top_movies}

    You answer the question by giving the name of the movie and a short description of it.

    Answer:"""

    with st.spinner("Generating response..."):
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_template}],
            model="llama-3.3-70b-versatile",
        )

    answer = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
