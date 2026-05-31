# Story To Movie — RAG Tutorial

A beginner tutorial on **Retrieval-Augmented Generation (RAG)** built around a movie recommender: describe a plot, get the matching film.

## Tutorial

| Part | Topic | Open |
|------|-------|------|
| **Part 1** | How RAG works — embeddings, cosine similarity, LLM prompting | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elewah/Chatbot-Story-To-Movie/blob/main/project.ipynb) |
| **Part 2** | Building & hosting the app — Streamlit, Docker, CI/CD | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elewah/Chatbot-Story-To-Movie/blob/main/application.ipynb) |

## What You'll Learn

- What RAG is and why it works better than asking an LLM to recall facts
- How to turn text into embeddings and measure similarity with cosine distance
- How to build a chat interface with Streamlit
- How to deploy the app for free on Streamlit Cloud, containerize it with Docker, and automate builds with GitHub Actions

## Prerequisites

- Basic Python
- A free [Groq API key](https://console.groq.com) (takes 30 seconds to get)

## Run Locally

```bash
git clone https://github.com/elewah/Chatbot-Story-To-Movie.git
cd Chatbot-Story-To-Movie
pip install -r requirements.txt
echo 'GROQ_API_KEY=your_key_here' > .env
streamlit run main.py
```

Access at [http://localhost:8501](http://localhost:8501).

## Dataset

[IMDB Top 1000 Movies and TV Shows](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows) from Kaggle.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
