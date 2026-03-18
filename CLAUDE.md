# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG-based movie recommendation chatbot. Users describe a story/plot and the system recommends matching movies from the IMDB Top 1000 dataset. Supports two fastembed embedding models (all-MiniLM-L6-v2 and BAAI/bge-small-en-v1.5) switchable via sidebar toggle for comparison. Uses Groq LLM for generating recommendation responses. No OpenAI API key required.

## Architecture

**Pipeline:** User query → fastembed local embedding (384-dim, model selected via sidebar) → cosine similarity against pre-computed movie embeddings → top 3 movies → Groq LLM (llama-3.3-70b-versatile) generates recommendation response.

- **main.py** — Streamlit web app. Sidebar radio button selects between two embedding models. Loads selected fastembed model via `@st.cache_resource` (keyed by model name), calls `movie_search.get_relevant_movies()` with the corresponding embeddings file, sends results + user query to Groq LLM.
- **movie_search.py** — Core search module. `load_embeddings(embeddings_file)` loads pre-computed embeddings via `@st.cache_data` (keyed by file path). `get_relevant_movies(user_question, model, embeddings_file)` generates query embedding, computes cosine distances, returns top 5 movies.
- **embeddings_minilm.csv** — 1000 pre-computed 384-dim embeddings (all-MiniLM-L6-v2).
- **embeddings_bge.csv** — 1000 pre-computed 384-dim embeddings (BAAI/bge-small-en-v1.5).
- **regenerate_embeddings.py** — Downloads IMDB dataset via kagglehub and generates both embedding CSVs.
- **project.ipynb** — Notebook showing the original embeddings generation pipeline (OpenAI, historical reference).

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt
pip install streamlit

# Run (requires .env with GROQ_API_KEY)
streamlit run main.py --server.enableCORS false --server.enableXsrfProtection false
```

Access at `http://localhost:8501`.

## Docker

```bash
docker build -t test-chatbot-story-to-movie:latest .
docker run -p 8501:8501 test-chatbot-story-to-movie:latest
```

## API Configuration

- Embedding models (switchable via sidebar): `sentence-transformers/all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5` via fastembed (local, no API key needed)
- LLM: Groq `llama-3.3-70b-versatile` (requires `GROQ_API_KEY`)

## CI/CD

GitHub Actions workflow (`.github/workflows/docker-build.yml`) builds the Docker image on push/PR to main. DockerHub push is currently disabled (`push: false`). Requires `DOCKER_USERNAME` and `DOCKER_PASSWORD` secrets.
