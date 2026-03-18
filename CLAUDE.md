# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG-based movie recommendation chatbot. Users describe a story/plot and the system recommends matching movies from the IMDB Top 1000 dataset. Uses local fastembed embeddings (all-MiniLM-L6-v2) for similarity search and Groq LLM for generating recommendation responses. No OpenAI API key required.

## Architecture

**Pipeline:** User query → fastembed local embedding (all-MiniLM-L6-v2, 384-dim) → cosine similarity against pre-computed movie embeddings (`embeddings.csv`) → top 3 movies → Groq LLM (llama-3.3-70b-versatile) generates recommendation response.

- **main.py** — Streamlit web app. Loads fastembed model via `@st.cache_resource`, manages chat session state, calls `movie_search.get_relevant_movies()`, sends results + user query to Groq LLM via prompt template.
- **movie_search.py** — Core search module. Loads pre-computed embeddings from `embeddings.csv` via `@st.cache_data`, generates query embedding via fastembed, computes cosine distances (scikit-learn), returns top 3 movies with titles and overviews.
- **embeddings.csv** — 1000 pre-computed 384-dim embeddings (all-MiniLM-L6-v2). This is the "database" — no external DB.
- **regenerate_embeddings.py** — One-time script to regenerate embeddings.csv using fastembed.
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

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` via fastembed (local, no API key needed)
- LLM: Groq `llama-3.3-70b-versatile` (requires `GROQ_API_KEY`)

## CI/CD

GitHub Actions workflow (`.github/workflows/docker-build.yml`) builds the Docker image on push/PR to main. DockerHub push is currently disabled (`push: false`). Requires `DOCKER_USERNAME` and `DOCKER_PASSWORD` secrets.
