# Chatbot-Story-To-Movie

## Project Overview
The "Chatbot-Story-To-Movie" project utilizes a Retrieval-Augmented Generation (RAG) approach to recommend movies based on user-provided descriptions or questions. By combining advanced natural language processing (NLP) techniques with embeddings, the system retrieves relevant movie descriptions from a dataset of top-rated films and generates detailed recommendations using OpenAI's GPT models. This hybrid method ensures both accuracy and contextual relevance in the recommendations.

## Objectives
- Build a movie recommendation system that matches user queries to movie descriptions.
- Utilize embeddings to measure the similarity between user input and movie descriptions.
- Implement a pipeline to process, analyze, and recommend movies based on cosine similarity.
- Demonstrate the use of OpenAI's GPT models for generating embeddings and recommendations.

## Files Description
- **project.ipynb**: The main Jupyter Notebook containing the code for the entire project, including data preparation, embeddings generation, and recommendation logic.
- **application.ipynb**: This notebook is designed to provide an interactive experience for users to get movie recommendations based on their input descriptions. It leverages OpenAI's advanced language models for both embedding generation and natural language completion.
- **distances.csv**: A CSV file containing the calculated cosine distances between user queries and movie embeddings.
- **embeddings.csv**: A CSV file storing the embeddings for all movies in the dataset.
- **Movie Description Help.pdf**: A document providing additional context or guidelines for movie descriptions.

## Dataset
The dataset used in this project is the "IMDB Dataset of Top 1000 Movies and TV Shows" sourced from Kaggle. This dataset was chosen because:
- It contains detailed information about top-rated movies and TV shows, including titles, genres, and overviews.
- The dataset is well-structured and suitable for NLP tasks.
- It provides a rich variety of movie descriptions, making it ideal for building a recommendation system.

## Technologies Used
- **Python**: The primary programming language for the project.
- **Jupyter Notebook**: For interactive development and visualization.
- **OpenAI API**: To generate embeddings and recommendations using GPT models.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **tiktoken**: For tokenization and managing token limits.
- **Tenacity**: For handling retries in API calls.

## Applied Skills
- Natural Language Processing (NLP): Extracting and analyzing text data.
- Machine Learning: Using embeddings to measure similarity.
- Data Analysis: Cleaning, processing, and visualizing data.
- API Integration: Leveraging OpenAI's API for embeddings and completions.
- Software Development: Building a modular and reusable codebase.

## Project Workflow
1. **Data Preparation**:
   - Download the dataset from Kaggle.
   - Load and preprocess the data using Pandas.
2. **Embeddings Generation**:
   - Use OpenAI's `text-embedding-ada-002` model to generate embeddings for movie descriptions.
   - Store the embeddings in a CSV file for future use.
3. **Similarity Calculation**:
   - Compute cosine distances between user queries and movie embeddings.
   - Sort movies by relevance based on the calculated distances.
4. **Recommendation System**:
   - Build a custom text prompt for GPT-3.5 to recommend the best movie based on user input.
   - Generate responses using OpenAI's completion API.
5. **Evaluation**:
   - Test the system with various user queries to ensure accuracy and relevance.

## How to Run the Project
1. Clone the repository to your local machine.
2. Set up a Python virtual environment and install the required dependencies using `pip install -r requirements.txt`.
3. Obtain an OpenAI API key and set it as an environment variable (`OPENAI_API_KEY`).
4. Open `project.ipynb` in Jupyter Notebook and run the cells sequentially.

## Future Improvements
- Expand the dataset to include more movies and TV shows.
- Optimize the embeddings generation process for faster computation.
- Integrate a user-friendly web interface for real-time recommendations.
- Explore additional NLP techniques to enhance recommendation accuracy.

## Acknowledgments
- Kaggle for providing the IMDB dataset.
- OpenAI for their powerful API and models.
- Udacity for their coding exercises, which inspired and enhanced the development of this project.


## License
This project is licensed under the Apache License, Version 2.0, January 2004. See the LICENSE file included in the repository for details.
