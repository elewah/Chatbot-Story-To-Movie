import streamlit as st
# from openai import OpenAI
from dotenv import load_dotenv
import openai
import os
from movie_search import get_relevant_movies
load_dotenv()
from groq import Groq
api_key = os.environ.get("GROQ_API_KEY")
client = Groq(
    api_key=api_key,
)

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Explain the importance of fast language models",
#         }
#     ],
#     model="llama-3.3-70b-versatile",
# )

# print(chat_completion.choices[0].message.content)

openai.api_base = "https://openai.vocareum.com/v1"
openai_api_key = os.getenv("OPENAI_API_KEY")
with st.sidebar:
    # openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    # "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[![View the source code](https://img.shields.io/badge/Source%20Code-GitHub-blue?logo=github&logoColor=white)](https://github.com/elewah/Chatbot-Story-To-Movie)"
    # "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"
    # Increase the sidebar default width using Streamlit's custom CSS
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

st.title("💬 Store To Movie Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi! I am a chatbot that recommends movies based on your description. How can I assist you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.chat_message("user").write(user_prompt)
    st.session_state.messages.append({"role": "assistant", "content": "Searching for relevant movies..."})
    df_3movies_list_str= get_relevant_movies(user_prompt, embedding_model_name="text-embedding-ada-002")
    st.session_state.messages.append({"role": "assistant", "content": "Generating response..."})

    # Create the prompt template
    prompt_template = f"""You are a movie recommendation system. You will be given a list of movies and their descriptions. Based on the descriptions, you will recommend the best movie that matches the user's question.

    The user question is: {user_prompt}  

    The list of movies and their descriptions is:
    {df_3movies_list_str}

    You answer the question by giving the name of the movie and a short description of it.

    Answer:"""

    # Append the prompt to Streamlit's session state messages
    # st.session_state.messages.append({"role": "user", "content": prompt_template})

    # Display the prompt in the chat interface
    # st.chat_message("user").write(prompt_template)

    # Call the LLM
    response = client.chat.completions.create(
        messages=st.session_state.messages,
        model="llama-3.3-70b-versatile",
    )

    # Optionally, show the model's response
    st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})
    st.chat_message("assistant").write(response.choices[0].message.content)

    # response = client.chat.completions.create(
    #     messages=st.session_state.messages,
    #     model="llama-3.3-70b-versatile",
    # )
    # # response = openai.chat.completions.create(model="gpt-3.5-turbo-instruct", messages=st.session_state.messages)
    # msg = response.choices[0].message.content
    # st.session_state.messages.append({"role": "assistant", "content": msg})
    # st.chat_message("assistant").write(msg)