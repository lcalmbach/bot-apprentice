import streamlit as st
import os
import socket
#from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import load_tools
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from streamlit_chat import message
import openai

import const

APP_NAME = "Bot-Apprentice"
LOCAL_HOST = 'liestal'

def get_var(varname: str) -> str:
    """
    Retrieves the value of a given environment variable or secret from the Streamlit configuration.

    If the current host is the local machine (according to the hostname), the environment variable is looked up in the system's environment variables.
    Otherwise, the secret value is fetched from Streamlit's secrets dictionary.

    Args:
        varname (str): The name of the environment variable or secret to retrieve.

    Returns:
        The value of the environment variable or secret, as a string.

    Raises:
        KeyError: If the environment variable or secret is not defined.
    """
    if socket.gethostname().lower() == LOCAL_HOST:
        return os.environ[varname]
    else:
        return st.secrets[varname]
    

def init():
    st.set_page_config(page_title=APP_NAME, page_icon="🤖")
    openai.api_key  = get_var("OPENAI_API_KEY")


llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)


def show_mesages():
    for m in st.session_state.messages:
        message(m["text"], m["is_user"])


def main():
    init()
    st.header("St@taBot-Apprentice")
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
    if "messages" not in st.session_state:
        first_message = {"text": const.bot_greeting, "is_user": False}
        st.session_state.messages = [first_message]

    show_mesages()
    cols = st.columns([10, 1, 1])
    with cols[0]:
        question = st.text_area(label="Your Question", label_visibility="collapsed")
    with cols[1]:
        if st.button("📨"):
            q = {"text": question, "is_user": True}
            st.session_state.messages.append(q)
            question_context = const.context.format(question)
            response = agent.run(question_context)
            st.write(response)
            r = {"text": response, "is_user": False}
            st.session_state.messages.append(r)
            question = ""
            st._rerun()
    with cols[2]:
        if st.button("🗑️"):
            st.session_state.messages = st.session_state.messages[:1]
            st._rerun()


if __name__ == "__main__":
    main()
