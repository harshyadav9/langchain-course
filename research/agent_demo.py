#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import certifi
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langsmith import Client
from langchain.tools import tool
import requests


# In[3]:


from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools.render import render_text_description_and_args


# In[4]:


# ==========================================
# LOAD ENV VARIABLES
# ==========================================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WHEATHERSTACK_API_KEY")

# ==========================================
# ENABLE LANGSMITH TRACING
# ==========================================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "default")

# ==========================================
# STREAMLIT PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Agentic AI Assistant")
st.markdown("Search + Weather AI Agent using LangChain")



# In[6]:


search_tool = TavilySearchResults(max_results=2)
# result = search_tool.invoke("what is the capital of france")


@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )

# In[ ]:


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

client = Client()
prompt = client.pull_prompt("hwchase17/react")


tools = [search_tool, get_weather_data]

agent1 = create_react_agent(llm = llm,tools=tools , prompt = prompt, tools_renderer=render_text_description_and_args)


agent_exec = AgentExecutor(agent = agent1, tools=tools , verbose = True, handle_parsing_errors=True)

# ==========================================
# UI INPUT
# ==========================================

user_query = st.text_input(
    "Enter your query:",
    placeholder="Example: Find the capital of India and current weather"
)

# ==========================================
# RUN AGENT
# ==========================================

if st.button("Run Agent"):

    if user_query:

        with st.spinner("Agent is thinking..."):

            try:
                response = agent_exec.invoke({
                    "input": user_query
                })

                st.success("Response Generated")

                st.markdown("## Final Response")
                st.write(response["output"])

            except Exception as e:
                st.error(f"Error: {str(e)}")

    else:
        st.warning("Please enter a query")

