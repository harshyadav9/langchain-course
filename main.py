from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field


from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
# from tavily import TavilyClient

from schemas import AgentResponse
load_dotenv()

class Source(BaseModel):
    """ Schema for source used by agent"""
    url: str = Field(description="The URL of the source")
    
    
class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )

# tavily = TavilyClient()

# @tool
# def search(query:str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query: The query to search for over the internet
#     Returns:
#         The search result
#     """
    
#     print(f"Searching for {query}")
#     return tavily.search(query = query)


llm = ChatOpenAI(model="gpt-5.4-nano")
tools = [TavilySearch()]
agent = create_agent(model=llm , tools = tools , response_format=ToolStrategy(AgentResponse))

def main():
    print("creating a agent that is calling tool !!!")
    result = agent.invoke({"messages":HumanMessage(content="what is the biggest panet in our solar system?")})
    print(result)

if __name__ == "__main__":
    main()
