from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools.tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()




# Model Initialization
llm = ChatOpenAI(model = "gpt-5-nano",temperature=0)



def build_search_agent():
    return create_agent(
        model = llm,
        tools=[web_search],
    )
    
    
def build_reader_agent():
    return create_agent(
        model = llm,
        tools=[scrape_url],
    )
    
    
#writer chain 
    
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a professional research analyst and report writer.

Your responsibility is to synthesize the provided research into a clear,
accurate, well-structured report.

Rules:
1. Use only the information contained in the provided research.
2. Do not invent facts, statistics, quotes, sources, or URLs.
3. If the available research does not support a claim, do not make the claim.
4. Clearly distinguish established facts from uncertain or conflicting information.
5. Combine overlapping information rather than repeating it.
6. Preserve important numbers, dates, names, and technical details.
7. Prefer insights supported by multiple sources when possible.
8. Include only URLs that actually appear in the provided research.
9. Do not claim that you visited or verified a source unless the provided research says so.
10. If the research is insufficient, explicitly mention the limitation.

Write in a professional, factual, analytical style.
"""
    ),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])


writer_chain = writer_prompt | llm | StrOutputParser()


#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a sharp, constructive research critic.

Evaluate reports objectively and specifically.
Do not rewrite the report.
Do not invent missing facts.
Base your evaluation only on the report provided."""
    ),
    (
        "human",
        """Review the research report below strictly.

Report:
{report}

Evaluate it based on:
- Accuracy and factual grounding
- Depth of analysis
- Clarity and structure
- Quality of key findings
- Whether conclusions are supported by the report
- Quality and use of sources

Respond in exactly this format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""
    ),
])


critic_chain = critic_prompt | llm | StrOutputParser()