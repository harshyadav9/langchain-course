import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# ENABLE LANGSMITH TRACING
# ==========================================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "default")

from pipelines.pipeline import run_research_pipeline


topic = "The impact of AI on the job market in 2026"
run_research_pipeline(topic)