import json
from langchain_core.messages import HumanMessage, ToolMessage
from agents.agents import build_search_agent, build_reader_agent, writer_chain, critic_chain


def tool_artifacts(result: dict, tool_name: str) -> list:
    """Read original tool evidence, excluding limit/error messages without artifacts."""
    return [message.artifact for message in result["messages"]
            if isinstance(message, ToolMessage) and message.name == tool_name
            and message.artifact is not None]


def selected_reader_evidence(reader_result: dict, candidates: list[dict]) -> dict:
    """Resolve the reader's selection to original, successful tool evidence."""
    selection = reader_result.get("structured_response")
    if selection is None or selection.selected_url is None:
        raise RuntimeError("Reader did not select a suitable extracted source.")
    if selection.selected_url not in {candidate["url"] for candidate in candidates}:
        raise RuntimeError("Reader selected a URL outside the search candidates.")
    for result in reversed(tool_artifacts(reader_result, "scrape_url")):
        if (result.get("url") == selection.selected_url
                and result.get("status") == "ok"
                and result.get("content", "").strip()):
            return result
    raise RuntimeError("Selected URL has no successful, non-empty extraction.")


def run_research_pipeline(topic : str) -> dict:

    state={}
    print("\n"+"="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages" : [HumanMessage(f"Find recent, reliable and detailed information about: {topic}")]
    })
    candidates = {}
    for batch in tool_artifacts(search_result, "web_search"):
        for candidate in batch:
            candidates.setdefault(candidate["url"], candidate)
    state["search_candidates"] = list(candidates.values())
    if not state["search_candidates"]:
        raise RuntimeError("Search returned no source candidates. Try a more specific topic.")
    state["search_results"] = json.dumps(state["search_candidates"], ensure_ascii=False)
    state["search_summary"] = search_result["messages"][-1].content

    print("\n search result ",state['search_results'])



    #step 2 - reader agent
    print("\n"+"="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [HumanMessage(
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results']}"
        )]
    })

    state["reader_summary"] = reader_result["messages"][-1].content
    state["scrape_results"] = tool_artifacts(reader_result, "scrape_url")
    if not state["scrape_results"]:
        raise RuntimeError("Reader did not extract a source within its call budget.")
    state["scraped_content"] = json.dumps(state["scrape_results"], ensure_ascii=False)
    state["selected_source"] = selected_reader_evidence(reader_result, state["search_candidates"])
    state["selection_reason"] = reader_result["structured_response"].reason


    print("\nscraped content: \n", state['scraped_content'])


    #step 3 - writer chain

    print("\n"+" ="*50)
    print("step 3 - Writer is drafting the report ...")
    print("="*50)

    # Keep all attempts in state for debugging; send only selected evidence to the writer.
    research_combined = (
        "SELECTED SOURCE EVIDENCE (one source only; note any truncation):\n"
        + json.dumps(state["selected_source"], ensure_ascii=False)
    )

    state["report"] = writer_chain.invoke({
        "topic" : topic,
        "research" : research_combined
    })

    print("\n Final Report\n",state['report'])

    #critic report

    print("\n"+" ="*50)
    print("step 4 - critic is reviewing the report ")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
        "report":state['report'],
        "research": research_combined
    })

    print("\n critic report \n", state['feedback'])

    return state
