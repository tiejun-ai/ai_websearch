"""
AI Web Search (v1) — minimal RAG-based demo.

Workflow:
  1. Web Search  — Tavily fetches top-K results (title, URL, snippet)
  2. LLM Generation — single LiteLLM call produces an answer with inline
     citations ([n](URL)) and a numbered Web Search Results section
  3. Output — answer written to a markdown file
"""

import os
import litellm
from tavily import TavilyClient

# --- Configuration -----------------------------------------------------------
TAVILY_API_KEY = "tvly-..."   # your Tavily API key
OPENAI_API_KEY = "sk-..."     # your OpenAI API key

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # make key available to LiteLLM

MODEL = "gpt-4o-mini"  # change to e.g. "gpt-4o", "claude-3-haiku-20240307"
TOP_K = 10             # number of web search results to retrieve

litellm.set_verbose = False  # suppress debug output


# --- Step 1: Web Search ------------------------------------------------------
def search_web(query, k=TOP_K):
    """Search the web and return top-k results from Tavily."""
    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(query, max_results=k)
    # Each result dict: {"title": str, "url": str, "content": str}
    return response["results"]


# --- Step 2: Generate Answer with LLM ----------------------------------------
def generate_answer(query, results, model=MODEL):
    """Send query + search results to LLM; return a markdown-formatted answer."""
    # Format each result for the prompt, numbered 1..n
    results_text = "\n\n".join(
        f"[{i+1}] Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
        for i, r in enumerate(results)
    )
    n = len(results)

    system_prompt = (
        "You are a helpful research assistant. "
        "Answer the user's query faithfully using ONLY the provided search results. "
        "Do not consider results that are irrelevant to the query. "
        "Add inline citations using the result number as the anchor text: [n](URL) "
        "(e.g. [1](https://example.com)) near supporting text. "
        f"End with a '## Web Search Results' section listing ALL {n} results as:\n"
        "### n. [Title](URL)\nSnippet text"
    )
    user_prompt = f"Question: {query}\n\nSearch Results:\n{results_text}"

    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return response.choices[0].message.content  # markdown string


# --- Step 3: Save Answer as Markdown -----------------------------------------
def save_answer(answer_md, output_file="answer.md"):
    """Write the markdown answer to a file."""
    with open(output_file, "w") as f:
        f.write(answer_md)
    print(f"Answer saved to {output_file}")


# --- Main --------------------------------------------------------------------
if __name__ == "__main__":
    query = "What are the latest AI breakthroughs in 2025?"  # <- change this

    results = search_web(query)                    # Step 1: fetch top-K web results
    answer_md = generate_answer(query, results)    # Step 2: LLM answer + citations
    save_answer(answer_md)                         # Step 3: write markdown file
