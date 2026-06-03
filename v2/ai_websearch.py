import os, json, sys
import litellm
from tavily import TavilyClient

# --- API Keys ----------------------------------------------------------------
TAVILY_API_KEY = "tvly-..."  # your Tavily API key
OPENAI_API_KEY = "sk-..."    # your OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- Configuration -----------------------------------------------------------
MODEL          = "gpt-4o-mini"  # change to e.g. "gpt-4o", "claude-3-5-haiku-20241022"
TOP_K          = 10             # max web results per search
MAX_LOOP_TIMES = 3              # max LLM calls per request (including final answer)

litellm.set_verbose = False


# --- Step 1: Web Search Tool -------------------------------------------------

def search_web(query, k=TOP_K):
    """Return top-k Tavily results for query."""
    response = TavilyClient(api_key=TAVILY_API_KEY).search(query, max_results=k)
    return response["results"]  # each: {title, url, content, score}


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information on a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"]
        }
    }
}


# --- Step 2: Agent Loop ------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful research assistant with access to a web_search tool. "
    "Use it to find information needed to answer the user's query; you may search multiple times with different queries. "
    "Each search result is labeled [n] with its URL. "
    "Cite supporting text in your answer as [n](URL) using those numbers. "
    "When you have enough information, write a clear, comprehensive markdown answer."
)


def run_agent(query, model=MODEL, max_loops=MAX_LOOP_TIMES):
    """Run the agent loop; return (answer_text, all_results, model_name)."""
    seen_urls = {}   # url -> result dict (augmented with "num" key)
    counter   = 0    # global discovery counter across all searches
    messages  = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": query},
    ]

    last_content = ""
    model_name   = model
    for loop_num in range(max_loops):
        print(f"\n[Loop {loop_num + 1}] Calling LLM...", file=sys.stderr)
        response = litellm.completion(
            model=model, messages=messages,
            tools=[WEB_SEARCH_TOOL], tool_choice="auto"
        )
        msg        = response.choices[0].message
        model_name = response.model  # actual model name returned by the API
        last_content = msg.content or ""

        if not msg.tool_calls:  # LLM produced a plain answer — exit loop
            print(f"[Loop {loop_num + 1}] LLM returned answer. Exiting loop.", file=sys.stderr)
            return last_content, list(seen_urls.values()), model_name

        # --- Execute tool calls and add results to context ---
        messages.append(msg)
        for tc in msg.tool_calls:
            search_query = json.loads(tc.function.arguments)["query"]
            print(f"[Loop {loop_num + 1}] Tool call: web_search('{search_query}')", file=sys.stderr)
            results = search_web(search_query)
            print(f"[Loop {loop_num + 1}] Got {len(results)} results", file=sys.stderr)

            tool_text = ""
            for r in results:
                if r["url"] not in seen_urls:  # deduplicate; first occurrence wins
                    counter += 1
                    r["num"] = counter
                    seen_urls[r["url"]] = r
                num = seen_urls[r["url"]]["num"]
                tool_text += f"[{num}] {r['title']}\nURL: {r['url']}\n{r['content']}\n\n"

            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "name": "web_search", "content": tool_text
            })

    # Max loops reached — return whatever the LLM last said
    print(f"[Loop {max_loops}] Max loops reached. Returning last LLM response.", file=sys.stderr)
    return last_content, list(seen_urls.values()), model_name


# --- Step 3: Format Output ---------------------------------------------------

def format_output(answer_text, all_results, model_name):
    """Return final markdown string with Answer and Web Search Results sections."""
    # Display results in ascending [n] (discovery) order
    ordered = sorted(all_results, key=lambda r: r["num"])

    results_md = "## Web Search Results\n\n"
    for r in ordered:
        results_md += f"- [{r['num']}] [{r['title']}]({r['url']})\n"

    return f"## Answer *(model: {model_name})*\n\n{answer_text}\n\n---\n\n{results_md}"


# --- Entry Point -------------------------------------------------------------

if __name__ == "__main__":
    query = "How was Claude Code implemented?"
    answer_text, all_results, model_name = run_agent(query)
    print(format_output(answer_text, all_results, model_name))
