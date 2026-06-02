# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`ai_websearch` is an AI-driven web search demo.

Input: User Query (Question)
Output Generation: 
  1. Web Search Result Section: list ALL Unique Web Search Results with:
    * Assigned result ranking number '[n]' based on discovery order
    * URL title as a clickable URL link
    * Summary snippet of the URL content
  2. LLM Answer Section: a markdown Answer to User Query with inline citations (`[n](URL)`), around texts supported by the URL `[n]`.

## Architecture (Agent Loop, Tool Use, Multiple LLM Calls):

1. Tool Use: define a function for Web Search via Tavily (top K <= 10 results) as tool for LLM
2. Run Agent Loop:
  * Init the Context with User Query
  * ask LLM with User Context to generate Output, providing the Web Search tool for LLM to use
  * check LLM response:
    * If the response is Tool Use (Web Search with LLM generated query), do the Web Search, collect the Search Results, add them to Context, and continue the Agent Loop
    * Otherwise, exit the Agent Loop and return the LLM response as Answer
  * add MAX_LOOP_TIMES (default=3) paramater, as maximal times the loop can run, which is also maximal number of LLM calls a request can incur.
3. Generate Output as markdown, convert it to html, and display html in Jupyter notebook.

## Tech Stack

- Language: Python
- Web search: Tavily Python SDK (`TavilyClient`)
- LLM: LiteLLM Python SDK (model is a selectable parameter, default to a cheap OpenAI model)

## Code Structure

  * a single Jupyter notebook file

## Code Guidelines

- Simple and concise; add key comments to explain non-obvious logic
- Keep text cells in notebooks to explain reasoning between code cells
