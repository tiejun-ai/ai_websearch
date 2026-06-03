# ai_websearch

  * AI-driven Web Search Version 2: Agent Loop with Tool Use
    1. Web Search defined as a **tool** for the LLM (via Tavily API)
    2. **Agent Loop**: LLM decides when and what to search; results accumulate and deduplicate across iterations
    3. Multiple LLM calls per request (up to `MAX_LOOP_TIMES`, default 3)
    4. LLM generates Answer with inline citations `[n](URL)` based on all collected Search Results

## Files

| File | Description |
|------|-------------|
| [ai_websearch.ipynb](ai_websearch.ipynb) | Jupyter notebook — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tiejun-ai/ai_websearch/blob/main/v2/ai_websearch.ipynb) |
| [ai_websearch.py](ai_websearch.py) | Python script version |
| [ai_websearch.ipynb.demo.pdf](ai_websearch.ipynb.demo.pdf) | Example output from a demo run of the notebook |
| [ai_websearch.ipynb.demo.chinese.pdf](ai_websearch.ipynb.demo.chinese.pdf) | Example output from a demo run on a Chinese query |
| [CLAUDE.md](CLAUDE.md) | Claude Code instructions — project context, tech stack, architecture, and code guidelines |
