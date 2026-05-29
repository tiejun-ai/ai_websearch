# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`ai_websearch` is an AI-driven web search application, aiming to develop a clear and concise demo of an AI Web Search Engine:
  * Input: User Query (Question)
  * Work Flow (RAG Architecure):
    * Search Web with the User Query, obtain top K (K = 10) results
    * Ask LLM to generate two Sections:
      * Answer Section: answer the User Query with information faithfully based on search results
        * Do not consider a result if it is considered irrelevant to Question
        * Generate Citation ([Link to the Result URL]) near the answer text if its information is based on the URL
      * Web Search Result Section: 
        * Must list ALL Web Search Results
        * Each Result is shown as a title (link to the URL) and a snippet of the URL content, relevant to User Query
  * Output Format
    * generate as markdown text
    * convert it to html from markdown text, and display it in Colab

## Tech Stack
 * Language: Python

## Code Guidelines
 * Simple and Concise
 * add key comments to explain code clearly

## Architecture 
 * a single ipynb file (ai_websearch.ipynb), open by Google Colab
 * add text cells to explain the logic of the code

## Web Search Tool
 * use Tavily Python API

## LLM Tool
 * use LiteLLM Python SDK

## LLM Model
 * use model as a selectable parameter
 * default model to cheap OpenAI model

