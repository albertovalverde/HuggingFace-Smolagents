import re
from smolagents import (
CodeAgent,
ToolCallingAgent,
HfApiModel,
ManagedAgent,
DuckDuckGoSearchTool,
LiteLLMModel,
tool
)
from smoltools.jinaai import scrape_page_with_jina_ai, search_facts_with_jina_ai
import os
from dotenv import load_dotenv

load_dotenv()


# model = HfApiModel()
# model = LiteLLMModel(model_id="gpt-4o-mini")
model = LiteLLMModel(model_id="gemini/gemini-2.0-flash-exp")

web_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), scrape_page_with_jina_ai],
    model=model,
    max_steps=10,
)

managed_web_agent = ManagedAgent(
    agent=web_agent,
    name="search",
    description="Runs web searches for you. Give it your query as an argument.",
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[managed_web_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],
)

# answer = manager_agent.run("What year was the movie 'Rebel Without a Cause' released and who was the star of it?")
answer = manager_agent.run("¿Quién es el director de la película de la guerra de las galaxias?")

print(answer)