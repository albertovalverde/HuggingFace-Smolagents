from smolagents import CodeAgent, ToolCallingAgent, DuckDuckGoSearchTool, LiteLLMModel, PythonInterpreterTool, tool, TOOL_CALLING_SYSTEM_PROMPT
from typing import Optional
import os

from dotenv import load_dotenv
load_dotenv()

model = LiteLLMModel(model_id="claude-3-5-sonnet-20240620",
                     api_key=os.getenv("ANTHROPIC_API_KEY"))


@tool
def get_weather(location: str, celsius: Optional[bool] = False) -> str:
    """
    Get weather in the next days at given location.
    Args:
        location: the location
        celsius: whether to use Celsius for temperature
    """
    return f"The weather in {location} is sunny with temperatures around 7°C."

# agent = ToolCallingAgent(tools=[get_weather], model=model)


agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)
answer = agent.run("What is the weather in Tokyo?")
print(answer)