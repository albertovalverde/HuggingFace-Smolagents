from smolagents import CodeAgent, ToolCallingAgent, DuckDuckGoSearchTool, LiteLLMModel, PythonInterpreterTool, tool, TOOL_CALLING_SYSTEM_PROMPT
from typing import Optional
import os

from dotenv import load_dotenv
load_dotenv()

model = LiteLLMModel(model_id="gemini/gemini-2.0-flash-exp",
                     api_key=os.getenv("GEMINI_API_KEY"))



@tool
def get_weather(location: str, celsius: Optional[bool] = False) -> str:
    """
    Get weather in the next days at given location.
    Args:
        location: the location
        celsius: whether to use Celsius for temperature
    """
    return f"The weather in {location} is sunny with temperatures around 7°C."

agent = ToolCallingAgent(tools=[get_weather], model=model, system_prompt=TOOL_CALLING_SYSTEM_PROMPT)


# agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model, additional_authorized_imports=["requests", "bs4"])
answer = agent.run("What is the weather in Tokyo?")
print(answer)