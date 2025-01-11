from smolagents import CodeAgent, ToolCallingAgent, DuckDuckGoSearchTool, LiteLLMModel, PythonInterpreterTool, tool
from typing import Optional

# model = LiteLLMModel(model_id="ollama/qwen2.5-coder:7b")

model = LiteLLMModel(
    model_id="ollama_chat/llama3.2:3b",
    api_base="http://localhost:11434",  # Adjust if using a remote server
    api_key="openai_should_release_more_open_models"  # Replace with your API key if required
)

@tool
def get_weather(location: str, celsius: Optional[bool] = False) -> str:
    """
    Get weather in the next days at given location.
    Args:
        location: the location
        celsius: whether to use Celsius for temperature
    """
    return f"The weather in {location} is sunny with temperatures around 25°C."

agent = ToolCallingAgent(tools=[get_weather], model=model)


# agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model, additional_authorized_imports=["requests", "re"])

answer = agent.run("What is the weather in Tokyo?")
print(answer)