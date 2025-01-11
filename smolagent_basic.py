from smolagents import CodeAgent, HfApiModel
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("HF_TOKEN")


model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")
agent = CodeAgent(tools=[], model=model)
answer = agent.run("What is the cube of 2?")
print(answer)