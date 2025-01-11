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
from smoltools.jinaai import scrape_page_with_jina_ai
import os
from dotenv import load_dotenv
import torch
from models import build_model
from kokoro import generate
from IPython.display import display, Audio

# Load environment variables
load_dotenv()

# Initialize the model
model = LiteLLMModel(model_id="gemini/gemini-2.0-flash-exp")

# Set up agents
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

# Load Kokoro model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = build_model('kokoro-v0_19.pth', device)
VOICE_NAME = 'af'  # Default voice
VOICEPACK = torch.load(f'voices/{VOICE_NAME}.pt', weights_only=True).to(device)
print(f"Loaded voice: {VOICE_NAME}")

# Function to convert text to audio
def text_to_audio(text):
    audio, out_ps = generate(MODEL, text, VOICEPACK, lang=VOICE_NAME[0])
    return audio, out_ps

# Agent query
query = "¿Quién es el director de la película de la guerra de las galaxias?"
answer = manager_agent.run(query)

# Generate audio from the answer
print(f"Agent's Answer: {answer}")
audio, phonemes = text_to_audio(answer)

# Play the audio
display(Audio(data=audio, rate=24000, autoplay=True))