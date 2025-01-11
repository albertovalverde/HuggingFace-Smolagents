from smolagents import ToolCallingAgent, HfApiModel, DuckDuckGoSearchTool

model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

agent = ToolCallingAgent(tools=[DuckDuckGoSearchTool()], model=model)

answer = agent.run("Could you get me the title of the page at url 'https://medium.com/@alberto.valverde.escribano/about'?")
print(answer)