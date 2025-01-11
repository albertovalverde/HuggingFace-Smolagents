from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    LiteLLMModel,
    ManagedAgent,
    DuckDuckGoSearchTool,
)
from smoltools.jinaai import scrape_page_with_jina_ai, search_facts_with_jina_ai
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the model
model = LiteLLMModel(model_id="gpt-4o-mini")

# Research Agent
research_agent = ToolCallingAgent(
    tools=[scrape_page_with_jina_ai, search_facts_with_jina_ai, DuckDuckGoSearchTool()],
    model=model,
    max_steps=10,
)

managed_research_agent = ManagedAgent(
    agent=research_agent,
    name="super_researcher",
    description="Researches topics thoroughly using web searches and content scraping. Provide the research topic as input.",
)

# Research Checker Agent
research_checker_agent = ToolCallingAgent(
    tools=[],
    model=model
)

managed_research_checker_agent = ManagedAgent(
    agent=research_checker_agent,
    name="research_checker",
    description="Checks the research for relevance to the original task request. If the research is not relevant, it will ask for more research.",
)

# Writer Agent
writer_agent = ToolCallingAgent(
    tools=[],
    model=model
)

managed_writer_agent = ManagedAgent(
    agent=writer_agent,
    name="writer",
    description="Writes blog posts based on the checkedresearch. Provide the research findings and desired tone/style.",
)

# Copy Editor Agent
copy_editor_agent = ToolCallingAgent(
    tools=[],
    model=model
)

managed_copy_editor = ManagedAgent(
    agent=copy_editor_agent,
    name="editor",
    description="Reviews and polishes the blog post based on the research and original task request. Order the final blog post and any lists in a way that is most engaging to someone working in AI. Provides the final, edited version in markdown.",
)

# Main Blog Writer Manager
blog_manager = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[managed_research_agent, managed_research_checker_agent, managed_writer_agent, managed_copy_editor],
    additional_authorized_imports=["re"],

    # system_prompt="""You are a blog post creation manager. Coordinate between research, writing, and editing teams.
    # Follow these steps:
    # 1. Use research_agent to gather information
    # 2. Pass research to research_checker_agent to check for relevance
    # 3. Pass research to writer_agent to create the initial draft
    # 4. Send draft to editor for final polish
    # 4. Save the final markdown file
    # """
)

def write_blog_post(topic, output_file="blog_post.md"):
    """
    Creates a blog post on the given topic using multiple agents
    
    Args:
        topic (str): The blog post topic or title
        output_file (str): The filename to save the markdown post
    """
    result = blog_manager.run(f"""Create a blog post about: {topic}
    1. First, research the topic thoroughly, focus on specific products and sources
    2. Then, write an engaging blog post not just a list
    3. Finally, edit and polish the content
    """)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)
    print(f"Blog post has been saved to {output_file}")
    
    return result

# print(blog_manager.system_prompt_template)
topic = "Create a blog post about the top 5 recent developments in jurisprudence and AI in 2025. Please include specific titles and sources"
print(topic)
write_blog_post(topic)

