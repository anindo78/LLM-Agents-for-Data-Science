'''
Create Local AI Agents with Qwen 3
Beyond answering questions based on provided text, LLMs can act as the reasoning engine for AI agents. Agents can plan sequences of actions,
interact with external tools (like functions or APIs), and work towards accomplishing more complex goals assigned by the user.

Qwen 3 models were specifically designed with strong tool-calling and agentic capabilities. While Alibaba provides the Qwen-Agent framework, 
this tutorial will continue using LangChain for consistency and because its integration with Ollama for agent tasks is more readily documented
in the provided materials.

We will build a simple agent that can use a custom Python function as a tool.
'''

import os
from dotenv import load_dotenv
from langchain.agents import tool
import datetime


# load environment variables from .env file
load_dotenv()

@tool
def current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Returns the current date and time, formatted according to the provided Python strftime format string.
    Use this tool whenever the user asks for the current date, time, or both.
    Example format strings: '%Y-%m-%d' for date, '%H:%M:%S' for time.
    If no format is specified, defaults to '%Y-%m-%d %H:%M:%S'.
    """
    try:
        return datetime.datetime.now().strftime(format)
    except Exception as e:
        return f"Error formatting date/time: {e}"   
    

print("Custom tool to get current time defined.")

import pandas as pd
@tool
def summarize_dataset(file_path: str) -> str:
    """
    Reads a CSV file and summarizes the dataset.
    Returns number of rows, columns, and a list of column names with data types.
    Use this tool when the user uploads or references a dataset and wants a quick overview.
    """
    try:
        df = pd.read_csv(file_path, nrows=1000)  # read sample for speed
        summary = {
            "rows (sampled)": len(df),
            "columns": len(df.columns),
            "column_info": df.dtypes.to_dict()
        }
        return str(summary)
    except Exception as e:
        return f"Error reading dataset: {e}"

# Add to your tools list

print("\nSummarize dataset tool defined.")


# creating a news scraping tool
import requests
from bs4 import BeautifulSoup

@tool
def get_yahoo_headlines(limit: int = 5) -> str:
    """
    Scrapes Yahoo News and returns the top headlines.
    Args:
        limit: number of headlines to return (default = 5).
    """
    try:
        url = "https://news.yahoo.com/"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        # Yahoo headlines are usually in <h3> or <h2> with links
        headlines = [h.get_text(strip=True) for h in soup.select("h3 a, h2 a")]

        top_headlines = headlines[:limit]
        return "\n".join([f"{i+1}. {h}" for i, h in enumerate(top_headlines)])
    except Exception as e:
        return f"Error fetching headlines: {e}"

# Add to tools list
print("Yahoo News headlines tool defined.")

tools = [summarize_dataset, current_time, get_yahoo_headlines]


'''
Set up the Agent LLM with Ollama

Instantiate the ChatOllama model again, using a Qwen 3 variant suitable for tool calling. The qwen3:8b model should be capable of 
handling simple tool use cases.

It's important to note that tool calling reliability with local models served via Ollama can sometimes be less consistent than with 
large commercial APIs like GPT-4 or Claude. The LLM might fail to recognize when a tool is needed, 
hallucinate arguments, or misinterpret the tool's output. Starting with clear prompts and simple tools is recommended.
'''


from langchain_ollama import ChatOllama

def get_agent_llm():
    """Instantiate the ChatOllama model for agent tasks."""
    model_name = os.getenv("OLLAMA_MODEL", "qwen3:4b")
    llm = ChatOllama(model=model_name, temperature=0)

    print (f"Agent LLM instantiated with model: {model_name}")
    return llm

# agent_llm = get_agent_llm() # Call this later



'''
Create the agent prompt:

Agents require specific prompt structures that guide their reasoning and tool use. The prompt typically includes
placeholders for user input (input), conversation history (chat_history), and the agent_scratchpad. The scratchpad
is where the agent records its internal "thought" process, the tools it decides to call, and the results (observations)
it gets back from those tools. LangChain Hub provides pre-built prompts suitable for tool-calling agents.
'''

from langchain import hub

def get_agent_prompt():
    """Load a prompt template for tool-using agents from LangChain Hub."""
    prompt = hub.pull("hwchase17/openai-tools-agent")
    # This prompt is designed for OpenAI but often works well with other tool-calling models.
    # Alternatively, define a custom ChatPromptTemplate.

    print("Agent prompt loaded from LangChain Hub.")
    return prompt

# agent_prompt = get_agent_prompt() # Call this later



'''Set up the agent executor:

Finally, we can create the agent executor that ties together the LLM, tools, and prompt. The executor manages the interaction flow
'''
from langchain.agents import create_tool_calling_agent

def build_agent(llm, tools, prompt):
    """Create the agent executor with the LLM, tools, and prompt."""

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    print("Agent runnable created.")
    
    return agent


''' Create the Agent Executor
The AgentExecutor is responsible for running the agent loop. It takes the agent runnable and the tools,
invokes the agent with the input, parses the agent's output (which could be a final answer or a tool call request), 
executes any requested tool calls, and feeds the results back to the agent until a final answer is reached. 


Setting verbose=True is highly recommended during development to observe the agent's step-by-step execution flow.
'''
from langchain.agents import AgentExecutor

def create_agent_executor(agent, tools):
    """Wrap the agent in an executor to handle tool calls and responses."""
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True  # Set to True to see detailed reasoning steps
    )
    print("Agent executor created.")
    return agent_executor

# agent_executor = create_agent_executor(agent_runnable, tools) # Call this later


'''
Run the Agent: Invoke the agent executor with a user query that should trigger the use of the defined tool.
'''

def run_agent(agent_executor, query):
    """Run the agent executor with a user query."""

    print(f"\nRunning agent with query: {query}\n")
    result = agent_executor.invoke({"input": query})                                                                                             
    response = result.get("output", result)  
    print(f"\nAgent response: {response}\n")
    return response




# --- Main Execution ---
if __name__ == "__main__":
    # 1. Define Tools (already done above)

    # 2. Get Agent LLM
    agent_llm = get_agent_llm() # Use the chosen Qwen 3 model

    # 3. Get Agent Prompt
    agent_prompt = get_agent_prompt()

    # 4. Build Agent Runnable
    agent_runnable = build_agent(agent_llm, tools, agent_prompt)

    # 5. Create Agent Executor
    agent_executor = create_agent_executor(agent_runnable, tools)

    # 6. Run Agent
    run_agent(agent_executor, "What is the current date?")
    run_agent(agent_executor, "What time is it right now? Use HH:MM format.")
    run_agent(agent_executor, "Summarize the dataset located at 'Spotify_Youtube_Dataset.csv'") # Ensure this file exists
    run_agent(agent_executor, "Get the top 3 headlines from Yahoo News.")
    run_agent(agent_executor, "Tell me a joke.") # Should not use the tool