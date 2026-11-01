
"""
===============================================================
              CITY INTELLIGENCE AI AGENT
===============================================================

PROJECT DESCRIPTION:
This project implements an AI-powered City Intelligence Agent
using LangChain, Groq LLM, OpenWeatherMap and Tavily APIs.

The agent can understand natural language queries and
automatically decide which external tools are required.

MAIN FUNCTIONALITIES:

1. WEATHER TOOL:
   - Fetches real-time weather information for a city.
   - Uses the OpenWeatherMap API.
   - Retrieves temperature and weather descriptions.

2. NEWS TOOL:
   - Fetches the latest news about a city.
   - Uses the Tavily Search API.
   - Retrieves news titles, URLs and content snippets.

3. GROQ LLM:
   - Uses the Groq API with the openai/gpt-oss-120b model.
   - Understands user queries.
   - Decides which tools need to be called.
   - Generates the final response.

4. CREATE_AGENT():
   - Uses LangChain's prebuilt agent architecture.
   - Automatically manages the agent execution loop.
   - Handles tool-call detection and tool execution.
   - Manages communication between the LLM and tools.
   - Repeats execution until the final answer is generated.

5. HUMAN-IN-THE-LOOP MIDDLEWARE:
   - Uses @wrap_tool_call middleware.
   - Intercepts tool execution requests.
   - Asks the user for approval before executing a tool.
   - Allows or denies tool execution.

6. CONVERSATION:
   - Uses the agent's message state to process user queries.
   - Sends user messages to the agent.
   - Displays the final response.

===============================================================
IMPORTANT: MANUAL AGENT VS CREATE_AGENT()
===============================================================

MANUAL AGENT IMPLEMENTATION:

In manual agent creation, the developer writes the entire
agent execution workflow.

The developer manually handles:
- LLM invocation using invoke().
- Tool-call detection using result.tool_calls.
- Tool selection and execution.
- ToolMessage creation and handling.
- Conversation history using a messages list.
- Repeating the LLM and tool execution loop.
- Human approval logic.

CREATE_AGENT() IMPLEMENTATION:

In this implementation, LangChain provides a prebuilt
agent runtime that handles the execution workflow.

The developer mainly defines:
- The LLM model.
- Available tools.
- System instructions.
- Middleware for custom execution logic.

ADVANTAGES OF CREATE_AGENT():

1. LESS BOILERPLATE CODE:
   No need to manually write the complete agent loop.

2. AUTOMATIC TOOL ORCHESTRATION:
   Handles tool-call detection, execution and responses.

3. MIDDLEWARE SUPPORT:
   Allows custom logic such as human approval,
   logging and tool-call validation.

4. STATE MANAGEMENT:
   Uses LangGraph-based agent state to manage messages
   and execution across agent steps.

5. EXTENSIBILITY:
   Easier to extend with additional tools, middleware,
   checkpointers and advanced agent workflows.

6. MAINTAINABILITY:
   Separates business logic (tools) from orchestration
   (agent runtime).

IMPORTANT DIFFERENCE:

Manual Agent = Developer controls the orchestration.

Create Agent = LangChain manages the orchestration.

Both approaches use the same fundamental LLM tool-calling
concept. create_agent() does not make the LLM smarter;
it provides a higher-level framework to manage execution.

===============================================================
TECHNOLOGIES USED:
- Python
- LangChain
- LangGraph
- Groq LLM
- OpenWeatherMap API
- Tavily Search API
- Python-dotenv
- Rich

===============================================================
ENVIRONMENT VARIABLES:
- OPENWEATHER_API_KEY
- GROQ_MODEL
- GROQ_API_KEY
- TAVILY_API_KEY

===============================================================
AGENT WORKFLOW:

User Input
    |
    v
Create Agent (LangChain)
    |
    v
Groq LLM
    |
    v
Tool Selection
    |
    +----------------------+
    |                      |
    v                      v
Weather Tool             News Tool
    |                      |
    v                      v
OpenWeatherMap           Tavily API
    |                      |
    +----------+-----------+
               |
               v
       Human Approval
               |
               v
        Tool Execution
               |
               v
       Tool Response
               |
               v
        LLM Invoked Again
               |
               v
          Final Answer

===============================================================
"""


# ==========================================
# IMPORTS AND ENVIRONMENT SETUP
# ==========================================

from dotenv import load_dotenv
import os
import requests

# Load API keys and environment variables from .env
load_dotenv()

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from rich import print
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call


# =========================
# 🌦️ Weather Tool
# =========================

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""

    # Get the OpenWeatherMap API key from .env
    api_key = os.getenv("OPENWEATHER_API_KEY")

    # API URL to fetch the current weather of an Indian city
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"

    # Send a request to the weather API
    response = requests.get(url)

    # Convert the API response into a Python dictionary
    data = response.json()

    # Check whether the API returned an error
    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    # Extract temperature and weather description
    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    # Return the weather information
    return f"Weather in {city}: {desc}, {temp}°C"


# =========================
# 📰 News Tool (Tavily)
# =========================

# Initialize Tavily using the API key from .env
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""

    # Search the web for the latest news about the given city
    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    # Extract the search results
    results = response.get("results", [])

    # Check whether any news results were found
    if not results:
        return f"No news found for {city}"

    news_list = []

    # Extract the title, URL and content from each result
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")

        # Add each news result to the list
        news_list.append(
            f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
        )

    # Return all the news results
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


# =========================
# 🧠 LLM Setup
# =========================

# Get the Groq model name from .env
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Initialize the Groq language model
llm = ChatGroq(model=GROQ_MODEL)


# =========================
# 👤 Human Approval Middleware
# =========================

@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before every tool call."""

    # Get the name of the tool selected by the agent
    tool_name = request.tool_call["name"]

    # Ask the user whether the tool call is allowed
    confirm = input(
        f"Agent wants to call '{tool_name}'. Approve? (yes/no): "
    )

    # If the user rejects, return a message without executing the tool
    if confirm.lower() != "yes":
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=request.tool_call["id"]
        )

    # If approved, execute the selected tool
    return handler(request)


# =========================
# 🤖 Create Agent
# =========================

agent = create_agent(
    llm,

    # Give the agent access to both tools
    tools=[get_weather, get_news],

    # Instructions that define the agent's role
    system_prompt="You are a helpful city assistant.",

    # Middleware asks for approval before tool execution
    middleware=[human_approval]
)


# =========================
# 💬 Chat Loop
# =========================

print("City Agent | type exit to quit")

while True:

    # Get input from the user
    user_input = input("You : ")

    # Stop the program when the user types exit
    if user_input.lower() == "exit":
        break

    # Send the user's message to the agent
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": user_input}
        ]
    })

    # Display the final response from the agent
    print("bot : ", result["messages"][-1].content)
