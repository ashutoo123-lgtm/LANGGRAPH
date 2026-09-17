import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
# LangChain & NVIDIA Imports
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# LangGraph Imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
load_dotenv()
API_KEY = os.getenv("NVIDIA_API_KEY")
# =====================================================================
# 1. DEFINE YOUR TOOLS
# =====================================================================
@tool
def calculate_power_efficiency(gpu_power_watts: int, core_clock_mhz: int) -> float:
    """
    Calculates a dummy performance-per-watt metric for an AI accelerator workload.
    Use this tool when users ask about GPU or chip efficiency calculations.
    """
    if gpu_power_watts <= 0:
        return 0.0
    # Dummy hardware performance score equation
    score = (core_clock_mhz * 1.5) / gpu_power_watts
    return round(score, 2)

# Pack tools into a list
tools = [calculate_power_efficiency]


# =====================================================================
# 2. INITIALIZE CHATNVIDIA AND BIND TOOLS
# =====================================================================
# We use an agent-optimized model like Llama-3.1 or Nemotron
llm = ChatNVIDIA(model="nvidia/nemotron-3-ultra-550b-a55b", temperature=0.2, api_key = API_KEY)

# Inform ChatNVIDIA about the tools it has permission to call
llm_with_tools = llm.bind_tools(tools)


# =====================================================================
# 3. DEFINE THE STATE & NODES OF THE GRAPH
# =====================================================================
# State tracks the sequence of messages passed around the graph
class State(TypedDict):
    messages: Annotated[list, add_messages]

# The agent node: takes the message list, calls ChatNVIDIA, and appends the response
def call_model(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# =====================================================================
# 4. CONSTRUCT THE LANGGRAPH WORKFLOW
# =====================================================================
workflow = StateGraph(State)

# Add our core AI agent node
workflow.add_node("agent", call_model)

# Add the prebuilt ToolNode to automatically execute tools when requested by the LLM
workflow.add_node("tools", ToolNode(tools))

# Establish execution path starting at the agent
workflow.add_edge(START, "agent")

# Add a conditional router edge from 'agent'.
# 'tools_condition' checks if ChatNVIDIA sent back a 'tool_calls' instruction:
# - If YES: routes execution to the "tools" node.
# - If NO: routes execution to END to stop the graph and respond to the user.
workflow.add_conditional_edges(
    "agent",
    tools_condition,
)

# After a tool executes, route its output (ToolMessage) back to the agent for interpretation
workflow.add_edge("tools", "agent")

# Compile into an executable application
app = workflow.compile()


# =====================================================================
# 5. EXECUTE THE AGENT
# =====================================================================
if __name__ == "__main__":
    # Ensure your environment variable is verified
    if "NVIDIA_API_KEY" not in os.environ:
        print("Please set your NVIDIA_API_KEY environment variable.")
        exit(1)

    print("--- Running LangGraph Agent with ChatNVIDIA ---\n")

    query = "I have a GPU running at 2100 MHz drawing 250 Watts. What is its power efficiency score?"

    # Run the graph and stream the execution steps
    events = app.stream(
        {"messages": [("user", query)]},
        stream_mode="values"
    )

    for event in events:
        # Beautifully print out messages as they update across nodes
        if "messages" in event:
            last_message = event["messages"][-1]
            last_message.pretty_print()
