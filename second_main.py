import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_core.tools import tool
# Correct import for NVIDIA AI Endpoints
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode , tools_condition
load_dotenv()
Api_Key = os.getenv("NVIDIA_API_KEY")
@tool
def multiply(a:int, b: int) -> str:
    """ This tool is to Multiply a with b """
    return f"calculation is { a * b}"
def get_weather( location : str) -> str:
    """This tool is used  to find weather of ceratain if its good or bad"""
    if location in ("Mumbai", "Delhi"):
        return "weather is good"
    else:
        return "weather is bad"
# 1. Define Stat
# e with 'messages' (plural)
class STATE(TypedDict):
    messages: Annotated[list, add_messages]

builder = StateGraph(STATE)
tools = [multiply , get_weather]
toolnode = ToolNode(tools = tools )

# Initialize LLM
LLM = ChatNVIDIA(model ="nvidia/nemotron-4-340b-instruct", api_key=Api_Key , )
LLM_with_tools = LLM.bind_tools(tools)
# 2. Fix: Return the correct key 'messages'
def chatbot(state: STATE):
    response_message = LLM_with_tools.invoke(state["messages"])
    return {"messages": [response_message]}


builder.add_node("chatbot", chatbot)
builder.add_node("tools" , toolnode)
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot" , tools_condition)
builder.add_edge("tools" , "chatbot")
Graph = builder.compile()

# 3. Fix: Invoke using the correct dictionary structure and key
response = Graph.invoke({"messages": ["Multiply 13 with 45"]})
print(response)
