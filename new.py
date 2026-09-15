import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
# Reverting back to native Hugging Face classes
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
key = os.getenv("HUGGING_FACE_TOKEN")

@tool
def multiply(a: int, b: int) -> str:
    """This tool is to Multiply a with b"""
    return f"calculation is {a * b}"

@tool
def get_weather(location: str) -> str:
    """This tool is used to find weather of certain if its good or bad"""
    if location in ("Mumbai", "Delhi"):
        return "weather is good"
    else:
        return "weather is bad"

tool_registry = [multiply, get_weather]

class State(TypedDict):
    messages: Annotated[list, add_messages]

stategraph = StateGraph(State)

# Native Hugging Face client initialization
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=key,
)
chat = ChatHuggingFace(llm=llm)
chat_with_tools = chat.bind_tools(tools=tool_registry)

def chatbot(state: State):
    RESPONSE = chat_with_tools.invoke(state["messages"])
    return {"messages": [RESPONSE]}

stategraph.add_node("chatbot", chatbot)
stategraph.add_node("tools", ToolNode(tools=tool_registry))
stategraph.add_edge(START, "chatbot")
stategraph.add_conditional_edges("chatbot", tools_condition)
stategraph.add_edge("tools", "chatbot")

checkpointer = MemorySaver()
Graph = stategraph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "thread_1"}}

# Properly wrapped message array containing the LangChain object
messages = {"messages": [HumanMessage(content="HI i am Shreyas. What is 12 multiplied by 56?")]}

# Streaming network execution loop
for event in Graph.stream(messages, stream_mode="values", config=config):
    print(event)
