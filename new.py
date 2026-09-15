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
from langgraph.types import Command , interrupt
load_dotenv()
key = os.getenv("HUGGING_FACE_TOKEN")

@tool
def multiply(a: int, b: int) -> str:
    """This tool is to Multiply a with b"""
    return f"calculation is {a * b}"
@tool
def human_assitance(Query : str) -> str :
    """get assitance from human"""
    human_respond = interrupt({"query" : Query})
    return human_respond["data"]
# @tool
# def human_assitance(query : str) -> str:
#     """ Give  assitance to human"""
# def get_weather(location: str) -> str:
#     """This tool is used to find weather of certain if its good or bad"""
#     if location in ("Mumbai", "Delhi"):
#         return "weather is good"
#     else:
#         return "weather is bad"
tool_registry = [multiply , human_assitance]

class State(TypedDict):
    messages: Annotated[list, add_messages]

stategraph = StateGraph(State)


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=key, )
chat = ChatHuggingFace(llm=llm)
chat_with_tools = chat.bind_tools(tools = tool_registry)

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

config = {"configurable": {"thread_id": "thread_1"}}

# --- FIRST: Start the graph with a prompt that triggers the human assistance tool ---
# change the text to something that forces the LLM to call your tool
initial_input = {"messages": [HumanMessage(content="delete my previous information")]}

for event in Graph.stream(initial_input, stream_mode="values", config=config):
    pass


human_response = "do you really want to delete your previous information"
human_msg = Command(resume={"data": human_response})


for event in Graph.stream(human_msg, stream_mode="values", config=config):
      event["messages"][-1].pretty_print()






