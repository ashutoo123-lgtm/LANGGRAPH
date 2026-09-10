from email import message

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph , START , END
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated
from dotenv import load_dotenv
import os
load_dotenv()
Api_key = os.getenv("NVIDIA_API_KEY")

class State(TypedDict):
    messages : Annotated[list , add_messages]

graph_builder = StateGraph(State)

LLM = ChatNVIDIA(model = "nvidia/nemotron-3.5-lightning-30b-a3b" , api_key = Api_key)
def graph_chatbot(state: State):
    return {"messages" : LLM.invoke(state["messages"]) }
graph_builder.add_node("chatbot" , graph_chatbot)
graph_builder.add_edge(START , "chatbot")
graph_builder.add_edge("chatbot" , END)
graph =  graph_builder.compile()
print(graph.invoke({"messages" : "HI friend will you help me"}))
