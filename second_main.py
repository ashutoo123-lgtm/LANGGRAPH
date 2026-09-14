# ------------------ packages -----------------
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
import time
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from requests.exceptions import ReadTimeout


# ---------------- API KEY -----------------
load_dotenv()
Api_key = os.getenv("NVIDIA_API_KEY")

if not Api_key:
    raise ValueError("NVIDIA_API_KEY not found in .env file")


# ---------------- STATE OF THE AGENT -----------------
class State(TypedDict):
    messages: Annotated[list, add_messages]


Graph_builder = StateGraph(State)


# ---------------- TOOL -----------------
@tool
def calculation(a: int, b: int, expression: str):
    """
    Calculate addition, subtraction, multiplication,
    or division of two numbers.
    """
    expression = expression.lower()

    if expression == "add":
        return f"Sum is {a + b}"
    elif expression == "subtract":
        return f"Subtraction is {a - b}"
    elif expression == "multiply":
        return f"Multiplication is {a * b}"
    elif expression == "divide":
        if b == 0:
            return "Cannot divide by zero."
        return f"Division is {a / b}"
    else:
        return f"Unknown expression: {expression}"


# ---------------- REGISTER TOOLS -----------------
tools_registered = [calculation]


# ---------------- LLM -----------------
chatbot = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    api_key=Api_key,
    timeout=120  # increased timeout
)


# ---------------- LLM WITH TOOLS -----------------
LLM_tools = chatbot.bind_tools(tools_registered)


# ---------------- CHATBOT NODE -----------------
def chatbot_node(state: State):
    # keep only last 3 messages to reduce payload size
    recent_messages = state["messages"][-3:]

    for attempt in range(3):  # retry up to 3 times
        try:
            response = LLM_tools.invoke(recent_messages)
            return {"messages": [response]}
        except ReadTimeout:
            print(f"Timeout on attempt {attempt+1}, retrying...")
            time.sleep(5)

    return {"messages": ["Service unavailable after retries."]}


# ---------------- ADD NODES -----------------
Graph_builder.add_node("chatbot", chatbot_node)
Graph_builder.add_node("tools", ToolNode(tools_registered))


# ---------------- EDGES -----------------
Graph_builder.add_edge(START, "chatbot")
Graph_builder.add_conditional_edges("chatbot", tools_condition)
Graph_builder.add_edge("tools", "chatbot")


# ---------------- COMPILE GRAPH -----------------
graph = Graph_builder.compile()


# ---------------- RUN AGENT -----------------
Response = graph.invoke(
    {
        "messages": [
            HumanMessage(content="Add 12 with 54")
        ]
    }
)


# ---------------- FINAL RESPONSE -----------------
print(Response)
