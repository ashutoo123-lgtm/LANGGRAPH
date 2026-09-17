import asyncio

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
import os
api_key = os.getenv("NVIDIA_API_KEY")
async def main():

    # Connect to TWO MCP servers
    client = MultiServerMCPClient(
        {
            "calculator": {
                "command": "python",
                "args": ["calculator.py"],
                "transport": "stdio",
            },

            "weather": {
                "command": "python",
                "args": ["weather.py"],
                "transport": "streamable_http",
            }
        }
    )

    # Discover tools from both servers
    tools = await client.get_tools()

    print("\nTOOLS DISCOVERED:")
    for tool in tools:
        print(tool.name)

    # LLM
    model = ChatNVIDIA(
        model="nvidia/nemotron-4-340b-instruct",
        api_key= api_key
    )

    # Give ALL discovered tools to the model
    model_with_tools = model.bind_tools(tools)

    # LLM node
    def call_model(state):

        response = model_with_tools.invoke(
            state["messages"]
        )

        return {
            "messages": [response]
        }

    # LangGraph
    builder = StateGraph(MessagesState)

    builder.add_node("llm", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "llm")

    builder.add_conditional_edges(
        "llm",
        tools_condition
    )

    builder.add_edge("tools", "llm")

    graph = builder.compile()

    # Ask something requiring a tool
    result = await graph.ainvoke(
        {
            "messages": [
                (
                    "user",
                    "What is the weather in Delhi?"
                )
            ]
        }
    )

    print("\nFINAL:")
    print(result["messages"][-1].content)


asyncio.run(main())
