from mcp.server.fastmcp import FastMCP
mcp= FastMCP("finding the weather")
@mcp.tool()
def get_weather(location : str):
    """Tell how is the weather based on location given"""
    if location in ("New York" , "Bali", "Mumbai" , "London"):
        return f"The weather of {location} is really very great!!"
    else :
        return f"The weather of {location } is not very good"
if __name__ == "__main__":
    mcp.run(transport= "streamable-http")
