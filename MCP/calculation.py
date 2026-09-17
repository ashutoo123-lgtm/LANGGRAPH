from  mcp.server.fastmcp import FastMCP
mcp = FastMCP("calculate_tool_server")
@mcp.tool()
def calculate(a:int, b: int , operation : str):
    """Tool to calculate addition , multiplication , subtraction , divison on  a and b  based on the operation provided"""
    if operation == "addition":
        return f"sum is {a + b}"
    elif operation == "subtraction":
        return f"subtraction is { a - b}"
    elif operation == "multiplication":
        return f"multiplication is { a * b}"
    else :
        return f"division is { a / b}"

if __name__ == "__main__":
    mcp.run( transport = "stdio")
    


