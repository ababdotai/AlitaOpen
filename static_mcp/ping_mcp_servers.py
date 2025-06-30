import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient

from mcp_config_loader import load_mcp_servers_config


async def ping_mcp_servers():
    # Load MCP servers configuration with environment variables processed
    server_config = load_mcp_servers_config()

    client = MultiServerMCPClient(server_config["mcpServers"])
    tools = await client.get_tools()
    for i, (server_name, server_info) in enumerate(server_config["mcpServers"].items()):
        print(f"Found MCP Server #{i}: {server_name}: {server_info}")

    if tools:
        print(f"\n=== Found {len(tools)} MCP Tools ===")
        for i, tool in enumerate(tools, 1):
            print(f"[Tool #{i}] {tool.name}")
            print(f"[Description] {tool.description}")
            print(f"[Arguments Schema]\n {tool.args_schema}")
            print(f"[Return Direct] {tool.return_direct}")
            print(f"[Response Format] {tool.response_format}")
            print("-" * 80)
    else:
        print("\n❌ No MCP tools found")


if __name__ == "__main__":
    asyncio.run(ping_mcp_servers())
