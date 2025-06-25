import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient

from mcp_config_loader import load_mcp_servers_config, process_environment_variables


async def ping_mcp_servers():
    # Load MCP servers configuration using shared module
    server_config = load_mcp_servers_config()

    # Process environment variable replacements
    server_config = process_environment_variables(server_config)

    client = MultiServerMCPClient(server_config["mcpServers"])
    tools = await client.get_tools()
    for i, (server_name, server_info) in enumerate(server_config["mcpServers"].items()):
        print(f"Found MCP Server #{i}: {server_name}: {server_info}")

    if tools:
        print(f"Found {len(tools)} MCP tools:")
        for tool in tools:
            print("-" * 100)
            print(tool.name)
            print(tool.description)
            print(tool.args_schema)
            print(tool.return_direct)
            print(tool.response_format)
    else:
        print("No MCP tools found")


if __name__ == "__main__":
    asyncio.run(ping_mcp_servers())
