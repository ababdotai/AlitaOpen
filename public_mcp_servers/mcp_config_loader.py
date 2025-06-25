import json
import logging
import os
from typing import Dict, Any

from dotenv import load_dotenv

load_dotenv()


def load_mcp_servers_config() -> Dict[str, Any]:
    """
    Load MCP servers configuration from the mcpServers.json file.

    Returns:
        Dict[str, Any]: MCP servers configuration dictionary.
    """
    try:
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mcp_config_path = os.path.join(current_dir, "mcpServers.json")

        if os.path.exists(mcp_config_path):
            with open(mcp_config_path, 'r', encoding='utf-8') as f:
                server_config = json.load(f)

                # Set default transport if not specified
                for server_name, server_info in server_config["mcpServers"].items():
                    if "transport" not in server_info:
                        if "url" in server_info:
                            server_info["transport"] = "streamable_http" if "mcp" in server_info["url"] else "sse"
                        else:
                            server_info["transport"] = "stdio"

                return server_config
        else:
            logging.warning("MCP servers config file not found at: %s", mcp_config_path)
            return {"mcpServers": {}}

    except Exception as e:
        logging.error("Failed to load MCP servers config: %s", str(e))
        return {"mcpServers": {}}


def process_environment_variables(server_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process environment variable replacements in MCP server configuration.
    
    Args:
        server_config: MCP server configuration dictionary
        
    Returns:
        Dict[str, Any]: Processed configuration with environment variables replaced
    """
    try:
        for server_name, server_info in server_config["mcpServers"].items():
            # Replace environment variables in args
            if "args" in server_info:
                for i, arg in enumerate(server_info["args"]):
                    if "$MCP_SERVERS_DIR" in arg:
                        server_info["args"][i] = arg.replace("$MCP_SERVERS_DIR", os.environ.get("MCP_SERVERS_DIR", ""))
                    if "$EXA_API_KEY" in arg:
                        server_info["args"][i] = arg.replace("$EXA_API_KEY", os.environ.get("EXA_API_KEY", ""))

            # Replace environment variables in headers
            if "headers" in server_info and "Authorization" in server_info["headers"]:
                if "$GITHUB_TOKEN" in server_info["headers"]["Authorization"]:
                    server_info["headers"]["Authorization"] = server_info["headers"]["Authorization"].replace(
                        "$GITHUB_TOKEN", os.environ.get("GITHUB_TOKEN", "")
                    )
            
            # Replace environment variables in env field
            if "env" in server_info:
                for env_key, env_value in server_info["env"].items():
                    if isinstance(env_value, str):
                        if "$EXA_API_KEY" in env_value:
                            server_info["env"][env_key] = env_value.replace("$EXA_API_KEY", os.environ.get("EXA_API_KEY", ""))
                        if "$GITHUB_TOKEN" in env_value:
                            server_info["env"][env_key] = env_value.replace("$GITHUB_TOKEN", os.environ.get("GITHUB_TOKEN", ""))
                        if "$MCP_SERVERS_DIR" in env_value:
                            server_info["env"][env_key] = env_value.replace("$MCP_SERVERS_DIR", os.environ.get("MCP_SERVERS_DIR", ""))

        return server_config

    except Exception as e:
        logging.error("Failed to process environment variables: %s", str(e))
        return server_config
