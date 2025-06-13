"""mcp_registry.py

This module implements the MCPRegistry class for storing and retrieving generated
Model Context Protocols (MCPs) in a persistent, file-based registry.

The MCPRegistry provides the following interface:
    - __init__(config: dict) -> None
    - register_mcp(name: str, code: str) -> None
    - get_mcp(name: str) -> str

It uses a JSON file (default: "mcp_registry.json") for storage. The registry file path
can be configured via the configuration dictionary under the "mcp_registry" key.
"""

import os
import json
import logging
from typing import Dict, Any

from utils import handle_error

class MCPRegistry:
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the MCPRegistry with configuration settings.

        Args:
            config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
                Optional Key:
                    - mcp_registry.registry_path: The file path for storing the MCP registry.
                      If not provided, defaults to "mcp_registry.json" in the current working directory.
        """
        # Retrieve configuration for the MCP registry; use default if not provided.
        mcp_registry_config: Dict[str, Any] = config.get("mcp_registry", {})
        self.registry_file: str = mcp_registry_config.get("registry_path", "mcp_registry.json")
        
        logging.info("Initializing MCPRegistry with registry file: %s", self.registry_file)
        
        # Attempt to load the existing registry from the file.
        self._registry: Dict[str, str] = {}
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r", encoding="utf-8") as registry_fp:
                    loaded_registry = json.load(registry_fp)
                    if isinstance(loaded_registry, dict):
                        self._registry = loaded_registry
                        logging.info("Loaded MCPRegistry with %d entries from file: %s", 
                                     len(self._registry), self.registry_file)
                    else:
                        logging.warning("Registry file %s does not contain a valid dictionary. "
                                        "Starting with an empty registry.", self.registry_file)
                        self._registry = {}
            except Exception as e:
                logging.error("Error loading MCPRegistry from file %s: %s", self.registry_file, str(e))
                self._registry = {}
        else:
            logging.info("Registry file %s not found. Starting with an empty MCPRegistry.", 
                         self.registry_file)

    def register_mcp(self, name: str, code: str) -> None:
        """
        Register or update an MCP in the registry.

        Args:
            name (str): A unique identifier for the MCP (e.g., "YouTube Video Subtitle Crawler").
            code (str): The Python script code implementing the MCP.

        Action:
            - Updates the in-memory registry with the given name and code.
            - Persists the updated registry to the file.
            - Logs the registration event.
        """
        try:
            self._registry[name] = code
            logging.info("Registering MCP with name: '%s'", name)
            self._persist_registry()
        except Exception as e:
            logging.error("Failed to register MCP '%s': %s", name, str(e))
            handle_error(e)

    def get_mcp(self, name: str) -> str:
        """
        Retrieve an MCP's code by its name.

        Args:
            name (str): The unique identifier for the MCP.

        Returns:
            str: The Python script code for the MCP if found; otherwise, an empty string.
        """
        mcp_code: str = self._registry.get(name, "")
        if mcp_code:
            logging.info("Retrieved MCP '%s' successfully.", name)
        else:
            logging.warning("MCP '%s' not found in the registry.", name)
        return mcp_code

    def _persist_registry(self) -> None:
        """
        Persist the in-memory registry to the registry file in JSON format.

        This method writes the current registry to disk to ensure persistence across runs.
        Errors encountered during file I/O are logged and handled using the handle_error function.
        """
        try:
            with open(self.registry_file, "w", encoding="utf-8") as registry_fp:
                json.dump(self._registry, registry_fp, indent=4)
            logging.info("MCPRegistry persisted successfully to file: %s", self.registry_file)
        except Exception as e:
            logging.error("Failed to persist MCPRegistry to file %s: %s", self.registry_file, str(e))
            handle_error(e)
