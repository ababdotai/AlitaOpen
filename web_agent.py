"""web_agent.py

This module implements the WebAgent class for external searches and webpage navigation.
It uses the exa-py library to perform semantic searches and content retrieval.
The WebAgent provides three main public methods:
    - search(query: str) -> Dict[str, List[Dict[str, str]]] - Enhanced search using multiple sources
    - search_web_pages(query: str) -> List[Dict[str, str]] - Traditional web search only
    - search_github_repositories(query: str) -> List[Dict[str, str]] - Search GitHub repositories
    - search_pypi_packages(query: str) -> List[Dict[str, str]] - Search PyPI packages
    - navigate(url: str) -> str

These methods are used by the ManagerAgent to gather external context and resource URLs for tool generation.
"""

from exa_py import Exa
import logging
import json
import os
import asyncio
from typing import List, Dict, Any, Optional
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

class WebAgent:
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the WebAgent with configuration settings.

        Args:
            config (Dict[str, Any]): Configuration dictionary, typically loaded from config.yaml.
                                      Must contain 'exa_api_key' for authentication.
                                      Optional settings include 'max_results' and 'use_autoprompt'.
        """
        exa_config: Dict[str, Any] = config.get("exa", {})

        # Get API key from config
        api_key: Optional[str] = exa_config.get("exa_api_key")
        if not api_key:
            raise ValueError("exa_api_key must be provided in config")
        
        # Initialize Exa client
        self.exa: Exa = Exa(api_key=api_key)
        
        # Configuration options
        self.max_results: int = exa_config.get("max_results", 10)
        self.use_autoprompt: bool = exa_config.get("use_autoprompt", True)
        self.include_text: bool = exa_config.get("include_text", True)
        
        # Load MCP servers configuration
        self.mcp_servers_config: Dict[str, Any] = self._load_mcp_servers_config()
        
        logging.info("WebAgent initialized with Exa API, max_results=%d, use_autoprompt=%s",
                     self.max_results, self.use_autoprompt)
        logging.info("Loaded MCP servers: %s", list(self.mcp_servers_config.keys()))

    def _load_mcp_servers_config(self) -> Dict[str, Any]:
        """
        Load MCP servers configuration from the public_mcp_servers/mcpServers.json file.
        
        Returns:
            Dict[str, Any]: MCP servers configuration dictionary.
        """
        try:
            # Get the directory of the current script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            mcp_config_path = os.path.join(current_dir, "public_mcp_servers", "mcpServers.json")
            
            if os.path.exists(mcp_config_path):
                with open(mcp_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config
            else:
                logging.warning("MCP servers config file not found at: %s", mcp_config_path)
                return {}
        except Exception as e:
            logging.error("Failed to load MCP servers config: %s", str(e))
            return {}

    def search_web_pages(self, query: str) -> List[Dict[str, str]]:
        """
        Perform an external search using the provided natural language query via Exa API.

        Uses Exa's semantic search capabilities to find relevant web pages.
        Each result item is a dictionary containing:
            - 'url': The hyperlink URL of the result.
            - 'title': The title of the web page.
            - 'snippet': A text snippet or summary from the page.

        Args:
            query (str): The natural language query to search for.

        Returns:
            List[Dict[str, str]]: A list of resource items with keys "url", "title", and "snippet".
                                  Returns an empty list if the search fails or no results are found.
        """
        try:
            logging.info("Executing Exa search for query: %s", query)
            
            # Perform search using Exa API
            search_response = self.exa.search(
                query=query,
                num_results=self.max_results,
                use_autoprompt=self.use_autoprompt
            )
            
            results: List[Dict[str, str]] = []
            
            # Process search results
            for result in search_response.results:
                result_item: Dict[str, str] = {
                    "url": result.url,
                    "title": getattr(result, 'title', '') or '',
                    "snippet": getattr(result, 'text', '') or ''
                }
                results.append(result_item)
            
            logging.info("Exa search query '%s' returned %d results", query, len(results))
            return results
            
        except Exception as e:
            logging.error("Exception occurred during Exa search for query '%s': %s", query, str(e))
            return []

    def navigate(self, url: str) -> str:
        """
        Retrieve and process the content of the web page at the given URL using Exa API.

        This method uses Exa's get_contents API to fetch clean, processed text content
        from the specified URL without needing to handle HTML parsing manually.

        Args:
            url (str): The URL of the web page to navigate.

        Returns:
            str: The cleaned textual content of the page. Returns an empty string if navigation fails.
        """
        try:
            logging.info("Navigating to URL using Exa: %s", url)
            
            # Use Exa's get_contents API to retrieve page content
            contents_response = self.exa.get_contents(
                urls=[url],
                text=True
            )
            
            # Extract text content from the response
            if contents_response.context:
                return contents_response.context
            elif contents_response.results and len(contents_response.results) > 0:
                content_item = contents_response.results[0]
                page_text: str = getattr(content_item, 'text', '') or ''
                
                # If text is not available, try extract field
                if not page_text:
                    page_text = getattr(content_item, 'extract', '') or ''
                
                logging.info("Navigation to URL '%s' succeeded; content length: %d characters", 
                           url, len(page_text))
                return page_text
            else:
                logging.warning("No content retrieved for URL: %s", url)
                return ""
                
        except Exception as e:
            logging.error("Exception occurred during Exa navigation for URL '%s': %s", url, str(e))
            return ""

    async def search_github_repositories(self, query: str) -> List[Dict[str, str]]:
        """
        Search GitHub repositories using MCP GitHub server.
        
        Args:
            query (str): Search query for GitHub repositories.
            
        Returns:
            List[Dict[str, str]]: List of repository information with keys like "name", "url", "description".
        """
        try:
            github_config = self.mcp_servers_config.get("mcpServers", {}).get("github-remote")
            if not github_config:
                logging.warning("GitHub MCP server not configured")
                return []
            
            # Use streamable HTTP client for GitHub remote MCP server
            async with streamablehttp_client(github_config["url"]) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # List available tools
                    tools = await session.list_tools()
                    logging.info("Available GitHub MCP tools: %s", [tool.name for tool in tools.tools])
                    
                    # Search for repositories using the search tool
                    search_tools = [tool for tool in tools.tools if 'search' in tool.name.lower()]
                    if search_tools:
                        result = await session.call_tool(
                            search_tools[0].name, 
                            {"query": query, "type": "repositories"}
                        )
                        
                        # Parse the result and format it
                        repositories = []
                        if result.content:
                            for content in result.content:
                                if hasattr(content, 'text'):
                                    # Parse the text content to extract repository information
                                    repo_info = self._parse_github_search_result(content.text)
                                    repositories.extend(repo_info)
                        
                        logging.info("GitHub search for '%s' returned %d repositories", query, len(repositories))
                        return repositories
                    else:
                        logging.warning("No search tool found in GitHub MCP server")
                        return []
                        
        except Exception as e:
            logging.error("Exception occurred during GitHub MCP search for query '%s': %s", query, str(e))
            return []

    async def search_pypi_packages(self, query: str) -> List[Dict[str, str]]:
        """
        Search PyPI packages using MCP PyPI server.
        
        Args:
            query (str): Search query for PyPI packages.
            
        Returns:
            List[Dict[str, str]]: List of package information with keys like "name", "version", "description".
        """
        try:
            pypi_config = self.mcp_servers_config.get("pypi")
            if not pypi_config:
                logging.warning("PyPI MCP server not configured")
                return []
            
            # Use stdio client for local PyPI MCP server
            server_params = StdioServerParameters(
                command=pypi_config["command"],
                args=pypi_config["args"]
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # List available tools
                    tools = await session.list_tools()
                    logging.info("Available PyPI MCP tools: %s", [tool.name for tool in tools.tools])
                    
                    # Search for packages using the search_packages tool
                    search_tools = [tool for tool in tools.tools if tool.name == 'search_packages']
                    if search_tools:
                        result = await session.call_tool(
                            "search_packages", 
                            {"query": query}
                        )
                        
                        # Parse the result and format it
                        packages = []
                        if result.content:
                            for content in result.content:
                                if hasattr(content, 'text'):
                                    # Parse the text content to extract package information
                                    package_info = self._parse_pypi_search_result(content.text)
                                    packages.extend(package_info)
                        
                        logging.info("PyPI search for '%s' returned %d packages", query, len(packages))
                        return packages
                    else:
                        logging.warning("No search_packages tool found in PyPI MCP server")
                        return []
                        
        except Exception as e:
            logging.error("Exception occurred during PyPI MCP search for query '%s': %s", query, str(e))
            return []

    def search(self, query: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Perform aggregated search using both traditional web search and MCP servers.
        
        Args:
            query (str): Search query.
            
        Returns:
            Dict[str, List[Dict[str, str]]]: Dictionary containing results from different sources:
                - 'web': Traditional web search results
                - 'github': GitHub repository results
                - 'pypi': PyPI package results
        """
        results = {
            'web': [],
            'github': [],
            'pypi': []
        }
        
        # Traditional web search
        try:
            results['web'] = self.search_web_pages(query)
        except Exception as e:
            logging.error("Web search failed: %s", str(e))
        
        # MCP-based searches (run asynchronously)
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # GitHub search
            try:
                github_results = loop.run_until_complete(self.search_github_repositories(query))
                results['github'] = github_results
            except Exception as e:
                logging.error("GitHub MCP search failed: %s", str(e))
            
            # PyPI search
            try:
                pypi_results = loop.run_until_complete(self.search_pypi_packages(query))
                results['pypi'] = pypi_results
            except Exception as e:
                logging.error("PyPI MCP search failed: %s", str(e))
            
            loop.close()
            
        except Exception as e:
            logging.error("MCP searches failed: %s", str(e))
        
        total_results = len(results['web']) + len(results['github']) + len(results['pypi'])
        logging.info("Enhanced search for '%s' returned %d total results (web: %d, github: %d, pypi: %d)", 
                     query, total_results, len(results['web']), len(results['github']), len(results['pypi']))
        
        return results

    def _parse_github_search_result(self, text: str) -> List[Dict[str, str]]:
        """
        Parse GitHub search result text and extract repository information.
        
        Args:
            text (str): Raw text result from GitHub MCP server.
            
        Returns:
            List[Dict[str, str]]: Parsed repository information.
        """
        repositories = []
        try:
            # Try to parse as JSON first
            if text.strip().startswith('[') or text.strip().startswith('{'):
                import json
                data = json.loads(text)
                if isinstance(data, list):
                    for repo in data:
                        if isinstance(repo, dict):
                            repositories.append({
                                "name": repo.get("name", ""),
                                "url": repo.get("html_url", repo.get("url", "")),
                                "description": repo.get("description", "")
                            })
                elif isinstance(data, dict) and "items" in data:
                    for repo in data["items"]:
                        repositories.append({
                            "name": repo.get("name", ""),
                            "url": repo.get("html_url", repo.get("url", "")),
                            "description": repo.get("description", "")
                        })
            else:
                # Fallback: treat as plain text description
                repositories.append({
                    "name": "GitHub Search Result",
                    "url": "",
                    "description": text[:500]  # Truncate long descriptions
                })
        except Exception as e:
            logging.error("Failed to parse GitHub search result: %s", str(e))
            # Fallback: treat as plain text
            repositories.append({
                "name": "GitHub Search Result",
                "url": "",
                "description": text[:500]
            })
        
        return repositories

    def _parse_pypi_search_result(self, text: str) -> List[Dict[str, str]]:
        """
        Parse PyPI search result text and extract package information.
        
        Args:
            text (str): Raw text result from PyPI MCP server.
            
        Returns:
            List[Dict[str, str]]: Parsed package information.
        """
        packages = []
        try:
            # Try to parse as JSON first
            if text.strip().startswith('[') or text.strip().startswith('{'):
                import json
                data = json.loads(text)
                if isinstance(data, list):
                    for pkg in data:
                        if isinstance(pkg, dict):
                            packages.append({
                                "name": pkg.get("name", ""),
                                "version": pkg.get("version", ""),
                                "description": pkg.get("summary", pkg.get("description", ""))
                            })
                elif isinstance(data, dict):
                    packages.append({
                        "name": data.get("name", ""),
                        "version": data.get("version", ""),
                        "description": data.get("summary", data.get("description", ""))
                    })
            else:
                # Fallback: treat as plain text description
                packages.append({
                    "name": "PyPI Search Result",
                    "version": "",
                    "description": text[:500]  # Truncate long descriptions
                })
        except Exception as e:
            logging.error("Failed to parse PyPI search result: %s", str(e))
            # Fallback: treat as plain text
            packages.append({
                "name": "PyPI Search Result",
                "version": "",
                "description": text[:500]
            })
        
        return packages
