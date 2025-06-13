"""web_agent.py

This module implements the WebAgent class for external searches and webpage navigation.
It uses the exa-py library to perform semantic searches and content retrieval.
The WebAgent provides two main public methods:
    - search(query: str) -> list
    - navigate(url: str) -> str

These methods are used by the ManagerAgent to gather external context and resource URLs for tool generation.
"""

from exa_py import Exa
import logging
from typing import List, Dict, Any, Optional

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
        
        logging.info("WebAgent initialized with Exa API, max_results=%d, use_autoprompt=%s",
                     self.max_results, self.use_autoprompt)

    def search(self, query: str) -> List[Dict[str, str]]:
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
