"""web_agent.py

This module implements the WebAgent class for external searches and webpage navigation.
It uses the requests library to perform HTTP requests and BeautifulSoup from bs4 to parse HTML responses.
The WebAgent provides two main public methods:
    - search(query: str) -> list
    - navigate(url: str) -> str

These methods are used by the ManagerAgent to gather external context and resource URLs for tool generation.
"""

import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Any
import urllib.parse

class WebAgent:
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the WebAgent with configuration settings.

        Args:
            config (Dict[str, Any]): Configuration dictionary, typically loaded from config.yaml.
                                      If no specific settings for the web agent are provided, 
                                      defaults will be used.
        """
        self.config: Dict[str, Any] = config
        # Set a default user agent string to mimic a typical web browser.
        self.user_agent: str = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
        # Headers to use for HTTP requests.
        self.headers: Dict[str, str] = {
            "User-Agent": self.user_agent
        }
        # Default search URL prefix. Using Google's search URL as default.
        self.search_url_prefix: str = "https://www.google.com/search?q="
        # Default timeout value for web requests (in seconds).
        self.timeout: int = self.config.get("web_agent_timeout", 5)

        logging.info("WebAgent initialized with timeout=%d seconds and search_url_prefix=%s",
                     self.timeout, self.search_url_prefix)

    def search(self, query: str) -> List[Dict[str, str]]:
        """
        Perform an external search using the provided natural language query.

        Constructs a search URL, sends an HTTP GET request, parses the returned HTML,
        and extracts a list of resource items. Each resource item is a dictionary containing:
            - 'url': The hyperlink URL of the result.
            - 'title': The text content of the link.
            - 'snippet': An optional snippet (left empty if not available).

        Args:
            query (str): The natural language query to search for.

        Returns:
            List[Dict[str, str]]: A list of resource items with keys "url", "title", and "snippet".
                                  Returns an empty list if the search fails or no results are found.
        """
        try:
            # Encode the query for inclusion in the URL.
            encoded_query: str = urllib.parse.quote(query)
            search_url: str = self.search_url_prefix + encoded_query
            logging.info("Executing search with URL: %s", search_url)

            response = requests.get(search_url, headers=self.headers, timeout=self.timeout)
            if response.status_code != 200:
                logging.error("Search request failed with status code %d for query: %s",
                              response.status_code, query)
                return []

            soup = BeautifulSoup(response.text, "html.parser")
            results: List[Dict[str, str]] = []
            seen_urls: set = set()

            # Extract anchor tags. Many anchors in the page are not search results,
            # so we filter out those that do not start with "http" and contain "google".
            anchor_tags = soup.find_all("a", href=True)
            for anchor in anchor_tags:
                href: str = anchor.get("href").strip()
                if href.startswith("http") and "google" not in href and href not in seen_urls:
                    # Use the anchor's text as title; it might be empty.
                    title: str = anchor.get_text().strip()
                    result_item: Dict[str, str] = {
                        "url": href,
                        "title": title,
                        "snippet": ""
                    }
                    results.append(result_item)
                    seen_urls.add(href)

            logging.info("Search query '%s' returned %d results", query, len(results))
            return results

        except Exception as e:
            logging.error("Exception occurred during search for query '%s': %s", query, str(e))
            return []

    def navigate(self, url: str) -> str:
        """
        Retrieve and process the content of the web page at the given URL.

        This method sends an HTTP GET request to the URL, parses the returned HTML,
        removes script and style tags for cleaner output, and returns the resulting text.

        Args:
            url (str): The URL of the web page to navigate.

        Returns:
            str: The cleaned textual content of the page. Returns an empty string if navigation fails.
        """
        try:
            logging.info("Navigating to URL: %s", url)
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            if response.status_code != 200:
                logging.error("Navigation request failed with status code %d for URL: %s",
                              response.status_code, url)
                return ""

            soup = BeautifulSoup(response.text, "html.parser")
            # Remove script and style elements from the page.
            for script_tag in soup(["script", "style"]):
                script_tag.decompose()

            # Get text content from the page and clean up whitespace.
            page_text: str = soup.get_text(separator=" ", strip=True)
            logging.info("Navigation to URL '%s' succeeded; content length: %d characters", url, len(page_text))
            return page_text

        except Exception as e:
            logging.error("Exception occurred during navigation for URL '%s': %s", url, str(e))
            return ""
