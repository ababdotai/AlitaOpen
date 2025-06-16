#!/usr/bin/env python3
"""
Test script for the updated WebAgent using exa-py.

This script demonstrates how to use the new WebAgent implementation
with Exa API for semantic search and content retrieval.
"""

import os
import sys
import logging
from typing import Dict, Any

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web_agent import WebAgent

def setup_logging() -> None:
    """
    Configure logging for the test script.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_test_config() -> Dict[str, Any]:
    """
    Create a test configuration for the WebAgent.
    
    Returns:
        Dict[str, Any]: Configuration dictionary with test settings.
    """
    # You need to set your actual Exa API key here or via environment variable
    exa_api_key = os.getenv('EXA_API_KEY', 'your-exa-api-key-here')
    
    if exa_api_key == 'your-exa-api-key-here':
        print("Warning: Please set your EXA_API_KEY environment variable or update the test script.")
        print("You can get an API key from: https://exa.ai/")
        return None
    
    config = {
        'exa_api_key': exa_api_key,
        'max_results': 5,
        'use_autoprompt': True,
        'include_text': True
    }
    
    return config

def test_search_functionality(web_agent: WebAgent) -> None:
    """
    Test the search functionality of the WebAgent.
    
    Args:
        web_agent (WebAgent): The WebAgent instance to test.
    """
    print("\n=== Testing Search Functionality ===")
    
    test_queries = [
        "latest developments in artificial intelligence",
        "Python programming best practices",
        "machine learning tutorials"
    ]
    
    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        results = web_agent.search(query)
        
        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results[:3], 1):  # Show first 3 results
                print(f"  {i}. Title: {result['title'][:80]}...")
                print(f"     URL: {result['url']}")
                print(f"     Snippet: {result['snippet'][:100]}...")
                print()
        else:
            print("No results found.")

def test_navigate_functionality(web_agent: WebAgent) -> None:
    """
    Test the navigate functionality of the WebAgent.
    
    Args:
        web_agent (WebAgent): The WebAgent instance to test.
    """
    print("\n=== Testing Navigate Functionality ===")
    
    # First, get some URLs from a search
    search_results = web_agent.search("Python programming tutorial")
    
    if search_results:
        test_url = search_results[0]['url']
        print(f"\nNavigating to: {test_url}")
        
        content = web_agent.navigate(test_url)
        
        if content:
            print(f"Successfully retrieved content ({len(content)} characters)")
            print(f"Content preview: {content[:200]}...")
        else:
            print("Failed to retrieve content.")
    else:
        print("No URLs available for navigation test.")

def main() -> None:
    """
    Main function to run the WebAgent tests.
    """
    setup_logging()
    
    print("WebAgent Test Script (using exa-py)")
    print("====================================")
    
    # Create test configuration
    config = create_test_config()
    if not config:
        return
    
    try:
        # Initialize WebAgent
        print("\nInitializing WebAgent...")
        web_agent = WebAgent(config)
        print("WebAgent initialized successfully!")
        
        # Test search functionality
        # test_search_functionality(web_agent)
        
        # Test navigate functionality
        test_navigate_functionality(web_agent)
        
        print("\n=== All tests completed! ===")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        logging.error(f"Test failed with exception: {e}", exc_info=True)

if __name__ == "__main__":
    main()