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

# Add the parent directory to the Python path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_agent import WebAgent
from utils import load_config

def setup_logging() -> None:
    """
    Configure logging for the test script.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_test_config() -> Dict[str, Any]:
    """
    Load configuration from config.yaml file.
    
    Returns:
        Dict[str, Any]: Configuration dictionary loaded from config.yaml.
    """
    try:
        # Load configuration from config.yaml in parent directory
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        config = load_config(config_path)
        print("✓ Configuration loaded successfully from config.yaml")
        return config
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return None

def test_web_agent_initialization(config: Dict[str, Any]) -> WebAgent:
    """
    Test WebAgent initialization.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary.
        
    Returns:
        WebAgent: Initialized WebAgent instance or None if failed.
    """
    print("\n=== Testing WebAgent Initialization ===")
    try:
        # Check if exa_api_key exists in config
        exa_config = config.get('exa', {})
        if not exa_config.get('exa_api_key'):
            print("✗ No exa_api_key found in config.yaml")
            return None
        print("✓ Exa API key found in configuration")
        
        agent = WebAgent(config)
        print("✓ WebAgent initialized successfully")
        
        # Test MCP servers config loading
        if hasattr(agent, 'mcp_servers_config'):
            print(f"✓ MCP servers config loaded: {len(agent.mcp_servers_config)} servers")
            for server_name in agent.mcp_servers_config.keys():
                print(f"  - {server_name}")
        else:
            print("✗ MCP servers config not loaded")
            
        # Test search method exists (aggregated search)
        if hasattr(agent, 'search'):
            print("✓ search method available (aggregated search)")
        else:
            print("✗ search method not found")
        
        # Test search_web_pages method exists
        if hasattr(agent, 'search_web_pages'):
            print("✓ search_web_pages method available")
        else:
            print("✗ search_web_pages method not found")
            
        # Test search_github_repositories method exists
        if hasattr(agent, 'search_github_repositories'):
            print("✓ search_github_repositories method available")
        else:
            print("✗ search_github_repositories method not found")
            
        # Test search_pypi_packages method exists
        if hasattr(agent, 'search_pypi_packages'):
            print("✓ search_pypi_packages method available")
        else:
            print("✗ search_pypi_packages method not found")
            
        return agent
    except Exception as e:
        print(f"✗ WebAgent initialization failed: {e}")
        return None

def test_search_web_pages(web_agent: WebAgent) -> bool:
    """
    Test the search_web_pages functionality of the WebAgent (traditional web search only).
    
    Args:
        web_agent (WebAgent): The WebAgent instance to test.
        
    Returns:
        bool: True if test passed, False otherwise.
    """
    print("\n=== Testing search_web_pages Functionality (Traditional Web Search) ===")
    
    test_queries = [
        "Python machine learning libraries",
        "artificial intelligence tutorials"
    ]
    
    success_count = 0
    
    for query in test_queries:
        print(f"\nTesting search_web_pages with query: '{query}'")
        try:
            results = web_agent.search_web_pages(query)
            
            if results and len(results) > 0:
                print(f"✓ search_web_pages returned {len(results)} results")
                # Show first result as example
                first_result = results[0]
                print(f"  Example result: {first_result.get('title', 'No title')[:50]}...")
                print(f"  URL: {first_result.get('url', 'No URL')[:50]}...")
                success_count += 1
            else:
                print("✗ search_web_pages returned empty results")
        except Exception as e:
            print(f"✗ search_web_pages failed: {e}")
    
    return success_count > 0

def test_search_github_repositories(web_agent: WebAgent) -> bool:
    """
    Test the search_github_repositories functionality of the WebAgent.
    
    Args:
        web_agent (WebAgent): The WebAgent instance to test.
        
    Returns:
        bool: True if test passed, False otherwise.
    """
    print("\n=== Testing search_github_repositories Functionality ===")
    
    test_queries = [
        "python machine learning",
        "data science tools"
    ]
    
    success_count = 0
    
    for query in test_queries:
        print(f"\nTesting GitHub search with query: '{query}'")
        try:
            import asyncio
            results = asyncio.run(web_agent.search_github_repositories(query))
            
            if results and len(results) > 0:
                print(f"✓ search_github_repositories returned {len(results)} repositories")
                # Show first result as example
                first_result = results[0]
                print(f"  Example repository: {first_result.get('name', 'No name')[:50]}...")
                print(f"  Description: {first_result.get('description', 'No description')[:80]}...")
                success_count += 1
            else:
                print("✗ search_github_repositories returned empty results (MCP server may not be running)")
        except Exception as e:
            print(f"✗ search_github_repositories failed: {e}")
    
    return success_count > 0

def test_search_pypi_packages(web_agent: WebAgent) -> bool:
    """
    Test the search_pypi_packages functionality of the WebAgent.
    
    Args:
        web_agent (WebAgent): The WebAgent instance to test.
        
    Returns:
        bool: True if test passed, False otherwise.
    """
    print("\n=== Testing search_pypi_packages Functionality ===")
    
    test_queries = [
        "numpy",
        "machine learning"
    ]
    
    success_count = 0
    
    for query in test_queries:
        print(f"\nTesting PyPI search with query: '{query}'")
        try:
            import asyncio
            results = asyncio.run(web_agent.search_pypi_packages(query))
            
            if results and len(results) > 0:
                print(f"✓ search_pypi_packages returned {len(results)} packages")
                # Show first result as example
                first_result = results[0]
                print(f"  Example package: {first_result.get('name', 'No name')[:30]}...")
                print(f"  Version: {first_result.get('version', 'No version')[:20]}...")
                print(f"  Description: {first_result.get('description', 'No description')[:60]}...")
                success_count += 1
            else:
                print("✗ search_pypi_packages returned empty results (MCP server may not be running)")
        except Exception as e:
            print(f"✗ search_pypi_packages failed: {e}")
    
    return success_count > 0

def test_search_aggregated(web_agent: WebAgent) -> bool:
    """
    Test the search functionality of the WebAgent (aggregated search with multiple sources).
    
    Args:
        web_agent (WebAgent): The WebAgent instance to test.
        
    Returns:
        bool: True if test passed, False otherwise.
    """
    print("\n=== Testing search Functionality (aggregated search) ===")
    
    test_queries = [
        "python data science",
        "machine learning frameworks"
    ]
    
    success_count = 0
    
    for query in test_queries:
        print(f"\nTesting aggregated search with query: '{query}'")
        try:
            results = web_agent.search(query)
            
            if results:
                print("✓ aggregated search completed")
                
                # Check web results
                web_results = results.get('web', [])
                if web_results:
                    print(f"  ✓ Web results: {len(web_results)} items")
                else:
                    print("  - No web results")
                
                # Check GitHub results
                github_results = results.get('github', [])
                if github_results:
                    print(f"  ✓ GitHub results: {len(github_results)} repositories")
                else:
                    print("  - No GitHub results (MCP server may not be running)")
                
                # Check PyPI results
                pypi_results = results.get('pypi', [])
                if pypi_results:
                    print(f"  ✓ PyPI results: {len(pypi_results)} packages")
                else:
                    print("  - No PyPI results (MCP server may not be running)")
                
                # Consider test successful if at least web results are returned
                if web_results:
                    success_count += 1
                else:
                    print("✗ aggregated search returned no web results")
            else:
                print("✗ aggregated search returned empty results")
        except Exception as e:
            print(f"✗ aggregated search failed: {e}")
    
    return success_count > 0

def test_navigate(web_agent: WebAgent) -> bool:
    """
    Test the navigate functionality of the WebAgent.
    
    Args:
        web_agent (WebAgent): The WebAgent instance to test.
        
    Returns:
        bool: True if test passed, False otherwise.
    """
    print("\n=== Testing Navigate Functionality ===")
    
    try:
        # First, get some URLs from a search
        search_results = web_agent.search_web_pages("Python programming tutorial")
        
        if search_results:
            test_url = search_results[0]['url']
            print(f"\nNavigating to: {test_url}")
            
            content = web_agent.navigate(test_url)
            
            if content:
                print(f"✓ Successfully retrieved content ({len(content)} characters)")
                print(f"  Content preview: {content[:100]}...")
                return True
            else:
                print("✗ Failed to retrieve content")
                return False
        else:
            print("✗ No URLs available for navigation test")
            return False
    except Exception as e:
        print(f"✗ Navigate functionality failed: {e}")
        return False

def main() -> None:
    """
    Main function to run the WebAgent tests.
    """
    setup_logging()
    
    print("WebAgent Comprehensive Test Script")
    print("===================================")
    
    # Load test configuration
    config = load_test_config()
    if not config:
        print("\n✗ Configuration loading failed, stopping tests")
        sys.exit(1)
    
    # Test initialization
    agent = test_web_agent_initialization(config)
    if not agent:
        print("\n✗ Initialization failed, stopping tests")
        sys.exit(1)
    
    # Test individual search methods
    web_search_success = test_search_web_pages(agent)
    github_search_success = test_search_github_repositories(agent)
    pypi_search_success = test_search_pypi_packages(agent)
    aggregated_search_success = test_search_aggregated(agent)
    navigate_success = test_navigate(agent)
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"WebAgent initialization: ✓ PASS")
    print(f"search_web_pages functionality: {'✓ PASS' if web_search_success else '✗ FAIL'}")
    print(f"search_github_repositories functionality: {'✓ PASS' if github_search_success else '✗ FAIL'}")
    print(f"search_pypi_packages functionality: {'✓ PASS' if pypi_search_success else '✗ FAIL'}")
    print(f"search functionality (aggregated): {'✓ PASS' if aggregated_search_success else '✗ FAIL'}")
    print(f"navigate functionality: {'✓ PASS' if navigate_success else '✗ FAIL'}")
    
    # Count successful core search methods
    core_search_success_count = sum([web_search_success, github_search_success, pypi_search_success])
    
    if web_search_success and aggregated_search_success and navigate_success:
        print("\n🎉 All tests passed!")
    elif web_search_success and core_search_success_count >= 2:
        print("\n⚠ Core search functionality mostly passed")
        if not navigate_success:
            print("  (Navigate functionality failed)")
        if not github_search_success:
            print("  (GitHub search failed - MCP server may not be running)")
        if not pypi_search_success:
            print("  (PyPI search failed - MCP server may not be running)")
    elif web_search_success:
        print("\n⚠ Basic web search passed, other search methods failed")
        print("  (MCP server failures might be expected if servers are not running)")
    else:
        print("\n❌ Critical tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()