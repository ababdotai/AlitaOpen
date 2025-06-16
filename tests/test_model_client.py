#!/usr/bin/env python3
"""
Test script for the ModelClientFactory to verify OpenAI and Anthropic model support.
"""

import os
import sys
import time
from typing import Dict, Any

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_client import ModelClientFactory
from utils import load_config

def test_openai_client():
    """Test OpenAI client creation and API connectivity."""
    print("Testing OpenAI client...")
    
    # Load config
    config = load_config("config.yaml")
    
    # Temporarily set primary_llm to an OpenAI model
    config["agent"]["primary_llm"] = "gpt-3.5-turbo"
    
    try:
        client = ModelClientFactory.create_client(config)
        print(f"✓ OpenAI client created successfully")
        print(f"✓ Model name: {client.get_model_name()}")
        print(f"✓ Client type: {type(client).__name__}")
        
        # Test actual API call
        print("  Testing API connectivity...")
        test_messages = [{"role": "user", "content": "Hello! Please respond with just 'API test successful'"}]
        response = client.create_completion(
            messages=test_messages,
            temperature=0.1,
            max_tokens=50
        )
        print(response)
    
        if response and len(response.strip()) > 0:
            print(f"✓ API call successful, response: {response.strip()[:50]}...")
            return True
        else:
            print("✗ API call returned empty response")
            return False
            
    except Exception as e:
        print(f"✗ OpenAI test failed: {e}")
        return False

def test_anthropic_client():
    """Test Anthropic client creation and API connectivity."""
    print("\nTesting Anthropic client...")
    
    # Load config
    config = load_config("config.yaml")
    
    # Temporarily set primary_llm to an Anthropic model
    config["agent"]["primary_llm"] = "claude-3-5-haiku-20241022"
    
    try:
        client = ModelClientFactory.create_client(config)
        print(f"✓ Anthropic client created successfully")
        print(f"✓ Model name: {client.get_model_name()}")
        print(f"✓ Client type: {type(client).__name__}")
        
        # Test actual API call
        print("  Testing API connectivity...")
        test_messages = [{"role": "user", "content": "Hello! Please respond with just 'API test successful'"}]
        response = client.create_completion(
            messages=test_messages,
            temperature=0.1,
            max_tokens=50
        )
        
        if response and len(response.strip()) > 0:
            print(f"✓ API call successful, response: {response.strip()[:50]}...")
            return True
        else:
            print("✗ API call returned empty response")
            return False
            
    except Exception as e:
        print(f"✗ Anthropic test failed: {e}")
        return False

def test_api_robustness():
    """Test API robustness with different scenarios."""
    print("\n=== Testing API Robustness ===")
    
    config = load_config("config.yaml")
    success_count = 0
    total_tests = 0
    
    # Test 1: Long conversation
    print("\nTest 1: Multi-turn conversation simulation...")
    try:
        config["agent"]["primary_llm"] = "gpt-3.5-turbo"
        client = ModelClientFactory.create_client(config)
        
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "What about 3+3?"}
        ]
        
        response = client.create_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=30
        )
        
        if response and "6" in response:
            print("✓ Multi-turn conversation test passed")
            success_count += 1
        else:
            print(f"✗ Multi-turn conversation test failed: {response}")
        total_tests += 1
            
    except Exception as e:
        print(f"✗ Multi-turn conversation test error: {e}")
        total_tests += 1
    
    # Test 2: Different temperature settings
    print("\nTest 2: Temperature variation test...")
    try:
        config["agent"]["primary_llm"] = "gpt-3.5-turbo"
        client = ModelClientFactory.create_client(config)
        
        messages = [{"role": "user", "content": "Say 'temperature test'"}]
        
        # Test with temperature 0 (deterministic)
        response1 = client.create_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=20
        )
        
        # Test with temperature 0.8 (more creative)
        response2 = client.create_completion(
            messages=messages,
            temperature=0.8,
            max_tokens=20
        )
        
        print(response1)
        print(response2)
        if response1 and response2:
            print("✓ Temperature variation test passed")
            success_count += 1
        else:
            print("✗ Temperature variation test failed")
        total_tests += 1
            
    except Exception as e:
        print(f"✗ Temperature variation test error: {e}")
        total_tests += 1
    
    # Test 3: Invalid model handling
    print("\nTest 3: Invalid model handling...")
    try:
        config["agent"]["primary_llm"] = "invalid-model-name"
        client = ModelClientFactory.create_client(config)
        print("✗ Invalid model test failed - should have raised exception")
        total_tests += 1
    except Exception as e:
        print(f"✓ Invalid model test passed - correctly raised: {type(e).__name__}")
        success_count += 1
        total_tests += 1
    
    print(f"\nRobustness tests: {success_count}/{total_tests} passed")
    return success_count == total_tests

def test_performance_and_errors():
    """Test performance metrics and error handling."""
    print("\n=== Testing Performance and Error Handling ===")
    
    config = load_config("config.yaml")
    success_count = 0
    total_tests = 0
    
    # Test 1: Response time measurement
    print("\nTest 1: Response time measurement...")
    try:
        config["agent"]["primary_llm"] = "gpt-3.5-turbo"
        client = ModelClientFactory.create_client(config)
        
        messages = [{"role": "user", "content": "Count from 1 to 5"}]
        
        start_time = time.time()
        response = client.create_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=50
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        if response and response_time < 30:  # Should respond within 30 seconds
            print(f"✓ Response time test passed ({response_time:.2f}s)")
            success_count += 1
        else:
            print(f"✗ Response time test failed ({response_time:.2f}s or empty response)")
        total_tests += 1
            
    except Exception as e:
        print(f"✗ Response time test error: {e}")
        total_tests += 1
    
    # Test 2: Empty message handling
    print("\nTest 2: Empty message handling...")
    try:
        config["agent"]["primary_llm"] = "gpt-3.5-turbo"
        client = ModelClientFactory.create_client(config)
        
        messages = [{"role": "user", "content": ""}]
        
        response = client.create_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=20
        )
        
        # Should handle empty messages gracefully
        if response is not None:  # Any response (even error) is acceptable
            print("✓ Empty message handling test passed")
            success_count += 1
        else:
            print("✗ Empty message handling test failed")
        total_tests += 1
            
    except Exception as e:
        print(f"✓ Empty message handling test passed - correctly raised: {type(e).__name__}")
        success_count += 1
        total_tests += 1
    
    # Test 3: Very long input handling
    print("\nTest 3: Long input handling...")
    try:
        config["agent"]["primary_llm"] = "gpt-3.5-turbo"
        client = ModelClientFactory.create_client(config)
        
        # Create a reasonably long message (not too long to avoid token limits)
        long_content = "Please summarize this text: " + "This is a test sentence. " * 10
        messages = [{"role": "user", "content": long_content}]
        
        response = client.create_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=100
        )
        
        if response and len(response.strip()) > 0:
            print(response)
            print("✓ Long input handling test passed")
            success_count += 1
        else:
            print("✗ Long input handling test failed")
        total_tests += 1
            
    except Exception as e:
        print(f"✗ Long input handling test error: {e}")
        total_tests += 1
    
    # Test 4: Max tokens limit
    print("\nTest 4: Max tokens limit test...")
    try:
        config["agent"]["primary_llm"] = "gpt-3.5-turbo"
        client = ModelClientFactory.create_client(config)
        
        messages = [{"role": "user", "content": "Write a very long story about a cat"}]
        
        response = client.create_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=10  # Very small limit
        )
        
        if response and len(response.split()) <= 15:  # Should respect token limit roughly
            print("✓ Max tokens limit test passed")
            success_count += 1
        else:
            print(f"✗ Max tokens limit test failed - response too long: {len(response.split()) if response else 0} words")
        total_tests += 1
            
    except Exception as e:
        print(f"✗ Max tokens limit test error: {e}")
        total_tests += 1
    
    print(f"\nPerformance and error tests: {success_count}/{total_tests} passed")
    return success_count == total_tests

def main():
    """Run all tests."""
    print("=== Comprehensive Model Client Test Suite ===")
    
    # Basic functionality tests
    openai_success = test_openai_client()
    anthropic_success = test_anthropic_client()
    
    # Advanced tests
    robustness_success = test_api_robustness()
    performance_success = test_performance_and_errors()
    
    # Summary
    total_test_suites = 4
    passed_suites = sum([openai_success, anthropic_success, robustness_success, performance_success])
    
    print("\n" + "="*50)
    print("=== FINAL TEST SUMMARY ===")
    print("="*50)
    print(f"OpenAI Client Test:           {'✓ PASSED' if openai_success else '✗ FAILED'}")
    print(f"Anthropic Client Test:        {'✓ PASSED' if anthropic_success else '✗ FAILED'}")
    print(f"API Robustness Test:          {'✓ PASSED' if robustness_success else '✗ FAILED'}")
    print(f"Performance & Error Test:     {'✓ PASSED' if performance_success else '✗ FAILED'}")
    print("-" * 50)
    print(f"Overall Result: {passed_suites}/{total_test_suites} test suites passed")
    
    if passed_suites == total_test_suites:
        print("🎉 ALL TESTS PASSED! Model client is ready for production.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the configuration and API keys.")
        return 1

if __name__ == "__main__":
    sys.exit(main())