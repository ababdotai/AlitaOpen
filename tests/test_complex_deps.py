#!/usr/bin/env python3
"""
Test script to verify complex dependency installation.
"""

import sys
import yaml

# Add the parent directory to the Python path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_manager import EnvironmentManager

def test_complex_dependencies():
    """Test installation of multiple complex dependencies."""
    
    print("Testing complex dependency installation...")
    
    try:
        # Load configuration
        with open('config.yaml.example', 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize EnvironmentManager
        em = EnvironmentManager(config)
        
        # Test complex dependencies
        test_env = "a41585_1"
        complex_dependencies = [
            "opencv-python>=4.5.0",
            "numpy>=1.21.0",
            "pandas>=1.3.0"
        ]
        
        print(f"Testing complex dependencies in environment: {em._get_full_env_name(test_env)}")
        print(f"Dependencies to install: {complex_dependencies}")
        
        # Install dependencies
        result = em.install_dependencies(test_env, complex_dependencies)
        
        print(f"Complex installation result: {result}")
        
        if result:
            print("✅ Complex dependency installation test PASSED")
        else:
            print("❌ Complex dependency installation test FAILED")
            
        return result
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complex_dependencies()
    sys.exit(0 if success else 1)