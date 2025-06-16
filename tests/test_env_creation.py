#!/usr/bin/env python3

import sys
import os

# Add the parent directory to the Python path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_manager import EnvironmentManager
from utils import load_config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_env_creation_and_deps():
    """Test environment creation with Python and dependency installation."""
    try:
        # Load configuration
        config = load_config()
        
        # Initialize EnvironmentManager
        env_manager = EnvironmentManager(config)
        
        # Test environment name
        test_env_name = "test_env_creation"
        
        print(f"Testing environment creation: {test_env_name}")
        
        # Test dependencies
        test_deps = ["requests>=2.25.0", "numpy"]
        
        # Create environment
        success = env_manager.create_environment(test_env_name)
        if not success:
            print("Environment creation failed!")
            return False
        
        print("Environment created successfully!")
        
        # Test dependency installation
        print(f"Testing dependency installation: {test_deps}")
        
        success = env_manager.install_dependencies(test_env_name, test_deps)
        if not success:
            print("Dependency installation failed!")
            return False
        
        print("Dependencies installed successfully!")
        
        print("\n✅ Test completed successfully! Environment creation and dependency installation work correctly.")
        print(f"Note: Test environment '{env_manager._get_full_env_name(test_env_name)}' was created and can be manually removed if needed.")
        
        return True
        
    except Exception as e:
        print(f"Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = test_env_creation_and_deps()
    if success:
        print("\n✅ All tests passed! Environment creation and dependency installation work correctly.")
    else:
        print("\n❌ Tests failed! Check the logs for details.")
        sys.exit(1)