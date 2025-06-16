#!/usr/bin/env python3
"""
Integration test to verify the modified script generation and dependency extraction workflow.
"""

import sys
import os
import yaml

# Add the parent directory to the Python path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script_generator import ScriptGenerator

def load_config():
    """Load configuration from config.yaml.example"""
    try:
        with open('config.yaml.example', 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def test_script_generation():
    """Test the modified script generation with XML format."""
    
    print("Testing modified script generation workflow...")
    
    # Load configuration
    config = load_config()
    if not config:
        print("Failed to load configuration")
        return False
    
    # Mock API key for testing (won't actually call API)
    config['api']['openai_api_key'] = 'test-key'
    config['api']['openai_api_url'] = 'https://api.openai.com/v1'
    
    try:
        # Initialize ScriptGenerator
        script_gen = ScriptGenerator(config)
        print("ScriptGenerator initialized successfully")
        
        # Test the template loading
        print(f"Template loaded from: {script_gen.prompt_template_path}")
        print(f"Template contains XML format instructions: {'<requirements>' in script_gen.prompt_template}")
        
        # Test XML parsing with mock response
        mock_response = '''
<requirements>
opencv-python>=4.5.0
moviepy>=1.0.3
numpy>=1.21.0
</requirements>

<code>
#!/usr/bin/env python3
"""
Test script for video processing
"""

import cv2
import numpy as np

def main():
    print("Video processing script")
    return True

if __name__ == "__main__":
    main()
</code>
        '''
        
        # Test the _clean_script method
        script_gen._extracted_requirements = None
        cleaned_script = script_gen._clean_script(mock_response)
        
        print("\nXML Parsing Results:")
        print(f"Extracted requirements: {getattr(script_gen, '_extracted_requirements', 'None')}")
        print(f"Cleaned script (first 100 chars): {cleaned_script[:100]}...")
        
        # Test requirements parsing
        if hasattr(script_gen, '_extracted_requirements') and script_gen._extracted_requirements:
            requirements_list = [req.strip() for req in script_gen._extracted_requirements.split('\n') if req.strip()]
            print(f"Requirements as list: {requirements_list}")
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_script_generation()
    sys.exit(0 if success else 1)