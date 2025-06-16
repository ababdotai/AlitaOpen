#!/usr/bin/env python3
"""
Test script to verify XML parsing functionality for requirements and code extraction.
"""

import re

def test_xml_parsing():
    """Test the XML parsing logic for requirements and code sections."""
    
    # Sample XML format output
    sample_text = '''
<requirements>
opencv-python>=4.5.0
moviepy>=1.0.3
numpy>=1.21.0
pandas>=1.3.0
</requirements>

<code>
#!/usr/bin/env python3
"""
Sample script for video processing
"""

import cv2
import numpy as np

def main():
    print("Hello from video processing script!")
    return True

if __name__ == "__main__":
    main()
</code>
    '''
    
    print("Testing XML parsing...")
    print("Sample text:")
    print(sample_text)
    print("\n" + "="*50 + "\n")
    
    # Extract requirements section
    requirements_match = re.search(r'<requirements>(.*?)</requirements>', sample_text, re.DOTALL)
    code_match = re.search(r'<code>(.*?)</code>', sample_text, re.DOTALL)
    
    if requirements_match:
        requirements = requirements_match.group(1).strip()
        print("Extracted Requirements:")
        print(requirements)
        print("\nRequirements as list:")
        req_list = [req.strip() for req in requirements.split('\n') if req.strip()]
        print(req_list)
    else:
        print("No requirements section found")
    
    print("\n" + "-"*30 + "\n")
    
    if code_match:
        code = code_match.group(1).strip()
        print("Extracted Code:")
        print(code)
    else:
        print("No code section found")

if __name__ == "__main__":
    test_xml_parsing()