#!/usr/bin/env python3
"""
Fix numpy array boolean context errors in diet_classifiers.py
"""

import re

# Read the file
with open('diet_classifiers.py', 'r') as f:
    content = f.read()

# List of patterns to fix
replacements = [
    # Fix img_silver checks
    (r'if mode in {"image", "both"} and img_silver and img_silver\.size > 0:',
     'if mode in {"image", "both"} and img_silver is not None and img_silver.size > 0:'),
    
    (r'if mode == "both" and img_silver and img_silver\.size > 0:',
     'if mode == "both" and img_silver is not None and img_silver.size > 0:'),
    
    (r'elif mode == "image" and img_silver and img_silver\.size > 0:',
     'elif mode == "image" and img_silver is not None and img_silver.size > 0:'),
    
    # Fix img_gold checks
    (r'if img_gold and img_gold\.size > 0:',
     'if img_gold is not None and img_gold.size > 0:'),
    
    # Fix generic patterns
    (r'if (\w+) and \1\.size > 0:',
     r'if \1 is not None and \1.size > 0:'),
    
    # Fix in conditions with parentheses
    (r'\(img_silver\) and \(hasattr\(img_silver, \'size\'\)\) and \(img_silver\.size > 0\)',
     'img_silver is not None and hasattr(img_silver, "size") and img_silver.size > 0'),
    
    (r'\(img_gold\) and \(hasattr\(img_gold, \'size\'\)\) and \(img_gold\.size > 0\)',
     'img_gold is not None and hasattr(img_gold, "size") and img_gold.size > 0'),
]

# Apply all replacements
for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content)

# Write the fixed content back
with open('diet_classifiers.py', 'w') as f:
    f.write(content)

print("✅ Fixed numpy array boolean context errors")
print("Fixed patterns:")
for pattern, _ in replacements:
    print(f"  - {pattern}")