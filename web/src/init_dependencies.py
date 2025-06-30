#!/usr/bin/env python3
"""Initialize external dependencies (NLTK data and USDA database) at startup."""

import sys
import os

def init_nltk():
    """Download NLTK data if not already present."""
    try:
        import nltk
        
        # List of required NLTK data
        required_data = [
            ('corpora/wordnet', 'wordnet'),
            ('corpora/omw-1.4', 'omw-1.4'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
        ]
        
        for data_path, data_name in required_data:
            try:
                nltk.data.find(data_path)
                print(f"✓ NLTK {data_name} data already present")
            except LookupError:
                print(f"↓ Downloading NLTK {data_name} data...")
                nltk.download(data_name, quiet=True)
                print(f"✓ NLTK {data_name} downloaded successfully")
                
        return True
    except Exception as e:
        print(f"⚠ Warning: Could not setup NLTK: {e}")
        print("  Continuing without lemmatization support...")
        return False

def init_usda():
    """Pre-download USDA nutritional database."""
    try:
        # Add the web directory to path
        sys.path.insert(0, '/app/web')
        
        from diet_classifiers import _download_and_extract_usda
        
        print("↓ Checking USDA nutritional database...")
        usda_dir = _download_and_extract_usda()
        
        if usda_dir:
            print("✓ USDA database ready")
            return True
        else:
            print("⚠ Warning: USDA database could not be downloaded")
            print("  Continuing with rule-based classification only...")
            return False
            
    except Exception as e:
        print(f"⚠ Warning: Could not setup USDA database: {e}")
        print("  Continuing with rule-based classification only...")
        return False

def main():
    """Initialize all dependencies."""
    print("=== Initializing Dependencies ===")
    print()
    
    # Initialize NLTK
    print("1. NLTK Setup:")
    nltk_success = init_nltk()
    print()
    
    # Initialize USDA
    print("2. USDA Database Setup:")
    usda_success = init_usda()
    print()
    
    print("=== Initialization Complete ===")
    if nltk_success and usda_success:
        print("✓ All dependencies initialized successfully")
    else:
        print("⚠ Some dependencies could not be initialized")
        print("  The application will continue with fallback options")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())