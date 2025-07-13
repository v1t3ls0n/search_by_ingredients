import re
import unicodedata
from typing import List
# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

# Global cache for NLTK availability
_NLTK_AVAILABLE = None
_LEMMATIZER = None

def _ensure_nltk():
    """Ensure NLTK and required data are available."""
    global _NLTK_AVAILABLE, _LEMMATIZER
    
    if _NLTK_AVAILABLE is not None:
        return _NLTK_AVAILABLE
    
    try:
        import nltk
        try:
            # Try to find wordnet data
            nltk.data.find('corpora/wordnet')
            from nltk.stem import WordNetLemmatizer
            _LEMMATIZER = WordNetLemmatizer()
            _NLTK_AVAILABLE = True
        except LookupError:
            # Try to download
            try:
                import os
                nltk_data_dir = os.path.expanduser('~/.nltk_data')
                os.makedirs(nltk_data_dir, exist_ok=True)
                nltk.data.path.append(nltk_data_dir)
                
                print("NLTK wordnet data not found, downloading...")
                nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
                nltk.download('omw-1.4', download_dir=nltk_data_dir, quiet=True)
                nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir, quiet=True)
                
                from nltk.stem import WordNetLemmatizer
                _LEMMATIZER = WordNetLemmatizer()
                _NLTK_AVAILABLE = True
                print("NLTK data downloaded successfully")
            except Exception as e:
                print(f"Could not download NLTK data: {e}")
                _NLTK_AVAILABLE = False
    except ImportError:
        _NLTK_AVAILABLE = False
    
    return _NLTK_AVAILABLE

def tokenize_ingredient(text: str) -> List[str]:
    """Extract word tokens from ingredient text."""
    return re.findall(r"\b\w[\w-]*\b", text.lower())

def normalise(t: str) -> str:
    """Normalize ingredient text for consistent matching."""
    if not isinstance(t, str):
        t = str(t)
    
    # Unicode normalization - remove accents
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    
    # Remove parenthetical content and convert to lowercase
    t = re.sub(r"\([^)]*\)", " ", t.lower())
    
    # Remove units of measurement
    units = re.compile(r"\b(?:g|gram|kg|oz|ml|l|cup|cups|tsp|tbsp|teaspoon|"
                      r"tablespoon|pound|lb|slice|slices|small|large|medium)\b")
    t = units.sub(" ", t)
    
    # Remove numbers (including fractions)
    t = re.sub(r"\d+(?:[/\.]\d+)?", " ", t)
    
    # Remove punctuation
    t = re.sub(r"[^\w\s-]", " ", t)
    
    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    
    # Apply lemmatization if available
    if _ensure_nltk() and _LEMMATIZER:
        return " ".join(_LEMMATIZER.lemmatize(w) for w in t.split() if len(w) > 2)
    else:
        # Fallback: just filter short words
        return " ".join(w for w in t.split() if len(w) > 2)
