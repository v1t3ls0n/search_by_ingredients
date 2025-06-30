# 🥑 Solution For Argmax's Search By Ingredients Challenge By **Guy Vitelson**

---
##### Ping me via 🔗 **[Linkedin](https://www.linkedin.com/in/guyvitelson/)**  🐙 **[GitHub](https://github.com/v1t3ls0n)**  ✉️ **[Mail](mailto:guyvitelson@gmail.com)**
---

## 📋 What's Provided (Boilerplate) vs What I Built

### Boilerplate Provided:
- Basic Docker setup with OpenSearch, Flask web app, and Jupyter notebook
- Empty `is_ingredient_keto()` and `is_ingredient_vegan()` functions with TODO placeholders
- Recipe search infrastructure and UI
- Basic requirements without ML/NLP libraries

### My Solution Adds:
- ✅ **Complete implementation** of keto/vegan classifiers achieving 100% accuracy
- ✅ **Enhanced dependencies**: Added NLTK, RapidFuzz, scikit-learn to requirements
- ✅ **USDA nutritional database** integration with 303 food items
- ✅ **Custom initialization pipeline** (`init_dependencies.py`)
- ✅ **Comprehensive test suite** (`test_diet_classifiers.sh`)
- ✅ **Dockerfile optimizations** for faster startup
- ✅ **358 non-keto and 160 non-vegan** ingredient blacklists
- ✅ **Whitelist patterns** for edge cases (eggplant, peanut butter, etc.)

## 🧠 My Implementation

### Core Solution (`web/src/diet_classifiers.py`)

I replaced the TODO stubs with a comprehensive rule-based diet classification system:

#### 🥑 Keto Classification Pipeline
My keto classifier uses a sophisticated multi-stage decision pipeline:

1. **Whitelist Override** - Immediate acceptance for known keto ingredients
   - Added patterns: `r"\balmond flour\b"`, `r"\bheavy cream\b"`, `r"\bcoconut oil\b"`
   - Covers keto staples: nut flours, high-fat dairy, MCT oils, sugar-free sweeteners
   - Double-check approach: Tests both original and normalized forms

2. **Domain Blacklist (Priority)** - Hard rules override database lookups
   - Created NON_KETO list: 358 items covering grains, fruits, legumes, sugars
   - Regex patterns with word boundaries: `r"\b(?:rice|bread|sugar|banana)\b"`

3. **Token-level Blacklist Analysis** - Multi-word ingredient detection
   - Handles compound ingredients like "kidney beans", "sweet potato"

4. **USDA Nutritional Fallback** - Scientific validation for unknown ingredients
   - Added USDA FoodData Central database integration
   - Carbohydrate threshold: ≤10g/100g = keto-friendly
   - Fuzzy matching with 90% similarity threshold using RapidFuzz

5. **Intelligent Preprocessing** - Robust text normalization
   - Unicode normalization, unit removal, quantity stripping
   - Handles measurements: "2 cups almond flour" → "almond flour"

#### 🌱 Vegan Classification Pipeline
The vegan classifier implements a precision-focused approach:

1. **Whitelist Override** - Handles tricky edge cases
   - Plant-based exceptions: `r"\beggplant\b"` (not "egg"), `r"\bpeanut butter\b"` (not dairy)
   - Alternative products: `r"\balmond milk\b"`, `r"\bcoconut cream\b"`

2. **Comprehensive Animal Product Detection** - NON_VEGAN blacklist
   - 160 items: meat, dairy, eggs, honey, gelatin, etc.
   - Special handling for alcohol: kahlua (contains dairy)

3. **Smart Pattern Matching** - Regex with word boundaries
   - Prevents false positives: "butternut squash" doesn't match "butter"

## 📁 Files I Modified/Added

### Modified Files:
1. **`web/src/diet_classifiers.py`** - Complete implementation (was just TODOs)
2. **`web/requirements.txt`** - Added: scikit-learn, nltk, rapidfuzz
3. **`web/Dockerfile`** - Modified CMD to use init.sh
4. **`web/src/init.sh`** - Added NLTK initialization
5. **`nb/src/diet_classifiers.py`** - Complete implementation with CLI
6. **`nb/requirements.txt`** - Added: nltk, rapidfuzz
7. **`nb/Dockerfile`** - Added NLTK data download during build

### New Files Created:
1. **`web/src/init_dependencies.py`** - Handles NLTK/USDA downloads
2. **`test_diet_classifiers.sh`** - Comprehensive test suite
3. **This README.md** - Complete documentation

## 🚀 Quick Start

```bash
# Build and run (same as boilerplate)
docker compose build
docker compose up -d

# Run my test suite (new)
./test_diet_classifiers.sh

# View the web app with working keto/vegan badges
open http://localhost:8080
```

## 📊 Performance

My solution achieves on the provided test set:
- **Keto Classification**: 100% accuracy (463.5 recipes/second)
- **Vegan Classification**: 100% accuracy
- **Zero false negatives** for both classifiers

## 🔧 Key Implementation Details

### Text Preprocessing Pipeline
```python
def normalise(t: str) -> str:
    # Unicode normalization - remove accents
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    # Remove parenthetical content
    t = re.sub(r"\([^)]*\)", " ", t.lower())
    # Remove units of measurement
    # Remove numbers and fractions
    # Apply lemmatization if available
    return processed_text
```

### Custom Ingredient Parser
Handles malformed CSV format from ground truth:
```python
def parse_ingredients(ingredients_str):
    # Handles: "['ingredient1' 'ingredient2']" → ["ingredient1", "ingredient2"]
    content = ingredients_str.strip()[2:-2]
    ingredients = re.split(r"'\s+'", content)
    return ingredients
```

### USDA Integration
```python
def carbs_per_100g(ingredient: str, fuzzy: bool = True) -> Optional[float]:
    # Downloads USDA database on first run
    # Caches to ~/.cache/diet_classifier
    # Uses RapidFuzz for 90% similarity matching
```

## 🧪 Testing

The included test suite validates:
- Individual ingredients (48 test cases)
- Recipe parsing edge cases
- USDA database connectivity
- Cross-container consistency
- Performance benchmarks

```bash
# Example test output:
✅ heavy cream -> True (expected True)
✅ strawberries -> False (expected False)
🎯 Keto Accuracy: 24/24 (100.0%)
⚡ Processing rate: 463.5 recipes/second
```

## 🏗️ Architecture Decisions

1. **Rule-based over ML**: Deterministic, explainable, no training data needed
2. **Domain knowledge first**: Blacklists override USDA data
3. **Graceful degradation**: Works without internet after initial setup
4. **Production-ready**: Fast startup, proper error handling, comprehensive logging

## 🤖 Advanced ML Solution

I've also developed a comprehensive ML solution on the `ml-overskill-solution` branch featuring:
- Weak supervision with silver labels
- Multi-modal learning (text + images)
- 15+ ML models with ensemble methods
- F1-scores up to 0.963 (keto) and 1.0 (vegan)

---

This solution extends the provided boilerplate with production-ready enhancements while maintaining the original interface and structure.