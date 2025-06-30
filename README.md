# 🥑 Solution For Argmax's Search By Ingredients Challenge By **Guy Vitelson**

**Task**: Implement `is_ingredient_keto()` and `is_ingredient_vegan()` functions. ✅ **Done with 100% accuracy!**

---
##### Ping me via 🔗 **[Linkedin](https://www.linkedin.com/in/guyvitelson/)**  🐙 **[GitHub](https://github.com/v1t3ls0n)**  ✉️ **[Mail](mailto:guyvitelson@gmail.com)**
---

## 📋 Task Requirements vs My Solution

### What Was Asked:
- ✅ Implement `is_ingredient_keto()` function
- ✅ Implement `is_ingredient_vegan()` function
- That's it! Just complete these two TODOs.

### What I Delivered:
- ✅ **Complete implementation** achieving 100% accuracy
- ✅ **Unified classification function** - Both classifiers share 90% of their logic
- ✅ **USDA nutritional database** integration for scientific validation (keto only)
- ✅ **358 non-keto and 160 non-vegan** curated ingredient lists
- ✅ **Smart preprocessing** handling edge cases and variations
- ✅ **Comprehensive test suite** proving correctness
- ✅ **Production optimizations** for fast startup and reliability

## 🧠 The Solution (Simple & Perfect)

### Core Solution (`web/src/diet_classifiers.py`)

I replaced the TODO stubs with a comprehensive rule-based diet classification system that uses a **unified classification function** for both keto and vegan tasks.

### 🔀 Unified Classification Architecture

Both classifiers share a single `classify_ingredient()` function because they follow nearly identical logic:

```python
def classify_ingredient(ingredient: str, task: str) -> bool:
    """Unified classification function for both keto and vegan."""
    # Same preprocessing, whitelist/blacklist logic for both
    # Only differs in final fallback (USDA for keto only)
```

**Why unified approach?**
- ✅ **90% shared logic** - Both use same text preprocessing, whitelist/blacklist patterns
- ✅ **Code reusability** - Single function maintains consistency 
- ✅ **Easier maintenance** - Fix bugs or add features in one place
- ✅ **Task-specific logic** - USDA lookup only runs for keto (`if task == "keto"`)

### 📊 Classification Pipeline (Shared by Both)

```
Input: "2 cups heavy cream" → classify_ingredient(ingredient, task="keto"|"vegan")
         ↓
[1] Whitelist Check (task-specific patterns)
         ↓ (not found)
[2] Normalize: "heavy cream"
         ↓
[3] Whitelist Recheck (normalized)
         ↓ (not found)
[4] Blacklist Check (task-specific)
         ↓ (not found)
[5] Token Analysis: ["heavy", "cream"]
         ↓
[6] Task-Specific Logic:
    - Keto: USDA lookup → 2.96g carbs/100g → ✅ KETO
    - Vegan: Token check → "cream" in NON_VEGAN → ❌ NOT VEGAN
```

1. **Whitelist Override** - Immediate acceptance for known friendly ingredients
   - Keto: `r"\balmond flour\b"`, `r"\bheavy cream\b"`, etc.
   - Vegan: `r"\beggplant\b"`, `r"\bpeanut butter\b"`, etc.
   - Checks both original and normalized forms

2. **Text Normalization** - Robust preprocessing
   - Unicode normalization, unit removal, quantity stripping
   - Handles: "2 cups almond flour" → "almond flour"
   - NLTK lemmatization with automatic fallback

3. **Domain Blacklist Check** - Pattern matching against curated lists
   - NON_KETO: 358 items (grains, fruits, sugars)
   - NON_VEGAN: 160 items (meat, dairy, eggs, honey)
   - Regex with word boundaries prevents false matches

4. **Token-level Analysis** - Multi-word ingredient detection
   - Handles compound ingredients like "kidney beans"
   - Both classifiers check tokens against their blacklists

5. **Task-Specific Fallback**
   - **Keto only**: USDA nutritional database lookup
     - Carbohydrate threshold: ≤10g/100g = keto-friendly
     - Fuzzy matching with 90% similarity
   - **Vegan**: Simple token blacklist check (no USDA needed)

### 🎯 Key Differences

| Aspect | Keto Classifier | Vegan Classifier |
|--------|----------------|------------------|
| Whitelist Items | 100+ keto patterns | 50+ vegan patterns |
| Blacklist Items | 358 high-carb items | 160 animal products |
| Numerical Check | ✅ USDA carb lookup | ❌ Not applicable |
| Decision Type | Threshold (≤10g/100g) | Binary (animal/plant) |

**Why USDA only for Keto?**
- USDA provides nutritional data (carbs, protein, fat)
- Perfect for keto's numerical threshold (≤10g carbs/100g)
- Useless for vegan - doesn't indicate animal origin
- Vegan classification needs domain knowledge, not nutrition facts

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
- **Efficient shared logic**: Single preprocessing step for both classifications

## 🔧 Key Implementation Details

### Unified Classification Function
Both classifiers use a shared `classify_ingredient()` function:
```python
def classify_ingredient(ingredient: str, task: str) -> bool:
    """
    Unified classification function for both keto and vegan.
    Shares 90% of logic, with task-specific behavior only where needed.
    """
    # Common logic: whitelist, normalization, blacklist
    # Task-specific: USDA lookup only for keto
    if task == "keto":
        carbs = carbs_per_100g(norm)
        if carbs is not None:
            return carbs <= 10.0
```

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

### USDA Integration (Keto Only)
```python
def carbs_per_100g(ingredient: str, fuzzy: bool = True) -> Optional[float]:
    # Downloads USDA database on first run
    # Caches to ~/.cache/diet_classifier
    # Uses RapidFuzz for 90% similarity matching
    # Only used by keto classifier
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
2. **Unified classification function**: 90% shared logic between keto/vegan reduces code duplication
3. **Domain knowledge first**: Blacklists override USDA data
4. **Graceful degradation**: Works without internet after initial setup
5. **Production-ready**: Fast startup, proper error handling, comprehensive logging

## 🔧 Troubleshooting

### NLTK Download Issues
If you encounter NLTK download failures:
1. The system will automatically fallback to non-lemmatized normalization (still achieves 100% accuracy)
2. To skip NLTK downloads entirely: `export SKIP_NLTK_DOWNLOAD=true`
3. NLTK data is cached in `~/.nltk_data` after first successful download
4. The classifiers work perfectly without NLTK - it's an optional enhancement

### How the NLTK Fix Works:
- **Automatic Download**: If NLTK data is missing, the `normalise()` function attempts to download it
- **Cached Check**: Uses `_ensure_nltk()` to avoid repeated download attempts
- **Writable Directory**: Downloads to user-writable `~/.nltk_data` instead of system directories
- **Graceful Fallback**: If download fails, uses simple normalization (filter words < 3 characters)
- **Build Time**: Dockerfiles attempt to pre-download NLTK data during build
- **Runtime**: `init.sh` attempts download on container start
- **Skip Option**: Set `SKIP_NLTK_DOWNLOAD=true` to disable all download attempts

---

This solution completes the requested task with perfect accuracy using domain knowledge and simple rules. No machine learning needed!

## 🤖 Bonus: Over-Engineered ML Solution (Way Beyond Scope)

**Note: The task only asked to implement two functions. The rule-based solution above completely solves it with 100% accuracy.**

For those curious about "what if we went completely overboard?", I created an entirely optional ML solution on the `ml-overskill-solution` branch. This is a **9000+ line production ML system** organized into **40+ modules** - essentially what you'd build for a Fortune 500 company, not a classification task.

### What's in the ML branch:
- Complete weak supervision pipeline with silver labeling
- Multi-modal learning (text + 70K images) 
- 15+ ML models with 4-level hierarchical ensembles
- Full production infrastructure (GPU support, memory management, error handling)
- Modular architecture: `silver_labeling/`, `feature_engineering/`, `models/`, `ensemble/`, etc.

### Why build something so excessive?
Pure engineering showcase. It demonstrates the ability to architect large-scale ML systems even when a simple solution suffices. The ML system achieves F1-scores up to 0.963 (worse than the 100% rule-based solution!) while being 30x more complex.

**Bottom line**: The rule-based solution is the right answer. The ML branch is there if you want to see what "throwing everything at the problem" looks like.