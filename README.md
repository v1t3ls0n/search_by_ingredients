# 🥑 Solution For Argmax's Search By Ingredients Challenge By **Guy Vitelson**

---
##### Ping me via 🔗 **[Linkedin](https://www.linkedin.com/in/guyvitelson/)**  🐙 **[GitHub](https://github.com/v1t3ls0n)**  ✉️ **[Mail](mailto:guyvitelson@gmail.com)**
---

## 🧠 My Implementation

### Core Solution (`web/src/diet_classifiers.py`)

I've implemented a comprehensive rule-based diet classification system with the following key features:

#### 🥑 Keto Classification Pipeline
My keto classifier uses a multi-stage decision pipeline:

1. **Whitelist Override** - Immediate acceptance for known keto ingredients (almond flour, coconut oil, etc.)
2. **USDA Nutritional Lookup** - Downloads and integrates USDA FoodData Central database for authoritative carbohydrate content (≤10g/100g = keto-friendly)
3. **Fuzzy Matching** - Uses RapidFuzz (when available) for intelligent ingredient matching with 90% similarity threshold
4. **Regex Blacklist** - Pattern matching against comprehensive NON_KETO list
5. **Token-level Analysis** - Handles multi-word ingredients and edge cases

#### 🌱 Vegan Classification Pipeline
The vegan classifier implements:

1. **Whitelist Override** - Handles edge cases like "eggplant" (not "egg"), "peanut butter" (not dairy "butter")
2. **Text Normalization** - Same preprocessing pipeline as keto (Unicode normalization, unit removal, lemmatization)
3. **Comprehensive Animal Product Detection** - Extensive blacklist covering meat, dairy, seafood, eggs, and other animal-derived ingredients
4. **Smart Pattern Matching** - Regex patterns with word boundaries to avoid false positives

#### 🔧 Shared Advanced Features
Both classifiers leverage the same robust preprocessing pipeline:
- **Unicode Normalization** - Handles accented characters and special formatting
- **Text Preprocessing** - Removes units, numbers, parenthetical content for cleaner matching
- **Lemmatization** - Optional NLTK integration for better word matching (falls back gracefully)
- **Token Analysis** - Intelligent word-level processing for multi-word ingredients
- **Caching** - USDA database downloaded once and cached locally
- **Graceful Fallbacks** - Continues with rule-based classification if external dependencies unavailable

#### 📊 Performance Characteristics
- **Fast Execution** - Rule-based approach with compiled regex patterns
- **High Accuracy** - Leverages authoritative nutritional data from USDA
- **Robust Handling** - Comprehensive edge case coverage and error handling
- **Memory Efficient** - Lazy loading and intelligent caching

### Testing
You can test the implementation by running:
```bash
python nb/src/diet_classifiers.py --ground_truth /usr/src/data/ground_truth_sample.csv
```

## 🤖 Advanced ML Solution

For those interested in a more sophisticated approach, I've developed a comprehensive machine learning solution available on the `ml-overskill-solution` branch. This advanced implementation includes:

### 🚀 Full ML Pipeline Features
- **Weak Supervision** - Silver label generation using rule-based heuristics combined with USDA nutritional data
- **Multi-Modal Learning** - Text (TF-IDF) + Image (ResNet-50) feature extraction and fusion
- **Hierarchical Ensembles** - 4-level ensemble architecture with dynamic weighting
- **15+ ML Models** - Including Logistic Regression, Random Forest, LightGBM, Neural Networks
- **Advanced Training** - SMOTE class balancing, hyperparameter tuning, cross-validation
- **Production Features** - Memory management, error handling, Docker containerization, comprehensive logging

### 🎯 Model Performance
- **Keto Models**: F1-scores up to 0.963 with sophisticated ensemble methods
- **Vegan Models**: Perfect F1-scores (1.0) achieved with multi-modal ensembles
- **Image Processing**: 70K+ recipe images processed with GPU acceleration
- **Scale**: Handles datasets with millions of recipes efficiently

### 🏗️ Architecture Highlights
- **Modular Design** - Clean separation into 40+ focused modules
- **Memory Optimization** - Intelligent sparse matrix usage and GPU memory management
- **Ensemble Methods** - Top-N model selection, alpha blending, recursive ensembles
- **Rule Verification** - ML predictions verified against domain constraints

The ML solution demonstrates advanced techniques in weak supervision, ensemble learning, and production-grade ML systems while maintaining backward compatibility with the core rule-based approach.