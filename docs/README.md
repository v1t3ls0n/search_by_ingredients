
# 🥑 Search By Ingredients 2.0: Multi-Diet, Portion-Aware, and Transformative Recipe Search

**By Guy Vitelson**

---

**Task**: Originally, implement `is_ingredient_keto()` and `is_ingredient_vegan()`.  
**Now:** The project has evolved into a powerful, extensible tool for analyzing, searching, and transforming recipes for multiple diets (Keto, Vegan, and Both) with portion-aware logic, dynamic substitutions, scoring, and UI enhancements.

---

## 🚀 Major Upgrade: "Search By Ingredients 2.0"

**What’s new in 2.0?**
- **Keto/Vegan search filters:** Find recipes by #keto, #vegan, or both.
- **Optional #diet:threshold search**: Search with diet compliance thresholds (e.g., #keto:0.8).
- **Visual diet badges**: Results show keto/vegan score badges. 100% scores get a strict vegan/keto badge.
- **Recipe transformation:** Instantly convert non-keto/non-vegan recipes to fully compliant ones with substitution logic.
- **Ingredient analysis UI:** New element visualizing per-ingredient compliance and suggesting improvements.
- **Portion-aware diet logic:** Tiny amounts of “forbidden” ingredients allowed if they don’t practically affect diet compliance.
- **Multi-diet intersection:** Find and create recipes that satisfy several dietary restrictions at once.
- **Dynamic substitution system:** Combines and enhances substitutions for multiple diets (e.g., “unsweetened almond milk” for both keto and vegan).
- **Real-time diet scoring:** Percentage-based compliance, badge gradients for partial matches.
- **API and CLI support:** Powerful endpoints and batch tools for search, analyze, and export.

---

## 📋 Task Requirements vs My Solution

### What Was Asked:
- Implement `is_ingredient_keto()`
- Implement `is_ingredient_vegan()`

### What I Delivered:
- Full, unified classification logic for both diets with USDA integration
- Portion-aware net carbs analysis for real-world keto compliance
- Multi-diet logic, scoring, and UI
- Instant recipe transformation and dynamic substitutions
- Search, analysis, and export tools

---

## 🧠 The Solution: Portion-Aware, Multi-Diet, Transformative

### Core Innovations

#### 1. Portion-Aware Diet Classification

Traditional systems are binary. Now, we allow small amounts of “forbidden” ingredients if their actual per-serving impact is negligible.  
Example: 1/4 tsp vanilla in a cake is now accepted for keto.

#### 2. Multi-Diet Compliance and Intersection

Find, score, and transform recipes for any combination of diets. Dynamic substitutions are computed, not hardcoded.

#### 3. Diet Threshold Filters

Use queries like `#keto:0.8` or `#both:0.7` to filter for partial compliance.

#### 4. Visual Diet Badges

Recipes display color-coded badges and compliance percentages for each selected diet. 100% compliance gets a strict badge.

---

## 🏗️ Architecture & Module Structure

```
utils/
├── classifiers.py      # Core classification logic
├── constants.py        # Diet lists & substitutions
├── portions.py         # Net carbs logic
├── scores.py           # Scoring
├── substitutions.py    # Dynamic replacements
├── query_flags.py      # Search syntax
└── usda.py             # Nutrition data
```

- **Unified Classifier:** Shared logic, diet-specific only when needed.
- **No Redundant Data:** “Both” substitutions are computed dynamically.
- **Layered Matching:** Regex, normalization, USDA lookup, and intelligent token analysis.
- **Percentage Scoring:** Shows how close a recipe is to each diet.

---

## ✨ Major Features

### Diet-Aware Search & Filtering

- `"chicken #keto"` — Keto chicken recipes
- `"pasta #vegan:0.8"` — Pasta ≥80% vegan
- `"salad #both"` — Both keto & vegan
- `"soup #keto:0.9 #vegan:0.7"` — Multiple thresholds

### Visual Diet Indicators

- Badges and pills for each recipe
- 100% = strict badge, partials show percentage
- UI toggles diet info based on search

### Dynamic Substitution System

- Single-diet and multi-diet substitutions
- Example:
  - Flour: almond/coconut flour for both
  - Milk: unsweetened almond milk for both
  - Eggs: flax eggs (with “use sparingly” for keto)

### Recipe Analysis Interface

- Select diet mode (keto, vegan, both)
- Input/parse ingredients with quantities
- Real-time scoring and suggestions
- Export fully converted recipes

### Transformative Recipe Conversion

Convert any recipe to keto, vegan, or both:
- Input: ["2 cups flour", "1 cup sugar", "3 eggs", "1/2 cup butter"]
- Output: ["2 cups almond flour", "1 cup erythritol", "3 flax eggs", "1/2 cup coconut oil"]
- Result: 100% compliant

---

## ⚡ Performance

- ~460 recipes/second
- Fast regex and USDA caching
- Minimal overhead for dynamic substitutions
- Batch indexing for scoring

---

## 📚 API Endpoints

1. `GET /` — Main UI
2. `GET /search` — Recipe search with diet filters
3. `GET /select2` — Ingredient autocomplete
4. `GET /substitutions` — Substitution engine
5. `POST /modify-recipe` — Apply substitutions
6. `GET /check-compliance` — Compliance scoring
7. `GET /export-modified` — Export recipes
8. `POST /convert-recipes` — Batch conversion

---

## 💡 Usage Examples

### Find Keto-Vegan Recipes
```
Search: "lunch #both"
→ 100% compliant recipes
```

### Convert a Recipe
```
Original: ["2 cups flour", "1 cup sugar", "3 eggs", "1/2 cup butter"]
Converted: 100% keto-vegan compliant
```

### Partial Compliance Search
```
Search: "#both:0.7"
→ Results with at least 70% compliance for both diets
```

---

## 🔮 Future Improvements

- Support for more diet combinations (paleo, gluten-free, custom)
- Weighted compliance (main vs. minor ingredients)
- Smart quantity adjustments for substitutions
- Community ratings for substitution success

---

## Summary

Search By Ingredients 2.0 transforms simple diet classification into a comprehensive, practical tool.  
- **Quantity matters**: Portion-aware logic for real-world compliance.
- **Diets intersect intelligently**: Dynamic, maintainable substitution logic.
- **Context enhances compliance**: “Use sparingly” and “unsweetened” modifiers bridge gaps.
- **UI follows intent**: Visual cues reduce cognitive load.

This system empowers users—especially those with multiple dietary needs—to cook and eat with confidence and creativity.

---

**Contact:**  
🔗 [Linkedin](https://www.linkedin.com/in/guyvitelson/)  
🐙 [GitHub](https://github.com/v1t3ls0n)  
✉️ [Mail](mailto:guyvitelson@gmail.com)

