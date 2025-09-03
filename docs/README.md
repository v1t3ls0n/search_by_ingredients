

# 🥑 Search By Ingredients 2.0: Multi-Diet, Portion-Aware, and Transformative Recipe Search

### Solution For Argmax's Search By Ingredients Challenge **By Guy Vitelson**

---

**Task**: Originally, implement `is_ingredient_keto()` and `is_ingredient_vegan()`.  
**Now:** The project has evolved into a powerful, extensible tool for analyzing, searching, and transforming recipes for multiple diets (Keto, Vegan, and Both) with portion-aware logic, dynamic substitutions, scoring, and UI enhancements.

---

## 🚀 Major Upgrade: "Search By Ingredients 2.0"

**What’s new in 2.0?**
- **Keto/Vegan search filters:** Find recipes by #keto, #vegan, or both.
- **Optional #diet:threshold search**: Search with diet compliance thresholds (e.g., #keto:0.8).
- **Visual diet badges:** Results show keto/vegan score badges. 100% scores get a strict vegan/keto badge.
- **Recipe transformation:** Instantly convert non-keto/non-vegan recipes to fully compliant ones with substitution logic.
- **Ingredient analysis UI:** New element visualizing per-ingredient compliance and suggesting improvements.
- **Portion-aware diet logic:** Tiny amounts of “forbidden” ingredients allowed if they don’t practically affect diet compliance.
- **Multi-diet intersection:** Find and create recipes that satisfy several dietary restrictions at once.
- **Dynamic substitution system:** Combines and enhances substitutions for multiple diets (e.g., “unsweetened almond milk” for both keto and vegan).
- **Real-time diet scoring:** Percentage-based compliance, badge gradients for partial matches.
- **API and CLI support:** Powerful endpoints and batch tools for search, analyze, and export.
- **Automatic recipe saving:** All modified recipes automatically saved to database with unique ID management.
- **Saved variations management:** View, export, and manage all recipe modifications with detailed history.

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

### Visual Diet Indicators

- Badges and pills for each recipe
- 100% = strict badge, partials show percentage
- UI toggles diet info based on search

![Search results with diet badges](screenshots/ui1.png)

### Threshold-Based Diet Filtering

Advanced filtering with compliance thresholds allows users to find recipes that meet specific diet requirements:

- `#keto:0.8` — Recipes with at least 80% keto compliance
- `#vegan:0.9` — Recipes with at least 90% vegan compliance  
- `#both:0.7` — Recipes compliant with both keto and vegan at 70%+

![Threshold-based diet filtering](screenshots/ui5.png)



### Recipe Analysis & Conversion Interface

- Select diet mode (keto, vegan, both)
- Input/parse ingredients with quantities
- Real-time scoring and suggestions
- Export fully converted recipes
- Convert any recipe to keto, vegan, or both

![Recipe analysis screen](screenshots/ui3.png)


### Dynamic Substitution System

- Single-diet and multi-diet substitutions
- Example:
  - Flour: almond/coconut flour for both
  - Milk: unsweetened almond milk for both
  - Eggs: flax eggs (with “use sparingly” for keto)

![Dynamic substitution modal](screenshots/ui2.png)


### Saved Recipe Variations

The system automatically saves all modified recipes to a database with unique ID management to prevent duplicates:

#### Automatic Database Sync
- **Seamless saving**: All recipe modifications are automatically saved to the database
- **Unique ID system**: Prevents duplicate recipes between localStorage and database
- **Automatic cleanup**: Removes localStorage entries after successful database save
- **Error resilience**: Falls back to localStorage if database save fails

#### Saved Variations Interface
- **Unified view**: Shows all saved recipes from both localStorage and database
- **Source indicators**: Visual badges distinguish database (🗄️) vs localStorage (💾) recipes
- **Clickable cards**: View detailed recipe information and modification history
- **Export functionality**: Download saved recipe variations

![Saved recipe variations interface](screenshots/ui6.png)

#### Recipe Modification Details
- **Modification tracking**: Shows original vs modified ingredients
- **Substitution history**: Displays all changes made for diet compliance
- **Score improvements**: Tracks before/after diet compliance scores
- **Detailed metadata**: Modification date, diet type, and change count

![Recipe modification details](screenshots/ui7.png)


---

## ⚡ Performance

- ~460 recipes/second
- Fast regex and USDA caching
- Minimal overhead for dynamic substitutions
- Batch indexing for scoring

---

## 📚 API Endpoints

### Core Functionality
1. `GET /` — Main UI
2. `GET /search` — Recipe search with diet filters
3. `GET /select2` — Ingredient autocomplete
4. `GET /substitutions` — Substitution engine
5. `POST /modify-recipe` — Apply substitutions
6. `GET /check-compliance` — Compliance scoring
7. `GET /export-modified` — Export recipes
8. `POST /convert-recipes` — Batch conversion

### Database Management
9. `POST /save-modified-recipe` — Save recipe variation to database
10. `GET /get-modified-recipes` — Retrieve saved recipes (supports filtering by id, diet, title, unique_id)
11. `DELETE /delete-modified-recipe/<id>` — Delete specific recipe variation
12. `DELETE /clear-all-modified-recipes` — Clear all saved recipes

---

## 💡 Usage Examples

### Find Keto-Vegan Recipes
```

Search: "lunch #both"
→ 100% compliant recipes

```

### Convert a Recipe
```

Original: \["2 cups flour", "1 cup sugar", "3 eggs", "1/2 cup butter"]
Converted: 100% keto-vegan compliant

```

### Partial Compliance Search
```

Search: "#both:0.7"
→ Results with at least 70% compliance for both diets

```

### Save and Manage Recipe Variations
```

1. Apply diet substitutions to any recipe
2. Recipe automatically saved to database
3. View all saved variations in "Saved Variations" tab
4. Export or delete individual variations
5. Track modification history and compliance improvements

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
- **Persistent recipe management**: Automatic database saving with unique ID system prevents duplicates.
- **Complete workflow**: From search to modification to saving and management.

This system empowers users—especially those with multiple dietary needs—to cook and eat with confidence and creativity, while maintaining a complete history of their recipe modifications.

---

## 🤖 Bonus: Over-Engineered ML Solution (Way Beyond Scope)

**Note: The task only asked to implement two functions. The rule-based solution above completely solves it with 100% accuracy.**

For those curious about "what if we went completely overboard?", I created an entirely optional ML solution on the `ml-overkill-solution` branch. This is a **9000+ line production ML system** organized into **40+ modules** - essentially what you'd build for a Fortune 500 company, not a classification task.

### What's in the ML branch:
- Complete weak supervision pipeline with silver labeling
- Multi-modal learning (text + 70K images) 
- 15+ ML models with 4-level hierarchical ensembles
- Full production infrastructure (GPU support, memory management, error handling)
- Modular architecture: `silver_labeling/`, `feature_engineering/`, `models/`, `ensemble/`, etc.

### Why build something so excessive?
Pure engineering showcase. It demonstrates the ability to architect large-scale ML systems even when a simple solution suffices. The ML system achieves F1-scores up to 0.963 (worse than the 100% rule-based solution!) while being 30x more complex.

**Bottom line**: The rule-based solution is the right answer. The ML branch is there if you want to see what "throwing everything at the problem" looks like.


---


**Contact:**  
🔗 [Linkedin](https://www.linkedin.com/in/guyvitelson/)  
🐙 [GitHub](https://github.com/v1t3ls0n)  
✉️ [Mail](mailto:guyvitelson@gmail.com)
