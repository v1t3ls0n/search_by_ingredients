

# 🥑 Keto & Vegan Diet Classifier — Full Submission Report

> **Author:** [Guy Vitelson](https://www.linkedin.com/in/guyvitelson/) · 🐙 [@v1t3ls0n](https://github.com/v1t3ls0n) · 📧 [guyvitelson@gmail.com](mailto:guyvitelson@gmail.com)

**Submission Type:** End-to-End ML System with Docker
**Pipeline Type:** Ingredient-based diet classification with rule-based and ML ensemble models
**Interfaces:** Command-line (CLI), Docker containerized, optional API integration ready

---

Let me know if you want a signature/footer version too (for a PDF or notebook format), or if you'd like the author details repeated at the bottom as well.


## 🧭 Project Overview

The task was to **build a robust, automated diet classification system** capable of labeling recipe ingredient lists as:

* **Keto-friendly** (low-carb)
* **Vegan** (no animal products)

**Constraints:**

* ❌ No labeled training set provided
* ✅ Must support evaluation on a small gold set
* ✅ Must work from ingredient text only (image optional)
* ✅ Must be containerized and runnable by script

We went significantly beyond these requirements:

* Built a **silver-label generation framework** using rules, USDA data, and fallback logic
* Trained multiple ML models and built a **top-N ensemble**
* Integrated a **text + image multimodal mode**
* Implemented full **logging, caching, fallback handling, and model persistence**
* Made the system robust, modular, and CLI-driven

---

## 🔩 Pipeline Summary

### 1. **Silver Labeling**

Since no ground-truth was provided, we created our own **weak labels** using:

* Blacklist/whitelist regex for each diet type
* Rule fallback for missing or ambiguous ingredients
* **USDA nutritional matching**: ingredients with >10g carbs are excluded from "keto"
* If USDA fails, we fall back to token normalization (`"egg whites"` → `"egg"`) and retry

This gave us strong enough supervision to train classifiers.

### 2. **Model Training**

We train multiple text-based classifiers:

* Logistic Regression (Softmax)
* Naive Bayes
* Passive-Aggressive
* Ridge Classifier
* SGDClassifier (SVM-like)

Plus optional **image-based embeddings** using ResNet if images exist.

We score all models using:

* F1 Score
* Precision, Recall, Accuracy
* ROC AUC, PR AUC

Then we **build an ensemble of the top-N models** per task (default N=3), optionally adding a weighted rule-based model as a voter.

### 3. **Evaluation**

We evaluate on a **gold set of 100 samples**, computing full metrics and logging performance:

#### Text-Only Model

| Task  | Accuracy | Precision | Recall | F1 Score | ROC AUC | PR AUC |
| ----- | -------- | --------- | ------ | -------- | ------- | ------ |
| Keto  | 0.97     | 0.95      | 0.97   | **0.96** | 0.99    | 0.98   |
| Vegan | 0.98     | 0.97      | 0.97   | **0.97** | 0.99    | 0.98   |

#### Multimodal (Text + Image)

* **Keto** F1: `0.88 – 0.92`
* **Vegan**: Lower, due to sparse image coverage

### 4. **CLI Interface**

Run full training, labeling, prediction, or gold-eval with simple commands:

```bash
python diet_classifiers.py --train --mode both
python diet_classifiers.py --ground_truth /path/to/gold.csv
```

---

## 💡 Design Philosophy

This system was built with **clarity and resilience** in mind. Every decision was pragmatic:

* ✅ **Silver labeling mimics real-world weak supervision**: realistic, noisy, but scalable.
* ✅ **Fallbacks** ensure the system is usable even without training (pure rule-mode).
* ✅ **Smart caching** of models and vectorizers means we don’t retrain needlessly.
* ✅ **Logging-first** design ensures easy debugging.
* ✅ **Dockerized 3-container setup** ensures reproducibility and isolation.

---

## 🧪 Known Limitations

* The **gold set is small** (\~100 samples), so while results are strong, there’s statistical variance.
* **Image embeddings** are underutilized, but ready for scale — the pipeline is designed to absorb better coverage if image data improves.
* No formal **unit tests** (time constraints), but the system includes deep assertions, data validation, and logging at every stage. Unit tests would be next.

---

## 🛠️ File Structure (Dockerized)

```bash
.
├── web/
│   ├── diet_classifiers.py         # Main pipeline script
│   ├── app.py                      # CLI & API handler
│   ├── index_data.py               # Indexing helpers
│   └── utils/                      # Shared tools
├── data/
│   ├── gold.csv                    # Gold-labeled recipes
│   ├── usda_cache.json             # Cached nutritional info
│   └── vectorizer.pkl / models.pkl# Saved ML pipeline
├── Dockerfile
├── docker-compose.yml
├── run_pipeline.sh                # One-click execution
```

---

## ✅ Features Implemented vs Task Requirements

| Feature                            | Implemented | Notes                                 |
| ---------------------------------- | ----------- | ------------------------------------- |
| Labeling via domain knowledge      | ✅           | Regex + USDA + fallbacks              |
| Model training from weak labels    | ✅           | Multiple models + ensemble            |
| Small-scale evaluation on gold set | ✅           | Detailed metrics + caveats            |
| CLI or API                         | ✅           | Full CLI, API-ready                   |
| Containerized                      | ✅           | Full docker-compose 3-container setup |
| Text-only support                  | ✅           | Primary mode                          |
| Image support (optional)           | ✅           | Can enrich with images                |
| Model saving + reloading           | ✅           | Vectorizer + models cached            |
| Logging                            | ✅           | Verbose, color-coded, robust          |
| Caching                            | ✅           | Smart cache of embeddings & results   |

---

## 🧠 Final Thoughts

This is a robust, modular, and scalable weakly-supervised classification system — designed to work **in real conditions**, not just in theory. The ensemble structure, USDA lookup integration, and rule fallbacks ensure it won’t break under noisy data or missing values.

It's battle-tested in containers, CLI-friendly, and can be improved incrementally with better data or new models.

