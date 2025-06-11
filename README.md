# 🥑 Search By Ingredients Challenge Solution

$$
\hat{s} = \arg\max_{s \in \mathcal{S}} \ \mathrm{score}(s) \quad \text{subject to } I(s) = \text{v1t3ls0n}
$$

**By [Guy Vitelson](https://www.linkedin.com/in/guyvitelson/)** · 🐙 [@v1t3ls0n](https://github.com/v1t3ls0n) · 📧 [guyvitelson@gmail.com](mailto:guyvitelson@gmail.com)

---

## 🧭 Project Overview

This project classifies recipes as either:

- **Keto-Friendly** (≤10g net carbs per 100g)
- **Vegan** (strictly plant-based)

We assume **no labeled data is available**, and solve the task using weak supervision, rule-based logic, and machine learning. We go beyond requirements by integrating:

- ✅ USDA FoodData Central for numeric nutritional validation  
- ✅ Six-stage silver labeling with regex + fallback rules  
- ✅ ML model training over sparse text/image features  
- ✅ Ensemble optimization and dynamic voting  
- ✅ CLI + Docker + logging + caching for robust execution  

---

## ⚙️ Pipeline Architecture

```

┌─────────────────────┐
│ Raw Recipe Dataset  │ (No labels)
└────┬────────────────┘
▼
Silver Label Generator (rules + USDA)
▼
Text Vectorizer (TF-IDF)
│
├──▶ Optional: Image Embeddings (ResNet-50)
▼
Model Training (Text / Image / Hybrid)
▼
Top-N Ensemble + Rule Verification
▼
Export: Metrics, Plots, Artefacts

````

---

## 🔩 Silver Labeling Engine

Our system produces training labels (silver labels) via a **6-stage heuristic**:

| Stage | Description                          | Example                                |
|-------|--------------------------------------|----------------------------------------|
| 1.    | Whitelist (early positive)           | "almond flour" is always keto          |
| 2.    | USDA numeric check (carbs ≤ 10g)     | "jackfruit" fails: 23g carbs           |
| 3.    | Regex blacklist                      | Rejects "sugar", "rice"                |
| 4.    | Token combination matching           | "kidney beans" → non-keto              |
| 5.    | ML soft classification (probability) | Learns semantic context                |
| 6.    | Hard override with rules             | Final decision ensures dietary safety  |

> Fallbacks like token simplification (`egg whites` → `egg`) + USDA fuzzy match improve recall.

---

## 🧠 ML Models and Ensemble

Text classifiers:

- Logistic Regression (Softmax)
- Naive Bayes
- RidgeClassifier
- SGDClassifier
- Passive-Aggressive

Optional image support:

- Download photos from recipe URLs
- Run ResNet-50 embedding extraction
- Merge with text features for multimodal classification

We tune, score, and rank each model across metrics:

- Accuracy
- Precision, Recall
- F1 Score
- ROC AUC, PR AUC

Then, a dynamic ensemble (`top_n`) builds the optimal blend of models for each task, using weighted soft voting and rule verification.

---

## 🎯 Evaluation Results

**Text-only ensemble (best):**

| Task  | Accuracy | Precision | Recall | F1 Score | ROC AUC | PR AUC |
|-------|----------|-----------|--------|----------|---------|--------|
| Keto  | 0.97     | 0.95      | 0.97   | **0.96** | 0.99    | 0.98   |
| Vegan | 0.98     | 0.97      | 0.97   | **0.97** | 0.99    | 0.98   |

**Multimodal (text + image):**

| Task  | F1 Score Range |
|-------|----------------|
| Keto  | 0.88 – 0.92     |
| Vegan | Lower (image coverage sparse) |

---

## 🖥️ CLI Interface

Train, test, classify, or run gold-eval in one line:

```bash
# Train and evaluate on silver + gold
python diet_classifiers.py --train --mode both

# Evaluate only against gold set
python diet_classifiers.py --ground_truth data/gold_sample.csv

# Classify custom ingredient list
python diet_classifiers.py --ingredients "almond flour, erythritol, egg whites"
````

Or via Docker:

```bash
./run_pipeline.sh       # End-to-end build + run
```

---

## 💡 Design Principles

* ✅ **No training? Still usable.** Rule-based fallback ensures predictions even if ML fails.
* ✅ **Weak labels? Strong pipeline.** Ensemble softens rule noise with learned patterns.
* ✅ **Cachable & restart-safe.** Embeddings, models, predictions all memoized with backup.
* ✅ **Containerized deployment.** Three Docker services (CLI, notebook, web/API).
* ✅ **Resilient against partial data.** Works with missing ingredients, broken photos, or sparse recipes.

---

## 🧪 Robustness & Recovery

* ML, vectorizer, and image embeddings are cached with `.npy` or `.pkl` backups
* If cache is corrupted or missing → auto-regenerates from source
* Rule-based logic ensures fallback always available
* Embedding pipeline supports restart-safe env guard

---

## 📦 Directory Layout

```
.
├── nb/                            # 📓 Jupyter/CLI container
│   ├── src/
│   │   ├── diet_classifiers.py    # Diet classification logic (notebook/CLI)
│   │   ├── hybrid_classifier.py   # Optional hybrid model
│   │   └── task.ipynb             # Dev notebook
│   ├── Dockerfile
│   └── requirements.txt

├── web/                           # 🌐 Web/API container
│   ├── src/
│   │   ├── templates/
│   │   │   └── index.html
│   │   ├── app.py                 # Flask entrypoint
│   │   ├── diet_classifiers.py    # Main pipeline logic
│   │   ├── index_data.py          # Optional search support
│   │   └── init.sh                # CLI entry + startup
│   ├── Dockerfile
│   └── requirements.txt

├── data/                          # 📊 Raw and generated files
│   ├── usda/                      # USDA CSVs
│   └── gold_sample.csv            # 100-row hand-labeled test set

├── docker-compose.yml             # Multi-container runner
├── run_pipeline.sh                # One-click script
├── .gitignore
└── README.md
```

---

## ✅ Feature Matrix

| Feature                           | Status | Notes                              |
| --------------------------------- | ------ | ---------------------------------- |
| Weak supervision via rules        | ✅      | Regex, USDA carbs, token rules     |
| Image + Text dual domain          | ✅      | Optional multimodal with ResNet    |
| Caching + restarts                | ✅      | Backups + smart reuse              |
| Evaluation plots + exports        | ✅      | ROC, PR, Confusion Matrix, CSV     |
| Ensemble optimization             | ✅      | Top-N ranking across 6 metrics     |
| Full container setup              | ✅      | `docker-compose` 3-container setup |
| CLI usage                         | ✅      | Command-line friendly              |
| API readiness                     | ✅      | Flask entrypoint included          |
| Logging and debugging             | ✅      | Color logs + progress tracking     |
| Integration of external knowledge | ✅      | USDA nutrition database            |

---

## 🛠️ Future Improvements

* 💡 Net-carb detection (subtract fiber, sugar alcohol)
* 💡 Active learning to resolve USDA ambiguity
* 💡 UI for human feedback verification loop
* 💡 Auto-generated model cards and ONNX export

---

## 📚 References

* USDA FoodData Central (2023)
* Chawla et al., *SMOTE*, JAI 2002
* He et al., *ResNet*, CVPR 2016
* Salton & Buckley, *TF-IDF*, IR 1988

