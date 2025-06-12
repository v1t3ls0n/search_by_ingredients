## 🥑 Solution For Argmax's Search By Ingredients Challenge 
**$$ \arg\max_{r \in \mathcal{G}} \ \mathrm{score}(r) \quad \text{subject to } \I(r) = \text{@v1t3ls0n} $$**

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

## 🧪 Model Evaluation Results (Sorted by Task → F1 Score)
**Image models trained on ~70K images, all models evaluated on the gold set (ground_truth data)**
### 🥑 Keto Models

| 🧠 Model         | 🎯 Task | ✅ Accuracy | 🎯 Precision | 🔁 Recall | 🏆 F1-Score | ⏱️ Time (s) |
|------------------|--------|-------------|--------------|-----------|-------------|-------------|
| 🤖 Softmax_TEXT  | keto   | 0.970       | 0.951        | 0.975     | **0.963**   | 3.6         |
| 🧠 Ridge_TEXT    | keto   | 0.970       | 0.951        | 0.975     | **0.963**   | 12.4        |
| ⚔️ PA_TEXT       | keto   | 0.970       | 0.951        | 0.975     | **0.963**   | 4.2         |
| 🧪 SGD_TEXT      | keto   | 0.970       | 0.951        | 0.975     | **0.963**   | 0.6         |
| 🌲 RF_IMAGE      | keto   | 0.942       | 0.929        | 0.963     | 0.945       | 111.5       |
| 🧬 Softmax_BOTH  | keto   | 0.942       | 0.929        | 0.963     | 0.945       | 85.8        |
| 🌟 LGBM_BOTH     | keto   | 0.942       | 0.929        | 0.963     | 0.945       | 109.1       |
| 🧠 TxtImg        | keto   | 0.940       | 0.930        | 0.960     | 0.950       | –           |
| 🦠 NB_TEXT       | keto   | 0.960       | 0.950        | 0.950     | 0.950       | 0.3         |
| ⚡ LGBM_IMAGE     | keto   | 0.904       | 0.923        | 0.889     | 0.906       | 87.7        |
| 🐍 NB_BOTH       | keto   | 0.865       | 0.955        | 0.778     | 0.857       | 0.2         |
| 🧠 MLP_IMAGE     | keto   | 0.750       | 0.938        | 0.556     | 0.698       | 119.8       |

---

### 🌱 Vegan Models

| 🧠 Model         | 🎯 Task | ✅ Accuracy | 🎯 Precision | 🔁 Recall | 🏆 F1-Score | ⏱️ Time (s) |
|------------------|--------|-------------|--------------|-----------|-------------|-------------|
| 🤖 Softmax_TEXT  | vegan  | 0.980       | 0.975        | 0.975     | **0.975**   | 1.6         |
| ⚔️ PA_TEXT       | vegan  | 0.980       | 0.975        | 0.975     | **0.975**   | 4.5         |
| 🧠 Ridge_BOTH    | vegan  | 0.981       | 1.000        | 0.964     | **0.982**   | 462.9       |
| ⚔️ PA_BOTH       | vegan  | 0.981       | 1.000        | 0.964     | **0.982**   | 25.9        |
| 🌲 RF_IMAGE      | vegan  | 1.000       | 1.000        | 1.000     | **1.000**   | 77.6        |
| 🧬 Softmax_BOTH  | vegan  | 1.000       | 1.000        | 1.000     | **1.000**   | 73.2        |
| 🌲 RF_BOTH       | vegan  | 1.000       | 1.000        | 1.000     | **1.000**   | 29.0        |
| 🌟 LGBM_BOTH     | vegan  | 1.000       | 1.000        | 1.000     | **1.000**   | 127.9       |
| 🧠 TxtImg        | vegan  | 1.000       | 1.000        | 1.000     | **1.000**   | –           |
| 🧠 MLP_BOTH      | vegan  | 0.962       | 1.000        | 0.929     | 0.963       | 2515.7      |
| ⚡ LGBM_IMAGE     | vegan  | 0.962       | 1.000        | 0.929     | 0.963       | 54.3        |
| 🧠 Ridge_TEXT    | vegan  | 0.970       | 0.974        | 0.950     | 0.962       | 12.2        |
| 🧪 SGD_TEXT      | vegan  | 0.970       | 0.974        | 0.950     | 0.962       | 0.6         |
| 🦠 NB_TEXT       | vegan  | 0.960       | 0.974        | 0.925     | 0.949       | 0.1         |
| 🐍 NB_BOTH       | vegan  | 0.788       | 1.000        | 0.607     | 0.756       | 0.2         |
| 🧠 MLP_IMAGE     | vegan  | 0.596       | 1.000        | 0.250     | 0.400       | 100.6       |


## 🧠 Key Takeaways from Results

These results demonstrate **exceptionally strong performance**, especially considering:

* The **entire training pipeline is weakly supervised** (no ground-truth labels were available during training).
* The **evaluation set is small** (100 samples, of which only 66 have images), meaning scores can fluctuate due to variance.
* Image-based models were trained on **up to 70,000 images**, yielding significant gains over prior experiments with 700 or 7,000 samples — proving the benefit of scale in weak visual learning.
* Text-only models continue to dominate due to high signal in ingredient names, with F1-scores reaching **0.96+**.
* **Vegan models reached perfect classification (F1 = 1.0)** when using image+text ensembles — a rare result, highlighting alignment between rules, training data, and real-world gold labels.

⚠️ **Important Note:**
All the results above were achieved **before** enabling **dynamic per-row ensemble weight optimization**. The current ensemble logic uses static weightings, yet still delivers top-tier metrics. This strongly suggests that adding dynamic weights based on image/text availability could yield even higher confidence and class-specific reliability.

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

## 🔖 Artifacts Directory

Trained models and vectorizer are persisted here via the host-mounted  
`./artifacts` folder (inside the container at `/app/artifacts`).

After running the training pipeline, you’ll find:

```

artifacts/
├── vectorizer.pkl
└── models.pkl

````

Ensure your `docker-compose.yml` maps it:

```yaml
services:
  web:
    …
    volumes:
      - ./artifacts:/app/artifacts
      - recipe-data:/usr/src/data
      - ./web/src:/app/web
````


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

| Feature                                       | Status | Notes                                                                                                                  |
| --------------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------- |
| 💡 **Dynamic Ensemble Weighting (per-row)**   | 🔜     | Logic is implemented, pending evaluation. Will allow better fusion of text/image-only rows using adaptive soft voting. |
| 💡 Net-carb detection (fiber, sugar alcohol)  | ❌      | Enhance numeric validation by subtracting fiber/sugar alcohol from carb totals.                                        |
| 💡 Active learning to resolve USDA ambiguity  | ❌      | Use model uncertainty to suggest human verification.                                                                   |
| 💡 UI for human feedback verification loop    | ❌      | Let users refine silver labels over time.                                                                              |
| 💡 Auto-generated model cards and ONNX export | ❌      | For transparency and deployment readiness.                                                                             |

> ✅ "Dynamic voting is now supported by the ensemble system but **has not yet been used to generate the above results**. An updated run with this optimization is scheduled next."

---

## 📚 References

* USDA FoodData Central (2023)
* Chawla et al., *SMOTE*, JAI 2002
* He et al., *ResNet*, CVPR 2016
* Salton & Buckley, *TF-IDF*, IR 1988

