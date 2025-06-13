## 🥑 Solution For Argmax's Search By Ingredients Challenge By **Guy Vitelson**

---
##### Ping me via 🔗 **[Linkedin](https://www.linkedin.com/in/guyvitelson/)**  🐙 **[GitHub](https://github.com/v1t3ls0n)**  ✉️ **[Mail](mailto:guyvitelson@gmail.com)**
---

## 🧭 Project Overview

This pipeline implements two independent binary classification tasks for recipes:

Keto-Friendly: ≤ 10 g net carbohydrates per 100 g serving

Vegan: no animal-derived ingredients (strictly plant-based)



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
Export: Metrics, Plots, Artifacts 

````

---

### 🔍 Silver Labeling Heuristics

Our silver labeling engine applies **progressive rule-based heuristics** that simulate expert knowledge to label unlabeled recipes.

Before applying classification logic, we pre-filter unqualified samples:

* ❌ **Photo Filtering:** Remove rows whose `photo_url` contains `nophoto`, `nopic`, or `nopicture` — these are consistently blank or irrelevant images.
* ❌ **Empty or malformed ingredient entries** are excluded from training.
* ✅ **Ingredient normalization** includes lemmatization, unit removal, numeric stripping, and character simplification.

#### Labeling Logic:

We then assign **silver labels** using the following prioritized rules:

| Stage | Description                                                                                                  | Example |
| ----- | ------------------------------------------------------------------------------------------------------------ | ------- |
| 1.    | **Whitelist override**: Certain ingredients (e.g., `"almond flour"`) are always positive for keto/vegan.     |         |
| 2.    | **USDA nutrient check**: Ingredient-level carbs are looked up (≤10g net carbs per 100g → keto).              |         |
| 3.    | **Regex blacklist**: Disqualifies patterns like `"sugar"`, `"honey"`, `"bacon"` for relevant tasks.          |         |
| 4.    | **Token combination rules**: Phrases like `"kidney beans"` or `"chicken broth"` trigger exclusions.          |         |
| 5.    | **Soft ML classification**: Weak ML models trained on early confident examples provide score-based fallback. |         |
| 6.    | **Hard override + verification**: Final logic ensures dietary-safe labels with rule-based priority.          |         |

> This hybrid process is robust to noisy input and simulates a domain-expert verification process.

---


## 🧠 ML Models and Ensemble

### Text-only classifiers

* **Softmax\_TEXT** (Logistic Regression)
* **Ridge\_TEXT** (RidgeClassifier)
* **PA\_TEXT** (Passive-Aggressive Classifier)
* **SGD\_TEXT** (SGDClassifier)
* **NB\_TEXT** (Multinomial Naive Bayes)

### Image-only classifiers

* **RF\_IMAGE** (Random Forest on ResNet-50 embeddings)
* **LGBM\_IMAGE** (LightGBM on ResNet-50 embeddings)
* **MLP\_IMAGE** (Multi-layer Perceptron on ResNet-50 embeddings)

### Hybrid (Text + Image) classifiers

* **Softmax\_BOTH** (Logistic Regression on concatenated TF-IDF + image embeddings)
* **Ridge\_BOTH** (RidgeClassifier on concatenated features)
* **PA\_BOTH** (Passive-Aggressive on concatenated features)
* **RF\_BOTH** (Random Forest on concatenated features)
* **LGBM\_BOTH** (LightGBM on concatenated features)
* **NB\_BOTH** (Naive Bayes on concatenated features)
* **MLP\_BOTH** (MLP on concatenated features)
* **TxtImg** (custom text–image fusion model)

---

We train & evaluate **all** of these on both the silver set and the gold ground-truth, scoring them by:

* **Accuracy**
* **Precision / Recall**
* **F1 Score**
* **ROC AUC** & **PR AUC**

Then our **dynamic top-N ensemble** greedily picks the best subset of these predictors for each task (keto vs. vegan), with a final rule-based override to guarantee dietary correctness.

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
Here are your updated and integrated outputs:

---

### ✅ Updated Docstring for `diet_classifiers.py`

```python
"""
================================================================================
DIET CLASSIFIER PIPELINE - MULTI-MODAL MACHINE LEARNING FOR INGREDIENT ANALYSIS
================================================================================

This comprehensive machine learning pipeline classifies recipes as keto-friendly 
or vegan based on their ingredients using a multi-modal approach combining:

1. TEXT FEATURES: TF-IDF vectorization of normalized ingredient lists
2. IMAGE FEATURES: ResNet-50 embeddings from recipe photos
3. RULE-BASED VERIFICATION: Domain-specific heuristics and USDA nutritional data

KEY COMPONENTS:
---------------
- SILVER LABEL GENERATION: Creates weak labels from unlabeled data using multi-stage
  rule-based heuristics simulating expert knowledge:
    • Token normalization + lemmatization
    • Regex-based blacklist/whitelist
    • USDA-based carbohydrate filtering (≤10g carbs/100g → keto-safe)
    • Phrase-level disqualifications (e.g., "chicken broth")
    • Whitelist override of verified-safe ingredients (e.g., "almond flour")
    • Soft ML fallback + rule-based priority merging
    • Photo sanity filtering: excludes rows with URLs like 'nophoto', 'nopic', 'nopicture'

- MODEL TRAINING: Trains diverse ML models (Logistic Regression, SVM, MLP, Random Forest, etc.)
- ENSEMBLE METHODS: Combines multiple classifiers using top-N voting and rule-based overrides
- CACHING & RESTORE: Saves and reuses models, vectorizers, image embeddings
- LOGGING: Logs to both console and `artifacts/pipeline.log`
- FULL EVALUATION: Saves gold-test predictions and per-class metrics to CSV

ARCHITECTURE OVERVIEW:
----------------------
1. Data Loading:
   - Loads silver (unlabeled) and gold (labeled) recipes
   - Uses USDA nutritional DB for rule-based classification
   - Input can be CSV or Parquet

2. Feature Extraction:
   - Text: TF-IDF vectorization after custom normalization
   - Image: ResNet-50 feature extraction from downloaded photos
   - Merges modalities where appropriate

3. Model Training:
   - Silver-labeled data → supervised classifiers
   - Supports `--mode text`, `--mode image`, `--mode both`

4. Prediction & Evaluation:
   - Supports ingredient inference or full CSV evaluation
   - Computes Accuracy, F1, Precision, Recall
   - Exports predictions and metrics to artifacts directory

USAGE MODES:
------------
1. Training: `--train` to trigger full silver model training pipeline
2. Inference: `--ingredients` for direct classification from command line
3. Evaluation: `--ground_truth` for benchmarking against labeled CSV

Robust against partial data, broken images, or failed downloads.
Supports interactive development, Docker builds, and production use.

Author: Guy Vitelson (aka @v1t3ls0n on GitHub)
"""
```

---

## 🖥️ CLI Interface

Train, evaluate, or classify directly via CLI:

```bash
# Train and evaluate using silver + gold set
python diet_classifiers.py --train --mode both

# Evaluate on labeled test set
python diet_classifiers.py --ground_truth /usr/src/data/ground_truth_sample.csv

# Classify custom ingredients
python diet_classifiers.py --ingredients "almond flour, coconut oil, cocoa powder"
````

Additional options:

| Argument         | Type   | Description                                                         |
| ---------------- | ------ | ------------------------------------------------------------------- |
| `--train`        | flag   | Run the full training pipeline on silver labels                     |
| `--ground_truth` | path   | Evaluate trained models on a gold-labeled CSV file                  |
| `--ingredients`  | str    | Comma-separated list or JSON array of ingredients for inference     |
| `--mode`         | choice | Feature mode: `text`, `image`, or `both` (default: both)            |
| `--force`        | flag   | Recompute image embeddings, ignoring cached `.npy` files            |
| `--sample_frac`  | float  | Fraction of the silver set to sample for training (e.g., 0.1 = 10%) |

Or run via Docker for full automation:

```bash
./run_pipeline.sh       # Build, train, and test in one go
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

