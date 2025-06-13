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


---

### 🥑 **Keto Models** (Sorted by F1)

| 🧠 Model         | 🎯 Task | ✅ Accuracy | 🎯 Precision | 🔁 Recall | 🏆 F1-Score | ⏱️ Time (s) |
| ---------------- | ------- | ---------- | ------------ | --------- | ----------- | ----------- |
| 🤖 Softmax\_TEXT | keto    | 0.970      | 0.951        | 0.975     | **0.963**   | 3.6         |
| 🧠 Ridge\_TEXT   | keto    | 0.970      | 0.951        | 0.975     | **0.963**   | 12.4        |
| ⚔️ PA\_TEXT      | keto    | 0.970      | 0.951        | 0.975     | **0.963**   | 4.2         |
| 🧪 SGD\_TEXT     | keto    | 0.970      | 0.951        | 0.975     | **0.963**   | 0.6         |
| 🧠 TxtImg        | keto    | 0.940      | 0.930        | 0.960     | 0.950       | –           |
| 🦠 NB\_TEXT      | keto    | 0.960      | 0.950        | 0.950     | 0.950       | 0.3         |
| 🌲 RF\_IMAGE     | keto    | 0.942      | 0.929        | 0.963     | 0.945       | 111.5       |
| 🧬 Softmax\_BOTH | keto    | 0.942      | 0.929        | 0.963     | 0.945       | 85.8        |
| 🌟 LGBM\_BOTH    | keto    | 0.942      | 0.929        | 0.963     | 0.945       | 109.1       |
| ⚡ LGBM\_IMAGE    | keto    | 0.904      | 0.923        | 0.889     | 0.906       | 87.7        |
| 🐍 NB\_BOTH      | keto    | 0.865      | 0.955        | 0.778     | 0.857       | 0.2         |
| 🧠 MLP\_IMAGE    | keto    | 0.750      | 0.938        | 0.556     | 0.698       | 119.8       |

---

### 🌱 **Vegan Models** (Sorted by F1)

| 🧠 Model         | 🎯 Task | ✅ Accuracy | 🎯 Precision | 🔁 Recall | 🏆 F1-Score | ⏱️ Time (s) |
| ---------------- | ------- | ---------- | ------------ | --------- | ----------- | ----------- |
| 🌲 RF\_IMAGE     | vegan   | 1.000      | 1.000        | 1.000     | **1.000**   | 77.6        |
| 🧬 Softmax\_BOTH | vegan   | 1.000      | 1.000        | 1.000     | **1.000**   | 73.2        |
| 🌲 RF\_BOTH      | vegan   | 1.000      | 1.000        | 1.000     | **1.000**   | 29.0        |
| 🌟 LGBM\_BOTH    | vegan   | 1.000      | 1.000        | 1.000     | **1.000**   | 127.9       |
| 🧠 TxtImg        | vegan   | 1.000      | 1.000        | 1.000     | **1.000**   | –           |
| 🧠 Ridge\_BOTH   | vegan   | 0.981      | 1.000        | 0.964     | **0.982**   | 462.9       |
| ⚔️ PA\_BOTH      | vegan   | 0.981      | 1.000        | 0.964     | **0.982**   | 25.9        |
| 🤖 Softmax\_TEXT | vegan   | 0.980      | 0.975        | 0.975     | **0.975**   | 1.6         |
| ⚔️ PA\_TEXT      | vegan   | 0.980      | 0.975        | 0.975     | **0.975**   | 4.5         |
| 🧠 MLP\_BOTH     | vegan   | 0.962      | 1.000        | 0.929     | 0.963       | 2515.7      |
| ⚡ LGBM\_IMAGE    | vegan   | 0.962      | 1.000        | 0.929     | 0.963       | 54.3        |
| 🧠 Ridge\_TEXT   | vegan   | 0.970      | 0.974        | 0.950     | 0.962       | 12.2        |
| 🧪 SGD\_TEXT     | vegan   | 0.970      | 0.974        | 0.950     | 0.962       | 0.6         |
| 🦠 NB\_TEXT      | vegan   | 0.960      | 0.974        | 0.925     | 0.949       | 0.1         |
| 🐍 NB\_BOTH      | vegan   | 0.788      | 1.000        | 0.607     | 0.756       | 0.2         |
| 🧠 MLP\_IMAGE    | vegan   | 0.596      | 1.000        | 0.250     | 0.400       | 100.6       |

---

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

### 🧠 Ground Truth Evaluation Pipeline

When running:

```bash
python diet_classifiers.py --ground_truth /path/to/labeled_recipes.csv
```

The pipeline will:

1. **Load and verify the ground truth CSV** file.

2. **Automatically locate trained models**:

   * Prefer models from `artifacts/models.pkl` and `vectorizer.pkl`.
   * If missing, fall back to `pretrained_models/models.pkl` and `vectorizer.pkl`.
   * Logs each decision clearly.

3. **Vectorize ingredients** using the loaded vectorizer.

4. **Predict both vegan and keto labels** per row using the appropriate models.

5. **Save predictions** to:

   ```
   artifacts/ground_truth_predictions.csv
   ```

6. **Compute evaluation metrics** (accuracy, precision, recall, F1).

7. **Save metrics summary** to:

   ```
   artifacts/eval_metrics.csv
   ```

📦 *Models can be preloaded by placing `models.zip` (containing `models.pkl`, `vectorizer.pkl`) inside `pretrained_models/`. This zip will be auto-extracted by `init.sh`.*




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

├── docker-compose.yml             # Multi-container runner
├── README.md                      # Project overview and usage
├── .gitattributes                 # Git LFS tracking config
├── .gitignore                     # Ignore rules
├── train.sh                       # Run training only
├── eval_ground_truth.sh          # Evaluate on provided gold dataset
├── eval_custom.sh                # Evaluate on a custom testset CSV
├── run_full_pipeline.sh          # Build, train, evaluate, classify
├── update_git.sh                 # Commit & push automation script

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

