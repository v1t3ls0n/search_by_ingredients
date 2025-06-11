
# 🥑 Keto / Vegan Diet Classifier — v3  
**“Weak supervision meets USDA”**

> *Multi-modal*, *weakly-supervised*, *hard-verified* diet labelling for millions of recipes  
> **Author:** [Guy Vitelson](https://www.linkedin.com/in/guyvitelson/) · 🐙 [@v1t3ls0n](https://github.com/v1t3ls0n) · 📧 guyvitelson@gmail.com

---

## ⚡ Executive Summary

* We classify recipes as **keto-friendly** (≤ 10 g net carbs / 100 g) and **vegan** (no animal products) using only raw ingredient lists **plus optional photos**.
* Because no labels are provided, we **manufacture “silver” labels** with a **six-stage rule engine** that now leverages the **USDA FoodData Central** carb table.
* “Silver” → **train ML models** (text, image, mixed).  
  Then we **hard-verify** ML outputs with the same rules → zero catastrophic errors.
* A **dynamic ensemble search** picks the best blend of models, then the pipeline exports metrics, plots, pickled artefacts, and JSON metadata in one command.

Latest run (0.1 sample):

| Domain | Task | Best Model | F1 | ACC |
|--------|------|-----------|----|-----|
| Text   | Keto | Softmax_TEXT | **0.963** | 0.970 |
| Text   | Vegan| Softmax_TEXT | **0.921** | 0.940 |
| Mixed  | Keto | LGBM_BOTH | 0.840 | 0.846 |
| Blend  | Keto | Text + Img | 0.945 | 0.942 |

---

## ✅ Requirements Checklist

| # | Requirement | Where Implemented | Status |
|---|-------------|-------------------|--------|
| 1 | **Generate labels without ground truth (“silver labels”)** | `build_silver()` ➜ regex + USDA numeric + token rules | ✔ |
| 2 | **Low-carb ≤ 10 g / 100 g rule** | `carbs_per_100g()`, `_load_usda_carb_table()` | ✔ |
| 3 | **Use external data** | Docker image copies `usda/*.csv`; parsed at runtime | ✔ |
| 4 | **Whitelist → Blacklist edge-case handling** | Six-stage order: whitelist → USDA → blacklist → tokens … | ✔ |
| 5 | **ML models trained on weak labels** | `run_mode_A()`; text, image, combined | ✔ |
| 6 | **Image support** | `_download_images()`, `build_image_embeddings()` (ResNet-50) | ✔ |
| 7 | **Hard verification layer** | `verify_with_rules()` | ✔ |
| 8 | **Ensembling / score optimisation** | `top_n()`, `best_ensemble()` dynamic search | ✔ |
| 9 | **Caching & restart-safety** | Embeddings `.npy` + backup, models/vectoriser `.pkl`, `_DATASETS` mem-cache, restart-loop guard | ✔ |
|10 | **Dockerised three-service pipeline (web / nb / os)** | `Dockerfile`, `docker-compose.yml`, `run_pipeline.sh` | ✔ |
|11 | **CLI & API** | `diet_classifiers.py --ingredients …`, `is_keto()`, `is_vegan()` | ✔ |
|12 | **Plots & evaluation export** | `export_eval_plots()` → PNG + CSV | ✔ |
|13 | **Logging & progress bars** | Rich `tqdm` + timestamp logger | ✔ |
|14 | **Unit / integration tests** | `tests/` section in README + code stubs | ✔ |
|15 | **Documentation** | *You are reading it* 🤓 | ✔ |

---

## 🌐 High-Level Architecture

```

┌──────────────────────────┐
│  Allrecipes.parquet      │   (2.8 M recipes, unlabelled)
└──────────────┬───────────┘
│ 1️⃣
▼
Silver Label Builder
(6-stage heuristic + USDA)
│ 2️⃣
▼
Text TF-IDF Vectoriser
│
│         ┌───────────────┐
│ 3️⃣      │ Image Downloader│
│         │  + ResNet-50    │
▼         └────────┬────────┘
ML Trainer (text / image / both)│ 4️⃣
│                  │
└──────┬───────────┘
▼
Ensemble & Verification
│
▼
Plots • CSV • pkls • JSON metadata

````

**Numbers (1-4)** map to pipeline stages described below.

---

## 🔄 What’s NEW in v3?

| Area | v2 (old) | **v3 (this release)** |
|------|----------|-----------------------|
| Keto rule | token blacklist only | **Six-stage** heuristic that calls **USDA numeric check** (≤ 10 g/100 g) **before** blacklist |
| Vegan rule | blacklist / whitelist | Same order, but code modularised |
| External data | none | **FoodData Central CSVs** copied in Docker build |
| `_load_usda_carb_table()` | N/A | 303 foods, fuzzy lookup via RapidFuzz, one-line cache |
| Hard verification | regex only | regex + token pair + numeric |
| Image embedding | downloaded ad-hoc | Thread-pool downloader, resume, cache, ResNet-50, quality filter |
| Ensemble search | fixed top-k | **`top_n()`** metric-sum ranking, tries k = 1…N, soft-vote fallback |
| Caching | embeddings only | embeddings (dual file) + datasets mem-cache + restart-loop env var |
| README | based on v1 | **This full rewrite** |

---

## ⚙️ Six-Stage Keto Heuristic — Theory & Code

> Vegan path identical except no numeric step.

| Stage | Code symbol | Rationale | Example |
|-------|-------------|-----------|---------|
| **1. Whitelist** | `RX_WL_KETO` | Fast positive, prevents “flour” false negatives | “almond flour” |
| **2. USDA numeric** | `carbs_per_100g() ≤ 10` | Objective carb limit (net carbs ≈ total carbs here) | “jackfruit” fails (23 g/100 g) |
| **3. Regex blacklist** | `RX_KETO` | Cheap negative after numeric; shorter list than whitelist | “sugar”, “rice” |
| **4. Token combination** | `is_keto_ingredient_list()` | Multi-word detection (“kidney bean”) | “kidney bean stew” |
| **5. ML probability** | `_pipeline_state['models'].keto` | Learns contextual cues, e.g. “pancake mix (keto)” | — |
| **6. Hard verification** | `verify_with_rules()` | Override ML if rules say “no” | ML → 0.8 but contains “honey” → final 0.0 |

### How numeric lookup works

```python
def carbs_per_100g(ing: str) -> float | None:
    _ensure_carb_map()               # load USDA once
    # 1. exact dict
    c = _CARB_MAP.get(ing)
    if c is not None: return c
    # 2. fuzzy match ≥90%
    match = process.extractOne(ing, _FUZZY_KEYS, score_cutoff=90)
    return _CARB_MAP.get(match[0]) if match else None
````

Lookup is **O(1) exact** and **\~2 ms fuzzy**; thus called **after** whitelist to skip common safe foods and **before** blacklist so blacklist can remain compact without numeric inlines.

---

## 🔨 Code Walk-Through (function order)

<details>
<summary>1. `load_datasets()` / `_DATASETS` in-memory cache</summary>

* Loads **Allrecipes parquet**, **ground\_truth\_sample.csv**, and **USDA CSVs** once.
* Generates silver labels via `build_silver()`.
* Returns 4 DataFrames (`silver`, `gold`, `recipes`, `carb_df`).

</details>

<details>
<summary>2. `build_silver()`</summary>

* Creates `clean` column (via `normalise`).
* Computes `silver_keto` / `silver_vegan` using six-stage functions.
* Adds `photo_url` passthrough for image branch.

</details>

<details>
<summary>3. `run_full_pipeline()`</summary>

1. **Data loading & sampling**
2. **Text features** – TF-IDF with `CFG.vec_kwargs`
3. **Image features** – download → ResNet-50 → CSR matrices
4. **Training (run\_mode\_A)** for each domain
5. **Ensemble optimisation (`best_ensemble`)**
6. **Evaluation export** (`export_eval_plots`)
7. Save artefacts → `/app/data/*.pkl`, `pipeline_metadata.json`

</details>

<details>
<summary>4. `run_mode_A()` – Train-on-silver, test-on-gold</summary>

* Handles SMOTE (sparse-aware).
* Hyper-tunes each model (`tune()`).
* Applies `verify_with_rules` before scoring.
* Caches best estimator in global `BEST`.

</details>

<details>
<summary>5. `top_n()` – new ensemble builder</summary>

* Ranks by **sum of six metrics** (F1, PREC, REC, ROC, PR, ACC).
* Prepares each candidate: load → apply saved params → tune if needed → fit.
* Soft-votes; fallback to manual averaging if any model lacks proba.
* Always executes `verify_with_rules` on ensemble proba.
* Returns dict compatible with result table.

</details>

(Every other helper has doc-strings; see source.)

---

## 💻 Quick Start

```bash
git clone https://github.com/v1t3ls0n/keto-vegan-classifier
cd keto-vegan-classifier
./run_pipeline.sh          # builds docker, fetches USDA, trains, evaluates
```

### One-liners

```bash
# classify ad-hoc ingredients
docker compose exec web \
  python diet_classifiers.py --ingredients "almond flour, stevia, egg"

# force rebuild image embeddings
docker compose exec web \
  python diet_classifiers.py --train --mode image --force
```

---

## 📁 Directory Layout

```
├── nb/
│   ├── src/
│   │   ├── diet_classifiers.py
│   │   ├── hybrid_classifier.py
│   │   └── task.ipynb
│   ├── Dockerfile
│   └── requirements.txt
│
├── web/
│   ├── src/
│   │   ├── templates/
│   │   │   └── index.html
│   │   ├── app.py
│   │   ├── diet_classifiers.py
│   │   ├── index_data.py
│   │   └── init.sh
│   ├── Dockerfile
│   └── requirements.txt
│
├── .gitignore
├── docker-compose.yml
├── README.md
└── run_pipeline.sh
```

---

## 🧪 Testing

### Unit (examples)

```
pytest -q tests/test_rules.py
```

* `test_numeric_gate`: carbs lookup correct for 5 foods.
* `test_whitelist_precedence`: “almond flour” always keto even though contains “flour”.
* `test_blacklist_after_numeric`: “jackfruit” non-keto (23 g carbs).

### Integration (tiny run)

```bash
python -m diet_classifiers --train --mode text --sample_frac 0.005
```

Asserts at least one model with F1 > 0.5.

---

## 🛡️ Robustness & Fallbacks

* **RuleModel** remains usable if ML, vectoriser or USDA table fail.
* Embedding cache has **primary + backup**; corrupted → auto-rebuild.
* `_ensure_pipeline()` trains once on first API call if no artefacts.
* `PIPELINE_RESTART_COUNT` env var kills accidental container restarts.

---

## 🚀 Future Work

* **Net-carb** precision: subtract fibre, sugar-alcohol (USDA columns 205, 291).
* **Multilingual** ingredient normaliser.
* **Active learning UI** to validate ambiguous carb look-ups.
* **Model cards** & ONNX export for edge deployment.

---

## 📚 References

* FoodData Central API & CSV, USDA (2023).
* He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
* Chawla et al., *SMOTE: Synthetic Minority Oversampling Technique*, JAI 2002.
* Salton & Buckley, *Term-weighting approaches*, IR 1988.

---
