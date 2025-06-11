# 🥑 Search by Ingredients

### AI-powered diet labeling from recipe ingredients and images

#### By [Guy Vitelson](https://www.linkedin.com/in/guyvitelson/)  
📧 [Email me](mailto:guyvitelson@gmail.com) · 🐙 [GitHub @v1t3ls0n](https://github.com/v1t3ls0n) · 💼 [LinkedIn](https://www.linkedin.com/in/guyvitelson/)

---

## 🧠 Overview

This project demonstrates a complete pipeline for diet-based recipe classification using both textual ingredient lists and recipe images. It includes:

* ⚡ Silver label generation from raw data using hard-coded heuristics.
* 🧃 Text-based models: Logistic Regression, SVM, Naive Bayes, Ridge, Passive-Aggressive, LightGBM, etc.
* 🖼️ Image-based classifier using ResNet embeddings.
* 🤖 Ensemble of multiple models (text + image) for final prediction.
* ✅ Hard-verification using blacklists and whitelists.
* 📊 Full evaluation on a gold-standard dataset with metrics, plots, and logging.

---

## 🗂️ Project Structure

```
.
├── data/                        # Mounted volume with datasets and results
├── web/                         # Web app and model training code
│   ├── diet_classifiers.py      # Full ML pipeline, heuristics, training, evaluation
│   ├── index_data.py            # Index recipe data into OpenSearch
│   ├── app.py                   # Minimal Flask app
│   ├── requirements.txt
│   ├── Dockerfile
│   └── init.sh
├── nb/                          # Optional Jupyter exploration environment
│   ├── task.ipynb
│   └── Dockerfile
├── run_pipeline.sh              # One-click train + evaluate runner
├── train_and_evaluate.sh        # Internal container script
├── docker-compose.yml
└── README.md
```

---

## 🧰 Setup Instructions

### 🔧 Requirements

* Docker + Docker Compose
* OR manually: Python 3.11 + packages in `web/requirements.txt`

### 🚀 Quickstart

```bash
sh run_pipeline.sh
```

This script:

1. Builds the Docker containers
2. Starts services (OpenSearch, Flask, notebook)
3. Trains models
4. Evaluates on gold-standard data
5. Outputs metrics, plots, and logs

### 🧪 Web endpoints

* API: [http://localhost:8080](http://localhost:8080)
* Notebook: [http://localhost:8888](http://localhost:8888)

---

## 🏗️ Functional Pipeline Overview

### 📦 1. Silver Dataset Generation

Silver labels are heuristically derived for both `vegan` and `keto` using strong dietary blacklists/whitelists applied to the raw recipe ingredient lists. This pseudo-labeling allows weak supervision on the unlabeled corpus.

### 🧠 2. Text Models

Textual ingredient data is encoded using `TfidfVectorizer`. Models trained include:

* `LogisticRegression`
* `LinearSVC` (calibrated)
* `SGDClassifier` (hinge/log)
* `MultinomialNB`
* `PassiveAggressiveClassifier`
* `RidgeClassifier`
* `LightGBM`

All models are optionally tuned via `GridSearchCV` with `class_weight` adjustments.

### 🖼️ 3. Image Embeddings

* Images are filtered using `filter_photo_rows`.
* Downloaded with progress bars.
* Features are extracted using `torchvision.models.resnet18` on resized images.
* Embeddings are saved as `.npz`.

### 🤝 4. Ensemble

Final prediction is a weighted ensemble combining:

* Top-N text-based models (based on F1-score)
* Image-based classifier (if available)
* Optional rule-based hard override (blacklist/whitelist)
* Voting strategy: majority vote with fallback to rules

### 📈 5. Evaluation (Gold Set)

Evaluation is performed using a held-out gold-labeled set with:

* Stratified metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * AUC
  * Kappa
  * MCC
* ROC Curves (annotated)
* Confusion Matrix (auto-saved)
* Class-specific performance breakdown
* CSV export of results

Progress bars (`tqdm`) are integrated for all key steps.

---

## 🔍 Evaluation Output Example

* `results/metrics_keto.csv`
* `results/roc_keto.png`
* `results/confusion_matrix_vegan.png`

All assets are auto-generated and saved.

---

## 🧪 Manual Evaluation

```bash
docker compose exec web python web/diet_classifiers.py --ground_truth /usr/src/data/ground_truth_sample.csv
```

Or train only:

```bash
docker compose exec web python web/diet_classifiers.py --train --mode both
```

---

## 🌐 Indexing New Data

```bash
docker compose exec web python web/index_data.py \
    --data_file /usr/src/data/allrecipes.parquet \
    --opensearch_url http://localhost:9200
```

---

## 📬 Author

**Guy Vitelson**
📍 Ramat Gan, Israel
🔗 [LinkedIn](https://www.linkedin.com/in/guyvitelson/)
🐙 [GitHub @v1t3ls0n](https://github.com/v1t3ls0n)

---

## ⚖️ License

This project is provided as a technical coding assessment. No explicit license is granted.

