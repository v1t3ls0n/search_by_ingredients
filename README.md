# Search by Ingredients

This repository contains a small demonstration project for indexing recipe data and querying it by ingredients. A Flask based web service provides a search API backed by OpenSearch. Basic rule based classifiers are included to predict whether a recipe is keto friendly or vegan.

## Project layout

```
.
├── docker-compose.yml    # Docker services for OpenSearch, the web API and a notebook
├── web/                  # Web API source code and container
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── app.py        # Flask application
│       ├── diet_classifiers.py  # Heuristic and ML classifiers
│       ├── index_data.py # Script for indexing recipe data into OpenSearch
│       └── init.sh       # Container entrypoint
├── nb/                   # Jupyter environment with lightweight examples
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── diet_classifiers.py  # Minimal rule based implementation
│       └── task.ipynb
├── run_pipeline.sh       # Convenience script: builds containers and trains models
├── train_and_evaluate.sh # Runs training and evaluation inside containers
└── README.md
```

## Requirements

* [Docker](https://www.docker.com/) and Docker Compose.
* Alternatively, Python 3.11 with the packages listed in `web/requirements.txt` and `nb/requirements.txt`.

## Getting started

1. Build the Docker images and start the services:
   ```bash
   docker compose build
   docker compose up -d
   ```
   The first run downloads the recipe dataset and indexes it into OpenSearch.
2. Open <http://localhost:8080> to access the search API.
3. To experiment in a notebook environment visit <http://localhost:8888>.

The `web` service exposes a small search interface and an endpoint `/search` which accepts ingredient ids and returns matching recipes with predicted diet labels.

## Training the classifiers

The repository includes a script that can download the data, generate image embeddings using ResNet and train text and image based classifiers. Run it inside the `web` container:

```bash
docker compose exec web python web/diet_classifiers.py --train --mode both
```

Evaluation on a small gold dataset can be performed with:

```bash
docker compose exec web python web/diet_classifiers.py --ground_truth /usr/src/data/ground_truth_sample.csv
```

The `run_pipeline.sh` helper script combines the above steps.

## Indexing new data

`web/src/index_data.py` reads a parquet file of recipes and loads them into OpenSearch. It is executed automatically the first time the container starts, but can be run manually:

```bash
python web/src/index_data.py --data_file data/allrecipes.parquet --opensearch_url http://localhost:9200
```

## License

This project is provided as a coding exercise and carries no explicit license.
