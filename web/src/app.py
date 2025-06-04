from flask import Flask, request, jsonify, render_template
from opensearchpy import OpenSearch
from decouple import config
from diet_classifiers import is_keto, is_vegan
from time import sleep
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def wait_for_opensearch(client, max_retries=30, retry_interval=2):
    """Wait for OpenSearch to be ready"""
    logger.info("Starting to wait for OpenSearch...")
    for i in range(max_retries):
        logger.info(
            f"Attempting to connect to OpenSearch (attempt {i+1}/{max_retries})")
        try:
            if client.ping():
                logger.info("Successfully connected to OpenSearch!")
                return True
        except Exception as e:
            logger.error(f"Connection attempt {i+1} failed: {str(e)}")
        sleep(retry_interval)
    logger.error("Failed to connect to OpenSearch after maximum retries")
    return False


def init_opensearch():
    """Initialize OpenSearch client and wait for it to be ready"""
    logger.info("Initializing OpenSearch client...")
    logger.info(
        f"OpenSearch URL: {config('OPENSEARCH_URL', 'http://localhost:9200')}")

    client = OpenSearch(
        hosts=[config('OPENSEARCH_URL', 'http://localhost:9200')],
        http_auth=None,
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False,
    )

    if not wait_for_opensearch(client):
        logger.error("OpenSearch connection failed")
        sys.exit(1)

    try:
        # Load all ingredients once OpenSearch is ready
        logger.info("Loading ingredients from OpenSearch...")
        response = client.search(index="ingredients", body={
                                 "query": {"match_all": {}}}, size=10000)
        ingredients = [hit["_source"]["ingredients"]
                       for hit in response["hits"]["hits"]]
        logger.info(f"Successfully loaded {len(ingredients)} ingredients")
        return client, ingredients
    except Exception as e:
        logger.error(f"Error initializing OpenSearch: {str(e)}", exc_info=True)
        sys.exit(1)


logger.info("Starting application initialization...")
# Initialize OpenSearch and load ingredients
client, ingredients = init_opensearch()
logger.info("Application initialization completed successfully")


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/select2", methods=["GET"])
def select2():
    q = request.args.get("q", "").strip()
    results = [{"id": id_, "text": txt_}
               for id_, txt_ in enumerate(ingredients) if q in txt_]
    results = sorted(results, key=lambda x: len(x["text"]))
    return jsonify({"results": results})


@app.route('/search', methods=['GET'])
def search_by_ingredients():
    ingredient = request.args.get('q', '')
    if not ingredient:
        return jsonify({'error': 'Please provide an ingredient name'}), 400

    ingredient_ids = [int(id_) for id_ in ingredient.split() if id_.isdigit()]
    ingredient_ids = [ingredients[id_] for id_ in ingredient_ids]
    ingredient = " ".join(ingredient_ids)

    # Create the search query
    query = {
        "query": {
            "match": {
                "ingredients": {
                    "query": ingredient,
                    "fuzziness": "AUTO"
                }
            }
        }
    }

    try:
        # Execute the search
        response = client.search(
            index="recipes",
            body=query,
            size=12
        )

        # Format the results
        hits = response['hits']['hits']
        results = [{
            'title': hit['_source']['title'],
            'description': hit['_source'].get('description', ''),
            'ingredients': hit['_source']['ingredients'],
            'instructions': hit['_source'].get('instructions', ''),
            'photo_url': hit['_source'].get('photo_url', ''),
            'keto': is_keto(hit['_source']['ingredients']),
            'vegan': is_vegan(hit['_source']['ingredients']),
            'score': hit['_score']
        } for hit in hits]
        return jsonify({
            'total': response['hits']['total']['value'],
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
