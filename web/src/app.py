from flask import Flask, request, jsonify, render_template
from opensearchpy import OpenSearch
from decouple import config
from diet_classifiers import is_keto, is_vegan
from time import sleep
import sys
import logging
import os

# Configure logging - keep it simple
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'  # Simplified format
)
# Silence noisy loggers
logging.getLogger('opensearchpy').setLevel(logging.ERROR)  # Only show errors
logging.getLogger('urllib3').setLevel(logging.ERROR)       # Only show errors
logging.getLogger('opensearch').setLevel(logging.ERROR)    # Only show errors
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='/static')


def wait_for_opensearch(client, max_retries=30, retry_interval=2):
    """Wait for OpenSearch to be ready"""
    print("Waiting for OpenSearch to be ready...")  # Simple status message
    for i in range(max_retries):
        try:
            if client.ping():
                # Simple success message
                print("Successfully connected to OpenSearch!")
                return True
        except Exception as e:
            # Log connection attempts at debug level
            logger.debug(f"Connection attempt {i+1}/{max_retries} failed.")
        sleep(retry_interval)
    logger.error("Failed to connect to OpenSearch after maximum retries")
    return False


def init_opensearch():
    """Initialize OpenSearch client and wait for it to be ready"""
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

    # Just verify the indices exist, don't load everything
    try:
        if client.indices.exists(index="ingredients"):
            print("Ingredients index verified")
        if client.indices.exists(index="recipes"):
            print("Recipes index verified")
        return client
    except Exception as e:
        logger.error(f"Error verifying indices: {str(e)}")
        sys.exit(1)


logger.info("Starting application initialization...")
# Initialize OpenSearch client only
client = init_opensearch()
logger.info("Application initialization completed successfully")


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/select2", methods=["GET"])
def select2():
    """Search ingredients directly from OpenSearch"""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"results": []})
    
    # Use OpenSearch to search ingredients
    query = {
        "query": {
            "match_prefix": {
                "ingredients": {
                    "query": q,
                    "max_expansions": 10
                }
            }
        },
        "size": 20,
        "sort": [
            {"_score": {"order": "desc"}},
            {"ingredients.keyword": {"order": "asc"}}
        ]
    }
    
    try:
        response = client.search(index="ingredients", body=query)
        results = []
        for i, hit in enumerate(response["hits"]["hits"]):
            results.append({
                "id": hit["_source"]["ingredients"],  # Use ingredient name as ID
                "text": hit["_source"]["ingredients"]
            })
        # Sort by length for better UX
        results.sort(key=lambda x: len(x["text"]))
        return jsonify({"results": results})
    except Exception as e:
        logger.error(f"Error in select2 search: {str(e)}")
        return jsonify({"results": []})


@app.route('/search', methods=['GET'])
def search_by_ingredients():
    ingredient = request.args.get('q', '')
    
    # Get diet filter parameters
    keto_only = request.args.get('keto', '').lower() == 'true'
    vegan_only = request.args.get('vegan', '').lower() == 'true'
    
    # Check if at least one search criteria is provided
    if not ingredient and not keto_only and not vegan_only:
        return jsonify({'error': 'Please provide an ingredient name or select a diet filter'}), 400
    
    # Build query conditions
    must_conditions = []
    
    # Add ingredient search if provided
    # Note: Since we're now using ingredient names as IDs, no need to convert
    if ingredient:
        must_conditions.append({
            "match": {
                "ingredients": {
                    "query": ingredient,
                    "fuzziness": "AUTO"
                }
            }
        })
    
    # Add diet filters if specified
    if keto_only:
        must_conditions.append({"term": {"keto": True}})
    if vegan_only:
        must_conditions.append({"term": {"vegan": True}})

    # Build the query
    if must_conditions:
        query = {
            "query": {
                "bool": {
                    "must": must_conditions
                }
            }
        }
    else:
        query = {
            "query": {
                "match_all": {}
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
            'keto': hit['_source'].get('keto', False),
            'vegan': hit['_source'].get('vegan', False),
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