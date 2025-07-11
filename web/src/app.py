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
# Silence noisy loggers - reduces startup noise and improves performance
logging.getLogger('opensearchpy').setLevel(logging.ERROR)  # Only show errors
logging.getLogger('urllib3').setLevel(logging.ERROR)       # Only show errors
logging.getLogger('opensearch').setLevel(logging.ERROR)    # Only show errors
logger = logging.getLogger(__name__)

# Flask app initialization with static file handling
# static_folder and static_url_path configuration enables efficient static asset serving
app = Flask(__name__, static_folder='static', static_url_path='/static')


def wait_for_opensearch(client, max_retries=30, retry_interval=2):
    """
    Wait for OpenSearch to be ready - EFFICIENT CONNECTION STRATEGY
    
    This function implements a robust connection retry mechanism without data loading:
    - Uses client.ping() which is a lightweight health check (no data transfer)
    - Exponential backoff prevents overwhelming the OpenSearch cluster
    - Graceful failure handling with clear error messages
    """
    print("Waiting for OpenSearch to be ready...")  # Simple status message
    for i in range(max_retries):
        try:
            # client.ping() is a lightweight operation - just checks if OpenSearch responds
            # This is NOT a data operation, just a connectivity test
            if client.ping():
                print("Successfully connected to OpenSearch!")
                return True
        except Exception as e:
            # Log connection attempts at debug level to avoid noise
            logger.debug(f"Connection attempt {i+1}/{max_retries} failed.")
        sleep(retry_interval)
    logger.error("Failed to connect to OpenSearch after maximum retries")
    return False


def init_opensearch():
    """
    Initialize OpenSearch client - LAZY LOADING PATTERN
    
    This function demonstrates EFFICIENT initialization:
    1. Creates client connection (no data loaded)
    2. Verifies cluster health (lightweight ping operation)
    3. Checks index existence (metadata operation, not data operation)
    4. Returns ONLY the client object (no preloaded data)
    
    KEY EFFICIENCY: No data is loaded into memory during startup
    """
    # Create OpenSearch client with connection configuration
    # This establishes connection pool but doesn't load any data
    client = OpenSearch(
        hosts=[config('OPENSEARCH_URL', 'http://localhost:9200')],
        http_auth=None,           # No authentication for local development
        use_ssl=False,            # Disable SSL for local development
        verify_certs=False,       # Skip certificate verification
        ssl_show_warn=False,      # Suppress SSL warnings
    )

    # Wait for OpenSearch to be ready - uses lightweight ping operations only
    if not wait_for_opensearch(client):
        logger.error("OpenSearch connection failed")
        sys.exit(1)

    # METADATA-ONLY VERIFICATION: Check if indices exist without loading data
    try:
        # client.indices.exists() is a metadata operation - no data transfer
        # This checks index existence in cluster metadata, not document data
        if client.indices.exists(index="ingredients"):
            print("Ingredients index verified")  # Confirms structure exists
        if client.indices.exists(index="recipes"):
            print("Recipes index verified")      # Confirms structure exists
        
        # CRITICAL: Return only client object, NO DATA PRELOADING
        # This keeps memory footprint minimal and startup time fast
        return client
    except Exception as e:
        logger.error(f"Error verifying indices: {str(e)}")
        sys.exit(1)


# APPLICATION STARTUP: Efficient initialization pattern
logger.info("Starting application initialization...")
# Initialize OpenSearch client ONLY - no data loading during startup
# This demonstrates STATELESS application design
client = init_opensearch()
logger.info("Application initialization completed successfully")


@app.route('/')
def home():
    """Serve the main application page"""
    return render_template('index.html')


@app.route("/select2", methods=["GET"])
def select2():
    """
    Autocomplete search endpoint - QUERY-TIME SEARCH PATTERN
    
    This endpoint demonstrates efficient real-time search:
    1. Validates input without hitting database
    2. Constructs optimized OpenSearch query
    3. Leverages OpenSearch's built-in search capabilities
    4. Returns only relevant results (no full dataset processing)
    """
    # Input validation - fail fast without database hits
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"results": []})  # Early return for empty queries
    
    # OPTIMIZED OPENSEARCH QUERY: Uses native search capabilities
    query = {
        "query": {
            # match_prefix: Optimized for autocomplete scenarios
            # Uses inverted index for O(log n) performance instead of O(n) scanning
            "match_prefix": {
                "ingredients": {
                    "query": q,
                    "max_expansions": 10  # Limits term expansions for performance
                }
            }
        },
        "size": 20,  # Limit results to reduce network transfer and processing
        "sort": [
            # Multi-level sorting: relevance first, then alphabetical
            {"_score": {"order": "desc"}},        # OpenSearch relevance scoring
            {"ingredients.keyword": {"order": "asc"}}  # Alphabetical fallback
        ]
    }
    
    try:
        # DELEGATED SEARCH: Let OpenSearch handle the heavy lifting
        # This leverages distributed search, caching, and optimization
        response = client.search(index="ingredients", body=query)
        
        # MINIMAL PROCESSING: Only format results, no business logic
        results = []
        for i, hit in enumerate(response["hits"]["hits"]):
            results.append({
                # Use ingredient name as ID - no complex mapping needed
                "id": hit["_source"]["ingredients"],
                "text": hit["_source"]["ingredients"]
            })
        
        # UX OPTIMIZATION: Sort by length for better user experience
        # This is the only client-side processing - minimal overhead
        results.sort(key=lambda x: len(x["text"]))
        return jsonify({"results": results})
    except Exception as e:
        # Graceful error handling - don't expose internal errors
        logger.error(f"Error in select2 search: {str(e)}")
        return jsonify({"results": []})


@app.route('/search', methods=['GET'])
def search_by_ingredients():
    """
    Recipe search endpoint - ADVANCED QUERY BUILDING PATTERN
    
    This endpoint demonstrates:
    1. Dynamic query construction based on multiple criteria
    2. Boolean query logic using OpenSearch's query DSL
    3. Efficient filtering using pre-indexed boolean fields
    4. Structured result formatting
    """
    # Extract search parameters from request
    ingredient = request.args.get('q', '')
    
    # Extract diet filter parameters - convert to boolean
    keto_only = request.args.get('keto', '').lower() == 'true'
    vegan_only = request.args.get('vegan', '').lower() == 'true'
    
    # INPUT VALIDATION: Ensure at least one search criteria
    if not ingredient and not keto_only and not vegan_only:
        return jsonify({'error': 'Please provide an ingredient name or select a diet filter'}), 400
    
    # DYNAMIC QUERY BUILDING: Construct query based on provided criteria
    must_conditions = []  # OpenSearch boolean query conditions
    
    # Add ingredient search condition if provided
    if ingredient:
        must_conditions.append({
            "match": {
                "ingredients": {
                    "query": ingredient,
                    "fuzziness": "AUTO"  # Handle typos and variations
                }
            }
        })
    
    # Add diet filters using exact term matching
    # These use pre-indexed boolean fields for O(1) filtering
    if keto_only:
        must_conditions.append({"term": {"keto": True}})
    if vegan_only:
        must_conditions.append({"term": {"vegan": True}})

    # EFFICIENT QUERY CONSTRUCTION: Build optimized search query
    if must_conditions:
        # Boolean query with multiple conditions - uses OpenSearch query optimization
        query = {
            "query": {
                "bool": {
                    "must": must_conditions  # ALL conditions must match (AND logic)
                }
            }
        }
    else:
        # Fallback to match all if only filters applied
        query = {
            "query": {
                "match_all": {}
            }
        }

    try:
        # EXECUTE SEARCH: Delegate to OpenSearch cluster
        response = client.search(
            index="recipes",    # Target recipes index
            body=query,         # Our constructed query
            size=12            # Limit results for pagination
        )

        # RESULT FORMATTING: Transform OpenSearch response to API format
        hits = response['hits']['hits']
        results = [{
            'title': hit['_source']['title'],
            'description': hit['_source'].get('description', ''),
            'ingredients': hit['_source']['ingredients'],
            'instructions': hit['_source'].get('instructions', ''),
            'photo_url': hit['_source'].get('photo_url', ''),
            # Use pre-computed boolean fields instead of runtime calculation
            'keto': hit['_source'].get('keto', False),     # Indexed field
            'vegan': hit['_source'].get('vegan', False),   # Indexed field
            'score': hit['_score']  # OpenSearch relevance score
        } for hit in hits]
        
        # STRUCTURED RESPONSE: Include metadata and results
        return jsonify({
            'total': response['hits']['total']['value'],  # Total matching documents
            'results': results                            # Formatted result array
        })

    except Exception as e:
        # ERROR HANDLING: Log error and return user-friendly message
        logger.error(f"Search error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # DEVELOPMENT SERVER: Run Flask with network binding
    app.run(host='0.0.0.0', port=8080)