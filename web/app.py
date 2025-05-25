from flask import Flask, request, jsonify, render_template
from opensearchpy import OpenSearch
from decouple import config

app = Flask(__name__)

# Initialize OpenSearch client
client = OpenSearch(
    hosts=[config('OPENSEARCH_URL', 'http://localhost:9200')],
    http_auth=None,
    use_ssl=False,
    verify_certs=False,
    ssl_show_warn=False,
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/select2", methods=["GET"])
def select2():
    q = request.args.get("q").strip()
    ingredients = [("1", "apple"), ("2", "banana"), ("3", "cherry")]
    results = [{"id": id_, "text": txt_} for id_,txt_ in ingredients if q in txt_]
    return jsonify({"results": results})


@app.route('/search', methods=['GET'])
def search_by_ingredients():
    ingredient = request.args.get('q', '')
    if not ingredient:
        return jsonify({'error': 'Please provide an ingredient name'}), 400

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