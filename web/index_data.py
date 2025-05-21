from opensearchpy import OpenSearch
from decouple import config
from time import sleep
import sys

def wait_for_opensearch(client, max_retries=30, retry_interval=2):
    """Wait for OpenSearch to be ready"""
    for i in range(max_retries):
        try:
            if client.ping():
                return True
        except:
            pass
        print(f"Waiting for OpenSearch to be ready... (attempt {i+1}/{max_retries})")
        sleep(retry_interval)
    return False

def index_data():
    # Initialize OpenSearch client
    client = OpenSearch(
        hosts=[config('OPENSEARCH_URL')],
        http_auth=None,  # Add authentication if needed
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False,
    )

    # Wait for OpenSearch to be ready
    if not wait_for_opensearch(client):
        print("Failed to connect to OpenSearch")
        sys.exit(1)

    # Your indexing logic here
    # Example:
    # documents = [...]  # Your data to index
    # for doc in documents:
    #     client.index(
    #         index="your_index",
    #         body=doc,
    #         refresh=True
    #     )
    print("Indexing completed successfully")

if __name__ == "__main__":
    index_data() 