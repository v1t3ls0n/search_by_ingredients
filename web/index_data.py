from opensearchpy import OpenSearch
from time import sleep
import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
import logging
from argparse import ArgumentParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def create_index(client: OpenSearch, index_name: str = "recipes"):
    """Create the recipes index with appropriate mappings"""
    if not client.indices.exists(index=index_name):
        mappings = {
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "description": {"type": "text"},
                    "ingredients": {"type": "text"},
                    "instructions": {"type": "text"},
                    "photo_url": {"type": "keyword"}
                }
            }
        }
        client.indices.create(index=index_name, body=mappings)
        logger.info(f"Created index: {index_name}")

def batch_index_recipes(client: OpenSearch, recipes: List[Dict], index_name: str = "recipes", batch_size: int = 1024):
    """Index a batch of recipes into OpenSearch"""
    actions = []
    for recipe in recipes:
        # Add the action metadata line
        action_metadata = {
            "index": {
                "_index": index_name
            }
        }
        actions.append(action_metadata)
        # Add the document
        actions.append(recipe)
        
        # When we reach the batch size, index the batch
        if len(actions) >= batch_size * 2:  # Multiply by 2 because each document has 2 lines
            client.bulk(body=actions)
            logger.info(f"Indexed {len(actions)//2} recipes")
            actions = []
    
    # Index any remaining recipes
    if actions:
        client.bulk(body=actions)
        logger.info(f"Indexed {len(actions)//2} recipes")

def main(args):
    # Initialize OpenSearch client
    client = OpenSearch(
        hosts=[args.opensearch_url],
        http_auth=None,
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False,
    )

    # Wait for OpenSearch to be ready
    if not wait_for_opensearch(client):
        logger.error("Failed to connect to OpenSearch")
        sys.exit(1)

    # Create the index with mappings
    create_index(client)

    # Read and index the recipes
    data_path = Path(args.data_file)
    if not data_path.exists():
        logger.error(f"Could not find {data_path}")
        sys.exit(1)

    logger.info("Starting to index recipes...")
    # Read the parquet file
    df = pd.read_parquet(data_path)
    
    # Convert DataFrame to list of dictionaries
    recipes = df.to_dict('records')
    
    # Process in batches
    for i in range(0, len(recipes), args.batch_size):
        batch = recipes[i:i + args.batch_size]
        batch_index_recipes(client, batch)
        logger.info(f"Processed {i + len(batch)} recipes")

    logger.info("Indexing completed successfully")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for indexing")
    parser.add_argument("--data_file", type=str, default="data/allrecipes.parquet", help="Path to the Parquet file")
    parser.add_argument("--opensearch_url", type=str, default="http://localhost:9200", help="OpenSearch URL")
    main(parser.parse_args()) 