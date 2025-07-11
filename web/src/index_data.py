import sys
import string
import json
import re
import logging
from opensearchpy import OpenSearch
from time import sleep
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
from diet_classifiers import is_keto, is_vegan

# Try to import swifter for parallel processing
try:
    import swifter
    SWIFTER_AVAILABLE = True
except ImportError:
    SWIFTER_AVAILABLE = False

# Configure logging
logging.getLogger('opensearch').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_ingredient(ingredient: str) -> str:
    if type(ingredient) != str:
        return str(ingredient)
    ingredient = ingredient.lower().strip()
    ingredient = ingredient.rsplit(',', 1)[0]
    ingredient = re.sub(r"\([^()]+\)", "", ingredient)
    ingredient = re.sub(r'\d+\s*\d*/\d*', '', ingredient)
    ingredient = ingredient.translate(
        {ord(c): ' ' for c in string.punctuation})
    measurements = [
        'cup', 'cups', 'can', 'cans', 'tablespoon', 'tablespoons', 'tbsp', 'teaspoon', 'teaspoons', 'tsp',
        'ounce', 'ounces', 'oz', 'pound', 'pounds', 'lb', 'lbs', 'gram', 'grams', 'g',
        'kilogram', 'kilograms', 'kg', 'milliliter', 'milliliters', 'ml', 'liter', 'liters', 'l',
        'pinch', 'pinches', 'dash', 'dashes', 'piece', 'pieces', 'slice', 'slices', 'small', 'medium', 'large',
        'cube', 'cubes', 'inch', 'inches', 'cm', 'mm', 'quart', 'quarts', 'qt', 'jar', 'scoop', 'scoops',
        'gallon', 'gallons', 'gal', 'pint', 'pints', 'pt', 'fluid ounce', 'fluid ounces', 'fl oz', 'package', 'packages', 'pkg', 'pack', 'packs'
    ]
    pattern = r'\b(' + '|'.join(measurements) + r')\b'
    ingredient = re.sub(pattern, '', ingredient)
    ingredient = re.sub(r'[^a-z\s]', '', ingredient)
    if ingredient.endswith('ies'):
        ingredient = ingredient[:-3]+'y'
    elif ingredient.endswith('es'):
        if ingredient[-3] in {'s', 'x', 'z'}:
            ingredient = ingredient[:-2]
        else:
            ingredient = ingredient[:-1]
    elif ingredient.endswith('s'):
        ingredient = ingredient[:-1]
    ingredient = ' '.join(ingredient.split())
    return ingredient


def wait_for_opensearch(client, max_retries=30, retry_interval=2):
    """Wait for OpenSearch to be ready"""
    for i in range(max_retries):
        try:
            if client.ping():
                return True
        except:
            pass
        print(
            f"Waiting for OpenSearch to be ready... (attempt {i+1}/{max_retries})")
        sleep(retry_interval)
    return False


def check_data_exists(client: OpenSearch) -> bool:
    """Simple check if data exists in OpenSearch indices"""
    try:
        # Check if indices exist and have data
        recipes_count = client.count(index="recipes")["count"]
        ingredients_count = client.count(index="ingredients")["count"]

        if recipes_count > 0 and ingredients_count > 0:
            logger.info(
                f"Found existing data: {recipes_count} recipes and {ingredients_count} ingredients")
            return True

        return False
    except Exception as e:
        logger.error(f"Error checking data existence: {e}")
        return False


def delete_existing_data(client: OpenSearch):
    """Delete existing data from OpenSearch indices"""
    try:
        if client.indices.exists(index="recipes"):
            client.indices.delete(index="recipes")
            logger.info("Deleted existing recipes index")
        if client.indices.exists(index="ingredients"):
            client.indices.delete(index="ingredients")
            logger.info("Deleted existing ingredients index")
    except Exception as e:
        logger.error(f"Error deleting existing data: {e}")


def create_index(client: OpenSearch):
    """Create the recipes index with appropriate mappings"""
    if not client.indices.exists(index="recipes"):
        mappings = {
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "description": {"type": "text"},
                    "ingredients": {"type": "text"},
                    "instructions": {"type": "text"},
                    "photo_url": {"type": "keyword"},
                    # diet metadatas
                    "keto":        {"type": "boolean"},
                    "vegan":       {"type": "boolean"},
                }
            }
        }
        client.indices.create(index="recipes", body=mappings)
        logger.info(f"Created index: recipes")

    if not client.indices.exists(index="ingredients"):
        mappings = {
            "mappings": {
                "properties": {
                    "ingredients": {"type": "text"}
                }
            }
        }
        client.indices.create(index="ingredients", body=mappings)
        logger.info(f"Created index: ingredients")


def batch_index_recipes(client: OpenSearch, df: pd.DataFrame, batch_size: int = 10240):
    """
    Index a batch of recipes with diet classification using pandas with swifter for parallel processing.
    """
    if SWIFTER_AVAILABLE:
        logger.info("Applying diet classifiers with swifter (parallel processing)...")
        # Use swifter for parallel processing
        df['keto'] = df['ingredients'].swifter.apply(is_keto)
        df['vegan'] = df['ingredients'].swifter.apply(is_vegan)
    else:
        logger.warning("Swifter not available, using standard pandas (sequential processing)")
        logger.info("Install swifter for parallel processing: pip install swifter")
        # Fallback to standard pandas
        df['keto'] = df['ingredients'].apply(is_keto)
        df['vegan'] = df['ingredients'].apply(is_vegan)
    
    # Convert to records for OpenSearch indexing
    recipes = df.to_dict('records')
    
    # Bulk index to OpenSearch
    actions = []
    ingredients = set()
    
    for recipe in recipes:
        actions.append({"index": {"_index": "recipes"}})
        actions.append(recipe)
        ingredients |= {normalize_ingredient(ing) for ing in recipe["ingredients"]}
        
        if len(actions) >= batch_size * 2:
            client.bulk(body=actions)
            actions = []

    # Index any remaining recipes
    if actions:
        client.bulk(body=actions)

    # Index ingredients
    actions = []
    for ing in ingredients:
        actions.append({"index": {"_index": "ingredients"}})
        actions.append({"ingredients": ing})
    client.bulk(body=actions)


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

    # Check if data already exists
    if check_data_exists(client):
        logger.info("Data already exists in OpenSearch, skipping indexing")
        return

    # Create the index with mappings
    create_index(client)

    # Read the parquet file
    data_path = Path(args.data_file)
    if not data_path.exists():
        logger.error(f"Could not find {data_path}")
        sys.exit(1)

    logger.info("Reading parquet file...")
    df = pd.read_parquet(data_path)

    # Process in batches with progress bar
    processing_mode = "parallel" if SWIFTER_AVAILABLE else "sequential"
    logger.info(f"Starting to index recipes using {processing_mode} processing...")
    total_recipes = len(df)
    
    with tqdm(total=total_recipes, desc=f"Indexing recipes ({processing_mode})") as pbar:
        for i in range(0, total_recipes, args.batch_size):
            batch_df = df.iloc[i:i + args.batch_size].copy()
            
            # Index the batch with diet classification
            batch_index_recipes(client, batch_df, args.batch_size)
            
            pbar.update(len(batch_df))

    logger.info("Indexing completed successfully")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int,
                        default=1024, help="Batch size for indexing")
    parser.add_argument("--data_file", type=str,
                        default="data/allrecipes.parquet", help="Path to the Parquet file")
    parser.add_argument("--opensearch_url", type=str,
                        default="http://localhost:9200", help="OpenSearch URL")
    main(parser.parse_args())