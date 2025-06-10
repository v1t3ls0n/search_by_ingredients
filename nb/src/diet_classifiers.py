import json
import sys
from argparse import ArgumentParser
from typing import List
from time import time
import pandas as pd
try:
    from sklearn.metrics import classification_report
except ImportError:
    # sklearn is optional
    def classification_report(y, y_pred):
        print("sklearn is not installed, skipping classification report")


# Simple heuristic lists for demonstration purposes. These are far from
# exhaustive but provide a lightweight stand‑in for the heavy‑weight
# implementation found under ``web/src/diet_classifiers.py``.
_NON_KETO_TERMS = {
    "sugar",
    "flour",
    "rice",
    "bread",
    "pasta",
    "noodle",
    "potato",
    "honey",
    "corn",
    "bean",
    "juice",
    "syrup",
    "beer",
    "cake",
    "cookie",
    "biscuit",
}


def is_ingredient_keto(ingredient: str) -> bool:
    """Return ``True`` if the ingredient is considered keto friendly."""
    if not ingredient:
        return True

    text = ingredient.lower()
    return not any(term in text for term in _NON_KETO_TERMS)


_NON_VEGAN_TERMS = {
    "beef",
    "pork",
    "chicken",
    "fish",
    "egg",
    "cheese",
    "milk",
    "butter",
    "cream",
    "honey",
    "yogurt",
    "lamb",
    "gelatin",
}


def is_ingredient_vegan(ingredient: str) -> bool:
    """Return ``True`` if the ingredient is considered vegan."""
    if not ingredient:
        return True

    text = ingredient.lower()
    return not any(term in text for term in _NON_VEGAN_TERMS)


def is_keto(ingredients: List[str]) -> bool:
    return all(map(is_ingredient_keto, ingredients))


def is_vegan(ingredients: List[str]) -> bool:
    return all(map(is_ingredient_vegan, ingredients))


def _evaluate(ground_truth_path: str) -> int:
    ground_truth = pd.read_csv(ground_truth_path, index_col=None)
    try:
        start_time = time()
        ground_truth['keto_pred'] = ground_truth['ingredients'].apply(is_keto)
        ground_truth['vegan_pred'] = ground_truth['ingredients'].apply(
            is_vegan)

        end_time = time()
    except Exception as e:
        print(f"Error: {e}")
        return -1

    print("===Keto===")
    print(classification_report(
        ground_truth['keto'], ground_truth['keto_pred']))
    print("===Vegan===")
    print(classification_report(
        ground_truth['vegan'], ground_truth['vegan_pred']))
    print(f"== Time taken: {end_time - start_time} seconds ==")
    return 0


def _classify(ingredients: str) -> int:
    if ingredients.startswith('['):
        ing_list = json.loads(ingredients)
    else:
        ing_list = [i.strip() for i in ingredients.split(',') if i.strip()]

    keto = is_keto(ing_list)
    vegan = is_vegan(ing_list)
    print(json.dumps({'keto': keto, 'vegan': vegan}))
    return 0


def main(args):
    if args.ingredients:
        return _classify(args.ingredients)
    else:
        return _evaluate(args.ground_truth)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ground_truth", type=str,
                        default="/usr/src/data/ground_truth_sample.csv",
                        help="Path to CSV for evaluation")
    parser.add_argument("--ingredients", type=str,
                        help="Comma separated ingredients for quick classification")
    sys.exit(main(parser.parse_args()))
