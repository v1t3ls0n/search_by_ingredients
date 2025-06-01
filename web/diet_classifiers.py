import json
from argparse import ArgumentParser
from typing import List
from time import time
try:
    from sklearn.metrics import classification_report
except ImportError:
    # sklearn is optional
    def classification_report(y, y_pred):
        print("sklearn is not installed, skipping classification report")

def is_ingredient_keto(ingredient: str) -> bool:
    return False

def is_ingredient_vegan(ingredient: str) -> bool:
    return False

def is_keto(ingredients: List[str]) -> bool:
    return all(map(is_ingredient_keto, ingredients))

def is_vegan(ingredients: List[str]) -> bool:
    return all(map(is_ingredient_vegan, ingredients))

def main(args):
    with open(args.ground_truth, "r") as f:
        ground_truth = json.load(f)
    start_time = time()
    keto_pred, vegan_pred, keto_gt, vegan_gt = [], [], [], []
    for ingredient,v in ground_truth.items():
        keto_pred.append(is_ingredient_keto(ingredient))
        vegan_pred.append(is_ingredient_vegan(ingredient))
        keto_gt.append(v['keto'])
        vegan_gt.append(v['vegan'])
    end_time = time()
    print("===Keto===")
    print(classification_report(keto_gt, keto_pred))
    print("===Vegan===")
    print(classification_report(vegan_gt, vegan_pred))
    print(f"== Time taken: {end_time - start_time} seconds ==")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ground_truth", type=str, default="web/data/ingredients.json")
    main(parser.parse_args())