

from utils.classifiers import is_ingredient_keto, is_ingredient_vegan
from utils.scores import _diet_score_portion_aware, _diet_score_simple
from utils.parser import parse_ingredients


def is_keto(ingredients):
    if isinstance(ingredients, str):
        ingredients = parse_ingredients(ingredients)
    return all(map(is_ingredient_keto, ingredients))


def is_vegan(ingredients):
    if isinstance(ingredients, str):
        ingredients = parse_ingredients(ingredients)
    return all(map(is_ingredient_vegan, ingredients))


def get_diet_statistics(recipes):
    """Return diet compliance stats for recipe set"""
    return {
        'total': len(recipes),
        'keto': sum(1 for r in recipes if is_keto(r['ingredients'])),
        'vegan': sum(1 for r in recipes if is_vegan(r['ingredients'])),
        'both': sum(1 for r in recipes if is_keto(r['ingredients']) and is_vegan(r['ingredients']))
    }

def diet_score(
    ingredients: list[str],
    diet: str,
    *,
    portion_aware: bool = True
) -> int:
    """
    Public façade for recipe-level percentage score.

    Parameters
    ----------
    ingredients : list[str] | str
        Raw ingredient strings or the strange CSV-ish string from allrecipes.
    diet : {"keto","vegan"}
    portion_aware : bool, default True
        • True  -> use 1-g net-carb overflow logic (keto only).  
        • False -> simple count ratio (no USDA, no units).

    Returns
    -------
    int  between 0 and 100

    Examples
    --------
    >>> diet_score(["almond flour","egg"], "keto")          # portion-aware
    100
    >>> diet_score(["almond flour","1 tsp sugar"], "keto")
    100
    >>> diet_score(["almond flour","1 cup sugar"], "keto")
    50
    >>> diet_score(["almond flour","1 tsp sugar"], "keto", portion_aware=False)
    50
    """
    if isinstance(ingredients, str):
        ingredients = parse_ingredients(ingredients)

    # Convert numpy array to list if needed
    if hasattr(ingredients, 'tolist'):
        ingredients = ingredients.tolist()
    elif not isinstance(ingredients, list):
        ingredients = list(ingredients)
        
    if portion_aware:
        return _diet_score_portion_aware(ingredients, diet)
    
    return _diet_score_simple(ingredients, diet)
