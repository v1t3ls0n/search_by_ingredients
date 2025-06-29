#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Domain-specific constants for diet classification.

Based on original lines 1399-1758 from diet_classifiers.py
This version uses lazy compilation for regex patterns to improve startup time.
"""

import re
from functools import lru_cache
from typing import Pattern, Dict

# High-carb ingredients that disqualify keto classification
NON_KETO = list(set([
    # High-carb fruits
    "apple", "banana", "orange", "grape", "kiwi", "mango", "peach",
    "strawberry", "pineapple", "apricot", "tangerine", "persimmon",
    "pomegranate", "prune", "papaya", "jackfruit",

    # Grains and grain products
    "white rice", "long grain rice", "cornmeal", "corn",
    "all-purpose flour", "bread", "pasta", "couscous",
    "bulgur", "quinoa", "barley flour", "buckwheat flour",
    "durum wheat flour", "wheat flour", "whole-wheat flour",
    "oat flour", "oatmeal", "rye flour", "semolina",
    "amaranth", "millet", "sorghum flour", "sorghum grain",
    "spelt flour", "teff grain", "triticale",
    "einkorn flour", "emmer grain", "fonio", "freekeh",
    "kamut flour", "farina",

    # Starchy vegetables
    "potato", "baking potato", "potato wedge", "potato slice",
    "russet potato", "sweet potato", "yam", "cassava",
    "taro", "lotus root", "water chestnut", "ube",

    # Legumes
    "kidney bean", "black bean", "pinto bean", "navy bean",
    "lima bean", "cannellini bean", "great northern bean",
    "garbanzo bean", "chickpea", "adzuki bean", "baked bean",
    "refried bean", "hummus",

    # Sweeteners and sugars
    "sugar", "brown sugar", "coconut sugar", "muscovado sugar",
    "demerara", "turbinado", "molasses", "honey", "agave nectar",
    "maple syrup",

    # Sauces and condiments high in sugar
    "tomato sauce", "ketchup", "bbq sauce", "teriyaki sauce",
    "hoisin sauce", "sweet chili sauce", "sweet pickle",
    "sweet relish", "sweet soy glaze", "marmalade",

    # Processed foods and snacks
    "bread", "bagel", "muffin", "cookie", "cooky", "cake",
    "pastry", "pie crust", "pizza", "pizza crust", "pizza flour",
    "naan", "pita", "roti", "chapati", "tortilla",
    "pretzel", "chip", "french fry", "tater tot",
    "doughnut", "graham cracker", "hamburger bun", "hot-dog bun",

    # Breakfast items
    "breakfast cereal", "granola", "muesli", "energy bar",

    # Beverages
    "soy sauce", "orange juice", "fruit punch", "chocolate milk",
    "sweetened condensed milk", "sweetened cranberry", "sweetened yogurt",

    # Alcoholic beverages (carbs from alcohol and mixers)
    "ale", "beer", "lager", "ipa", "pilsner", "stout", "porter",
    "moscato", "riesling", "port", "sherry", "sangria",
    "margarita", "mojito", "pina colada", "daiquiri", "mai tai",
    "cosmopolitan", "whiskey sour", "bloody mary",
    "bailey", "kahlua", "amaretto", "frangelico", "limoncello",
    "triple sec", "curacao", "alcoholic lemonade", "alcopop",
    "breezer", "smirnoff ice", "mike hard lemonade", "hard cider", "cider",

    # Specialty items
    "tapioca", "arrowroot", "job's tear", "jobs tear", "job tear",
    "gnocchi", "tempura batter", "breading",
    "ice cream", "candy", "hard candy", "gummy bear",

    # Soy products (often sweetened)
    "soybean sweetened",
]))

# Animal-derived ingredients that disqualify vegan classification
NON_VEGAN = list(set([
    # Meat - Red meat
    'beef', 'steak', 'ribeye', 'sirloin', 'veal', 'lamb', 'mutton',
    'pork', 'bacon', 'ham', 'boar', 'goat', 'kid', 'venison',
    'rabbit', 'hare',

    # Meat - Poultry
    'chicken', 'turkey', 'duck', 'goose', 'quail', 'pheasant',
    'partridge', 'grouse',

    # Meat - Organ meats
    'liver', 'kidney', 'heart', 'tongue', 'brain', 'sweetbread',
    'tripe', 'gizzard', 'offal', 'bone', 'marrow', 'oxtail',

    # Meat - Processed
    'sausage', 'bratwurst', 'knackwurst', 'mettwurst',
    'salami', 'pepperoni', 'pastrami', 'bresaola',
    'prosciutto', 'pancetta', 'guanciale', 'speck',
    'mortadella', 'capocollo', 'coppa', 'cotechino',
    'chorizo', 'lard', 'tallow',

    # Fish and Seafood
    'fish', 'salmon', 'tuna', 'cod', 'haddock', 'halibut',
    'mackerel', 'herring', 'sardine', 'anchovy', 'trout',
    'tilapia', 'catfish', 'carp', 'sole', 'snapper', 'eel',
    'shrimp', 'prawn', 'crab', 'lobster', 'langoustine',
    'clam', 'mussel', 'oyster', 'scallop', 'squid', 'calamari',
    'octopus', 'krill', 'caviar', 'roe',
    'fishpaste', 'shrimppaste', 'anchovypaste', 'bonito',
    'katsuobushi', 'dashi', 'nampla',

    # Dairy - Milk products
    'milk', 'cream', 'butter', 'buttermilk', 'condensed', 'evaporated',
    'lactose', 'whey', 'casein', 'ghee', 'kefir',

    # Dairy - Cheese
    'cheese', 'cheddar', 'mozzarella', 'parmesan', 'parmigiano',
    'reggiano', 'pecorino', 'ricotta', 'mascarpone',
    'brie', 'camembert', 'roquefort', 'gorgonzola', 'stilton',
    'emmental', 'gruyere', 'fontina', 'asiago', 'manchego',
    'halloumi', 'feta', 'quark', 'paneer', 'stracciatella',
    'provolone', 'taleggio',

    # Dairy - Other
    'yogurt', 'sourcream', 'cremefraiche', 'curd', 'custard',
    'icecream', 'gelatin', 'collagen',

    # Eggs
    'egg', 'yolk', 'albumen', 'omelet', 'omelette', 'meringue',

    # Other animal products
    'honey', 'shellfish', 'escargot', 'snail', 'frog',
    'worcestershire', 'aioli', 'mayonnaise',
    'broth', 'stock', 'gravy',
]))

# Regex patterns for keto-friendly ingredients (overrides blacklist)
KETO_WHITELIST = [
    # Keto-friendly flours
    r"\balmond flour\b",
    r"\bcoconut flour\b",
    r"\bflaxseed flour\b",
    r"\bchia flour\b",
    r"\bsunflower seed flour\b",
    r"\bpeanut flour\b",
    r"\bhemp flour\b",
    r"\bsesame flour\b",
    r"\bwalnut flour\b",
    r"\bpecan flour\b",
    r"\bmacadamia flour\b",
    r"\bhazelnut flour\b",

    # Special exceptions for "kidney" (organ meat, not kidney beans)
    r"\bkidney\b",

    # Low-carb citrus exceptions
    r"\blemon juice\b",

    # Keto milk alternatives
    r"\balmond milk\b",
    r"\bcoconut milk\b",
    r"\bflax milk\b",
    r"\bmacadamia milk\b",
    r"\bhemp milk\b",
    r"\bcashew milk\b",
    r"\balmond cream\b",
    r"\bcoconut cream\b",
    r"\bsour cream\b",

    # Nut and seed butters
    r"\balmond butter\b",
    r"\bpeanut butter\b",
    r"\bcoconut butter\b",
    r"\bmacadamia butter\b",
    r"\bpecan butter\b",
    r"\bwalnut butter\b",
    r"\bhemp butter\b",

    # Keto bread alternatives
    r"\balmond bread\b",
    r"\bcoconut bread\b",
    r"\bcloud bread\b",
    r"\bketo bread\b",

    # Sugar-free sweeteners
    r"\bcoconut sugar[- ]free\b",
    r"\bstevia\b",
    r"\berytritol\b",
    r"\bmonk fruit\b",
    r"\bswerve\b",
    r"\ballulose\b",
    r"\bxylitol\b",
    r"\bsugar[- ]free\b",

    # Low-carb alternatives
    r"\bcauliflower rice\b",
    r"\bshirataki noodles\b",
    r"\bzucchini noodles\b",
    r"\bkelp noodles\b",
    r"\bsugar[- ]free chocolate\b",
    r"\bketo chocolate\b",

    # Low-carb vegetables and foods
    r"\bavocado\b",
    r"\bcacao\b",
    r"\bcocoa powder\b",
    r"\bketo ice[- ]cream\b",
    r"\bsugar[- ]free ice[- ]cream\b",
    r"\bjicama\b",
    r"\bzucchini\b",
    r"\bcucumber\b",
    r"\bbroccoli\b",
    r"\bcauliflower\b",
]

# Regex patterns for vegan-friendly ingredients (overrides blacklist)
VEGAN_WHITELIST = [
    # Egg exceptions (plant-based)
    r"\beggplant\b",
    r"\begg\s*fruit\b",
    r"\bvegan\s+egg\b",

    # Milk exceptions (plant-based)
    r"\bmillet\b",  # grain, not milk
    r"\bmilk\s+thistle\b",
    r"\bcoconut\s+milk\b",
    r"\boat\s+milk\b",
    r"\bsoy\s+milk\b",
    r"\balmond\s+milk\b",
    r"\bcashew\s+milk\b",
    r"\brice\s+milk\b",
    r"\bhazelnut\s+milk\b",
    r"\bpea\s+milk\b",

    # Rice alternatives (vegetable-based)
    r"\bcauliflower rice\b",
    r"\bbroccoli rice\b",
    r"\bsweet potato rice\b",
    r"\bzucchini rice\b",
    r"\bcabbage rice\b",
    r"\bkonjac rice\b",
    r"\bshirataki rice\b",
    r"\bmiracle rice\b",
    r"\bpalmini rice\b",

    # Butter exceptions (plant-based)
    r"\bbutternut\b",  # squash
    r"\bbutterfly\s+pea\b",
    r"\bcocoa\s+butter\b",
    r"\bpeanut\s+butter\b",
    r"\balmond\s+butter\b",
    r"\bsunflower(?:\s*seed)?\s+butter\b",
    r"\bpistachio\s+butter\b",
    r"\bvegan\s+butter\b",

    # Honey exceptions (plants)
    r"\bhoneydew\b",
    r"\bhoneysuckle\b",
    r"\bhoneycrisp\b",
    r"\bhoney\s+locust\b",
    r"\bhoneyberry\b",

    # Cream exceptions (plant-based)
    r"\bcream\s+of\s+tartar\b",
    r"\bice[- ]cream\s+bean\b",
    r"\bcoconut\s+cream\b",
    r"\bcashew\s+cream\b",
    r"\bvegan\s+cream\b",

    # Cheese exceptions (plant-based)
    r"\bcheesewood\b",
    r"\bvegan\s+cheese\b",
    r"\bcashew\s+cheese\b",

    # Fish exceptions (plants)
    r"\bfish\s+mint\b",
    r"\bfish\s+pepper\b",

    # Beef exceptions (plants/mushrooms)
    r"\bbeefsteak\s+plant\b",
    r"\bbeefsteak\s+mushroom\b",

    # Chicken/hen exceptions (mushrooms)
    r"\bchicken[- ]of[- ]the[- ]woods\b",
    r"\bchicken\s+mushroom\b",
    r"\bhen[- ]of[- ]the[- ]woods\b",

    # Meat exceptions (plants)
    r"\bsweetmeat\s+(?:pumpkin|squash)\b",

    # Bacon alternatives
    r"\bcoconut\s+bacon\b",
    r"\bmushroom\s+bacon\b",
    r"\bsoy\s+bacon\b",
    r"\bvegan\s+bacon\b",
]


def compile_any(words: list[str]) -> Pattern[str]:
    """
    Compile a list of words into a single regex pattern for efficient matching.
    
    Based on original line 1745 from diet_classifiers.py
    """
    return re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, words)), re.I)


@lru_cache(maxsize=1)
def get_keto_patterns() -> Dict[str, Pattern[str]]:
    """
    Get compiled keto patterns with lazy initialization.
    
    This replaces the global RX_KETO and RX_WL_KETO from original lines 1754-1755
    """
    from ..core import log
    log.debug("Compiling keto regex patterns...")
    
    return {
        'blacklist': compile_any(NON_KETO),
        'whitelist': re.compile("|".join(KETO_WHITELIST), re.I)
    }


@lru_cache(maxsize=1)
def get_vegan_patterns() -> Dict[str, Pattern[str]]:
    """
    Get compiled vegan patterns with lazy initialization.
    
    This replaces the global RX_VEGAN and RX_WL_VEGAN from original lines 1756-1757
    """
    from ..core import log
    log.debug("Compiling vegan regex patterns...")
    
    return {
        'blacklist': compile_any(NON_VEGAN),
        'whitelist': re.compile("|".join(VEGAN_WHITELIST), re.I)
    }