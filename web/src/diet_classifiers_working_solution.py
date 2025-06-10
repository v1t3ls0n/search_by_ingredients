#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Argmax ingredient-diet pipeline — v3
Hard-verify every model’s output with blacklist / whitelist rules
"""


from __future__ import annotations
from utils.non_starter_tokens import NON_VEGAN, NON_KETO, VEGAN_WHITELIST, KETO_WHITELIST
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_curve
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import SVC
import nltk
import json
import logging
import re
import unicodedata
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
nltk.download('wordnet')
nltk.download('omw-1.4')  # Optional but helps with lemmatization

# optional LightGBM
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None                                              # type: ignore

# domain hardcoded lists (heuristics)
NON_KETO = list(set([
    "apple", "banana", "orange", "grape", "kiwi", "mango", "peach",
    "apple", "white rice", "long grain rice", "cornmeal",
    "baking potato", "potato wedge", "potato slice", "russet potato",
    "potato", "potato wedge", "potato slice", "russet potato",
    "tomato sauce",
    "strawberry", "banana", "pineapple", "grape",
    "soy sauce", "teriyaki sauce", "hoisin sauce", "ketchup", "bbq sauce",
    "orange juice", "agave nectar",
    'pizza flour',
    'adzuki bean',
    'alcoholic lemonade',
    'alcopop',
    'ale',
    'all-purpose flour',
    'amaranth',
    'amaretto',
    'apricot',
    'arrowroot',
    'bagel',
    'bailey',
    'baked bean',
    'barley flour',
    'bbq sauce',
    'bloody mary',
    'bread',
    'breading',
    'breakfast cereal',
    'breezer',
    'brown sugar',
    'buckwheat flour',
    'buckwheat groat',
    'bulgur',
    'cake',
    'cannellini bean',
    'cassava',
    'chapati',
    'chip',
    'chocolate milk',
    'chutney',
    'coconut sugar',
    'cooky',
    'cosmopolitan',
    'couscous',
    'curacao',
    'daiquiri',
    'demerara',
    'doughnut',
    'durum wheat flour',
    'einkorn flour',
    'emmer grain',
    'energy bar',
    'farina',
    'fonio',
    'frangelico',
    'freekeh',
    'french fry',
    'fruit punch',
    'garbanzo bean',
    'gnocchi',
    'graham cracker',
    'granola',
    'great northern bean',
    'gummy bear',
    'hamburger bun',
    'hard candy',
    'candy',
    'hard cider',
    'cider',
    'hoisin sauce',
    'hot-dog bun',
    'hummus',
    'ice cream',
    'ipa',
    'jackfruit',
    'job’s tear',
    'jobs tear',
    'job tear',
    'kahlua',
    'kamut flour',
    'ketchup',
    'lager',
    'lima bean',
    'limoncello',
    'lotus root',
    'mai tai',
    'margarita',
    'marmalade',
    'mike hard lemonade',
    'millet',
    'mojito',
    'molasses',
    'moscato',
    'muesli',
    'muffin',
    'muscovado sugar',
    'naan',
    'navy bean',
    'oat flour',
    'oatmeal',
    'papaya',
    'pasta',
    'pastry',
    'persimmon',
    'pie crust',
    'pilsner',
    'pina colada',
    'pinto bean',
    'pita',
    'pizza crust',
    'pizza',
    'pomegranate',
    'port',
    'porter',
    'pretzel',
    'prune',
    'quinoa',
    'refried bean',
    'riesling',
    'roti',
    'rye flour',
    'sangria',
    'semolina',
    'sherry',
    'smirnoff ice',
    'sorghum flour',
    'sorghum grain',
    'soybean sweetened',
    'spelt flour',
    'stout',
    'sugar',
    'sweet chili sauce',
    'sweet pickle',
    'sweet relish',
    'sweet soy glaze',
    'sweetened condensed milk',
    'sweetened cranberry',
    'sweetened yogurt',
    'tangerine',
    'tapioca',
    'taro',
    'tater tot',
    'teff grain',
    'tempura batter',
    'teriyaki sauce',
    'tortilla',
    'triple sec',
    'triticale',
    'turbinado',
    'ube',
    'water chestnut',
    'wheat flour',
    'whiskey sour',
    'whole-wheat flour',
]))

NON_VEGAN = list(set([
    'aioli',
    'albumen',
    'anchovy',
    'anchovypaste',
    'asiago',
    'bacon',
    'beef',
    'boar',
    'bone',
    'bonito',
    'bratwurst',
    'bresaola',
    'brie',
    'broth',
    'butter',
    'buttermilk',
    'calamari',
    'camembert',
    'capocollo',
    'carp',
    'casein',
    'catfish',
    'caviar',
    'cheddar',
    'cheese',
    'chicken',
    'chorizo',
    'clam',
    'cod',
    'collagen',
    'condensed',
    'coppa',
    'cotechino',
    'crab',
    'cream',
    'cremefraiche',
    'curd',
    'custard',
    'dashi',
    'duck',
    'eel',
    'egg',
    'emmental',
    'escargot',
    'evaporated',
    'feta',
    'fish',
    'fishpaste',
    'fontina',
    'frog',
    'gelatin',
    'ghee',
    'gizzard',
    'goat',
    'goose',
    'gorgonzola',
    'gravy',
    'grouse',
    'gruyere',
    'guanciale',
    'haddock',
    'halibut',
    'halloumi',
    'ham',
    'hare',
    'heart',
    'herring',
    'honey',
    'icecream',
    'katsuobushi',
    'kefir',
    'kid',
    'kidney',
    'knackwurst',
    'krill',
    'lactose',
    'lamb',
    'langoustine',
    'lard',
    'liver',
    'lobster',
    'mackerel',
    'manchego',
    'mascarpone',
    'mayonnaise',
    'meringue',
    'mettwurst',
    'milk',
    'mortadella',
    'mozzarella',
    'mussel',
    'mutton',
    'nampla',
    'octopus',
    'offal',
    'omelet',
    'omelette',
    'oxtail',
    'oyster',
    'pancetta',
    'paneer',
    'parmesan',
    'parmigiano',
    'partridge',
    'pastrami',
    'pecorino',
    'pepperoni',
    'pheasant',
    'pollock',
    'pork',
    'prawn',
    'prosciutto',
    'provolone',
    'quail',
    'quark',
    'rabbit',
    'reggiano',
    'ribeye',
    'ricotta',
    'roe',
    'roquefort',
    'salami',
    'salmon',
    'sardine',
    'sausage',
    'scallop',
    'scampi',
    'shellfish',
    'shrimp',
    'shrimppaste',
    'sirloin',
    'snail',
    'snapper',
    'sole',
    'sourcream',
    'speck',
    'squid',
    'steak',
    'stilton',
    'stock',
    'stracciatella',
    'sweetbread',
    'taleggio',
    'tallow',
    'tilapia',
    'tongue',
    'tripe',
    'trout',
    'tuna',
    'turkey',
    'veal',
    'venison',
    'whey',
    'worcestershire',
    'yogurt',
    'yolk',
]))

KETO_WHITELIST = [
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
    r"\blemon juice\b",

    r"\balmond milk\b",
    r"\bcoconut milk\b",
    r"\bflax milk\b",
    r"\bmacadamia milk\b",
    r"\bhemp milk\b",
    r"\bcashew milk\b",
    r"\balmond cream\b",
    r"\bcoconut cream\b",

    r"\balmond butter\b",
    r"\bpeanut butter\b",
    r"\bcoconut butter\b",
    r"\bmacadamia butter\b",
    r"\bpecan butter\b",
    r"\bwalnut butter\b",
    r"\bhemp butter\b",

    r"\balmond bread\b",
    r"\bcoconut bread\b",
    r"\bcloud bread\b",
    r"\bketo bread\b",

    r"\bcoconut sugar[- ]free\b",
    r"\bstevia\b",
    r"\berytritol\b",
    r"\bmonk fruit\b",
    r"\bswerve\b",
    r"\ballulose\b",
    r"\bxylitol\b",
    r"\bsugar[- ]free\b",

    r"\bcauliflower rice\b",
    r"\bshirataki noodles\b",
    r"\bzucchini noodles\b",
    r"\bkelp noodles\b",

    r"\bsugar[- ]free chocolate\b",
    r"\bketo chocolate\b",
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
VEGAN_WHITELIST = [
    # — egg —
    r"\beggplant\b",
    r"\begg\s*fruit\b",                 # aka biribá
    r"\bvegan\s+egg\b",

    # — milk —
    r"\bmillet\b",
    r"\bmilk\s+thistle\b",
    r"\bcoconut\s+milk\b",
    r"\boat\s+milk\b",
    r"\bsoy\s+milk\b",
    r"\balmond\s+milk\b",
    r"\bcashew\s+milk\b",
    r"\brice\s+milk\b",
    r"\bhazelnut\s+milk\b",
    r"\bpea\s+milk\b",

    # — rice —
    r"\bcauliflower rice\b",
    r"\bbroccoli rice\b",
    r"\bsweet potato rice\b",
    r"\bzucchini rice\b",
    r"\bcabbage rice\b",
    r"\bkonjac rice\b",
    r"\bshirataki rice\b",
    r"\bmiracle rice\b",
    r"\bpalmini rice\b",
    # — butter —
    r"\bbutternut\b",
    r"\bbutterfly\s+pea\b",
    r"\bcocoa\s+butter\b",
    r"\bpeanut\s+butter\b",
    r"\balmond\s+butter\b",
    r"\bsunflower(\s*seed)?\s+butter\b",
    r"\bpistachio\s+butter\b",
    r"\bvegan\s+butter\b",

    # — honey —
    r"\bhoneydew\b",
    r"\bhoneysuckle\b",
    r"\bhoneycrisp\b",
    r"\bhoney\s+locust\b",
    r"\bhoneyberry\b",

    # — cream —
    r"\bcream\s+of\s+tartar\b",
    r"\bice[- ]cream\s+bean\b",
    r"\bcoconut\s+cream\b",
    r"\bcashew\s+cream\b",
    r"\bvegan\s+cream\b",

    # — cheese —
    r"\bcheesewood\b",
    r"\bvegan\s+cheese\b",
    r"\bcashew\s+cheese\b",

    # — fish —
    r"\bfish\s+mint\b",
    r"\bfish\s+pepper\b",

    # — beef —
    # r"\bbeefsteak\s+tomato\b",
    r"\bbeefsteak\s+plant\b",
    r"\bbeefsteak\s+mushroom\b",

    # — chicken / hen —
    r"\bchicken[- ]of[- ]the[- ]woods\b",
    r"\bchicken\s+mushroom\b",
    r"\bhen[- ]of[- ]the[- ]woods\b",

    # — meat —
    r"\bsweetmeat\s+(pumpkin|squash)\b",

    # — bacon —
    r"\bcoconut\s+bacon\b",
    r"\bmushroom\s+bacon\b",
    r"\bsoy\s+bacon\b",
    r"\bvegan\s+bacon\b",
]


# ───────────────────────── Logging ─────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)s │ %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("PIPE")

# ─────────────── Config ───────────────


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("dataset/arg_max")
    url_map: Mapping[str, str] = field(default_factory=lambda: {
        "allrecipes.parquet":
        "https://argmax.nyc3.digitaloceanspaces.com/recipes/allrecipes.parquet",
        "ground_truth_sample.csv":
        "https://argmax.nyc3.digitaloceanspaces.com/recipes/ground_truth_sample.csv",
    })
    vec_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
        min_df=2, ngram_range=(1, 3), max_features=50000, sublinear_tf=True))


CFG = Config()


def validate_keto_prediction(texts, preds, task, min_hits=2):
    """Override keto predictions using rule-based ingredient matching."""
    if task != "keto":
        return preds
    fixed_preds = []
    for text, pred in zip(texts, preds):
        hits = find_non_keto_hits(text)
        if pred == 1 and len(hits) >= min_hits:
            fixed_preds.append(0)
        else:
            fixed_preds.append(pred)
    return np.array(fixed_preds)

# Basic tokenizer (split on non-word characters, lowercase)


def tokenize_ingredient(text: str) -> list[str]:
    return re.findall(r"\b\w[\w-]*\b", text.lower())


# Check if any known non-keto ingredient appears in the tokens
def is_keto_ingredient_list(tokens: list[str]) -> bool:
    for ingredient in NON_KETO:
        ing_tokens = ingredient.split()
        if all(tok in tokens for tok in ing_tokens):
            return False  # matched a known high-carb ingredient
    return True


# Find non-keto ingredient hits (returns list of matching strings)
def find_non_keto_hits(text: str) -> list[str]:
    tokens = set(tokenize_ingredient(text))
    return sorted([
        ingredient for ingredient in NON_KETO
        if all(tok in tokens for tok in ingredient.split())
    ])


# Override predictions using rule-based check
def validate_keto_prediction(texts, preds, task):
    """Override keto predictions using rule-based ingredient matching."""
    if task != "keto":
        return preds
    fixed_preds = []
    for text, pred in zip(texts, preds):
        if pred == 1 and find_non_keto_hits(text):
            fixed_preds.append(0)  # flip to non-keto
        else:
            fixed_preds.append(pred)
    return np.array(fixed_preds)


# ─────────────── Regex helpers ───────────────
def compile_any(words: Iterable[str]) -> re.Pattern[str]:
    return re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, words)), re.I)


RX_KETO = compile_any(NON_KETO)
RX_VEGAN = compile_any(NON_VEGAN)
RX_WL_VEGAN = compile_any(VEGAN_WHITELIST)


# ─────────────── Normalization ───────────────
_LEMM = WordNetLemmatizer()
_UNITS = re.compile(r"\b(?:g|gram|kg|oz|ml|l|cup|cups|tsp|tbsp|teaspoon|"
                    r"tablespoon|pound|lb|slice|slices|small|large|medium)\b")


def normalise(t: str) -> str:
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    t = re.sub(r"\([^)]*\)", " ", t.lower())
    t = _UNITS.sub(" ", t)
    t = re.sub(r"\d+(?:[/\.]\d+)?", " ", t)
    t = re.sub(r"[^\w\s-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return " ".join(_LEMM.lemmatize(w) for w in t.split() if len(w) > 2)
# ─────────────── RuleModel (unchanged) ───────────────


class RuleModel(BaseEstimator, ClassifierMixin):
    def __init__(self, task: str, rx_black, rx_white=None,
                 pos_prob=0.98, neg_prob=0.02):
        self.task, self.rx_black, self.rx_white = task, rx_black, rx_white
        self.pos_prob, self.neg_prob = pos_prob, neg_prob

    def fit(self, X, y=None): return self

    def _pos(self, d: str) -> bool:
        return (not bool(self.rx_black.search(d)) if self.task == "keto"
                else not bool(self.rx_black.search(d)) or
                bool(self.rx_white and self.rx_white.search(d)))

    def predict_proba(self, X):
        p = np.fromiter((self.pos_prob if self._pos(d) else self.neg_prob
                         for d in X), float, count=len(X))
        return np.c_[1-p, p]

    def predict(self, X): return (
        self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# ─────────────── Verification layer ───────────────


def verify_with_rules(task: str, clean: pd.Series, prob: np.ndarray) -> np.ndarray:
    adjusted = prob.copy()

    if task == "keto":
        # Regex-based whitelist/blacklist
        is_whitelisted = clean.str.contains(compile_any(KETO_WHITELIST))
        is_blacklisted = clean.str.contains(RX_KETO)
        forced_non_keto = is_blacklisted & ~is_whitelisted
        adjusted[forced_non_keto.values] = 0.0

        # Token-based ingredient verification
        for i, txt in enumerate(clean):
            if adjusted[i] > 0.5:  # Only inspect items likely predicted as keto
                tokens = tokenize_ingredient(normalise(txt))
                if not is_keto_ingredient_list(tokens):
                    adjusted[i] = 0.0
                    log.debug("Heuristically rejected '%s' as non-keto", txt)

        if forced_non_keto.any():
            log.debug("Keto Verification: forced %d probs to 0 (regex)",
                      forced_non_keto.sum())

    else:  # vegan logic remains unchanged
        bad = clean.str.contains(RX_VEGAN) & ~clean.str.contains(RX_WL_VEGAN)
        adjusted[bad.values] = 0.0
        if bad.any():
            log.debug("Vegan Verification: forced %d probs to 0", bad.sum())

    return adjusted

# ─────────────── Data I/O ───────────────


def download_raw():
    CFG.data_dir.mkdir(parents=True, exist_ok=True)
    for f, u in CFG.url_map.items():
        dst = CFG.data_dir/f
        if not dst.exists():
            log.info("⬇️ %s", f)
            urllib.request.urlretrieve(u, dst)


def parquet_to_csv() -> Path:
    pq, csv = CFG.data_dir/"allrecipes.parquet", CFG.data_dir/"allrecipes.csv"
    if csv.exists():
        return csv
    if not pq.exists():
        raise FileNotFoundError("Run download_raw first")
    log.info("Parquet → CSV")
    pd.read_parquet(pq).to_csv(csv, index=False)
    return csv

# ─────────────── Silver labels ───────────────


def build_silver(csv: Path) -> pd.DataFrame:
    sk, sv = CFG.data_dir/"silver_keto.csv", CFG.data_dir/"silver_vegan.csv"
    if sk.exists() and sv.exists():
        df = pd.read_csv(sk)
        df["silver_vegan"] = pd.read_csv(sv).silver_vegan
        df["clean"] = df.clean.fillna("").astype(str)
        return df
    df = pd.read_csv(csv, usecols=["ingredients"])
    df["clean"] = df.ingredients.fillna("").map(normalise)
    df["silver_keto"] = (~df.clean.str.contains(RX_KETO)).astype(int)
    bad = (df.clean.str.contains(RX_VEGAN) & ~
           df.clean.str.contains(RX_WL_VEGAN))
    df["silver_vegan"] = (~bad).astype(int)
    sk.parent.mkdir(parents=True, exist_ok=True)
    df[["clean", "silver_keto"]].to_csv(sk, index=False)
    df[["silver_vegan"]].to_csv(sv, index=False)
    return df

# ─────────────────── Class-balance helper (robust) ────────────────────


def show_balance(df: pd.DataFrame, title: str) -> None:
    """Print positives / total for whichever keto / vegan column exists."""
    print(f"\n── {title} set class counts ─────────────")
    for lab in ("keto", "vegan"):
        # pick the first matching column
        for col in (f"label_{lab}", f"silver_{lab}"):
            if col in df.columns:
                pos, tot = int(df[col].sum()), len(df)
                print(f"{lab:>5}: {pos:6}/{tot} ({pos/tot:>5.1%})")
                break


# ─────────────── Model registry ───────────────


def build_models(task: str) -> Dict[str, BaseEstimator]:
    m = {
        "Rule": RuleModel("keto", RX_KETO, compile_any(KETO_WHITELIST)) if task == "keto" else
        RuleModel("vegan", RX_VEGAN, RX_WL_VEGAN),
        "NB": MultinomialNB(),
        "PA": PassiveAggressiveClassifier(max_iter=1000, class_weight="balanced", random_state=42),
        "Ridge": RidgeClassifier(class_weight="balanced", random_state=42),
        "LR": LogisticRegression(solver="lbfgs"),
        "SGD": SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, class_weight="balanced", n_jobs=-1),
        # Optional: Simulated RBF-SGD
        # "RBF_SGD": make_pipeline(RBFSampler(gamma=1.0, random_state=42),
        #                          SGDClassifier(loss="log_loss", max_iter=1000, class_weight="balanced", random_state=42)),
        # "MLP": MLPClassifier((40,), alpha=1e-3, max_iter=100, random_state=42),
        # "LGBM": lgb.LGBMClassifier(
        #     num_leaves=15,
        #     learning_rate=0.3,
        #     n_estimators=50,
        #     subsample=0.7,
        #     colsample_bytree=0.7,
        #     objective="binary",
        #     n_jobs=-1,
        #     random_state=42
        # )
    }
    return m


HYPER = {
    "LR": {"C": [0.2, 1, 5], "class_weight": [None, "balanced"]},
    "SGD": {"estimator__alpha": [1e-4, 1e-3]},
    "MLP": {"hidden_layer_sizes": [(40,), (80,), (80, 40)],
            "alpha": [1e-4, 1e-3]},
    "LGBM": {"learning_rate": [0.05, 0.1], "num_leaves": [31, 63],
             "n_estimators": [200, 400]},
    "PA": {"C": [0.1, 0.5, 1.0]},
    "Ridge": {"alpha": [0.1, 1.0, 10.0]},
    "SVC": {  # Leave for future if you re-add SVC
        "C": [0.1, 1, 5],
        "kernel": ["linear", "rbf"],
        "class_weight": ["balanced"]
    },
    "NB": {}
}


BEST: Dict[str, BaseEstimator] = {}
FAST = True
CV = 2 if FAST else 3
N_IT = 2 if FAST else 6


def tune(name, model, X, y):
    """Tune hyperparameters for models using GridSearchCV where relevant."""
    if name in BEST:
        return BEST[name]

    S, kw = model.__class__, model.get_params()

    if name in ["NB", "PA", "Ridge", "SVC", "LR", "SGD"]:
        # Define the grid specific to each model
        if name == "SGD":
            grid = {"alpha": [1e-4, 1e-3]}
        elif name == "LR":
            grid = {"C": [0.1, 1, 10], "solver": ["liblinear"]}
        elif name == "Ridge":
            grid = {"alpha": [0.1, 1.0, 10.0]}
        elif name == "SVC":
            grid = {"C": [0.1, 1, 10], "kernel": ["linear"]}
        elif name == "PA":
            grid = {"C": [0.1, 1, 10]}
        elif name == "NB":
            grid = {"alpha": [0.5, 1.0, 1.5]}

        try:
            grid_search = GridSearchCV(
                S(**kw), grid, scoring="f1", n_jobs=-1, cv=3)
            grid_search.fit(X, y)
            BEST[name] = grid_search.best_estimator_
        except ValueError as e:
            print(f"[WARN] GridSearch failed for model '{name}' due to: {e}")
            print("→ Using default model instead.")
            BEST[name] = model
    else:
        # No tuning for rule-based or unsupported models
        BEST[name] = model

    return BEST[name]

# ─────────────── Metrics / table ───────────────


def pack(y, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(ACC=accuracy_score(y, pred),
                PREC=precision_score(y, pred, zero_division=0),
                REC=recall_score(y, pred, zero_division=0),
                F1=f1_score(y, pred, zero_division=0),
                ROC=roc_auc_score(y, prob),
                PR=average_precision_score(y, prob))


def table(title, rows):
    cols = ("ACC", "PREC", "REC", "F1", "ROC", "PR")
    pad = 11+8*len(cols)
    hdr = "│ model task "+" ".join(f"{c:>7}" for c in cols)+" │"
    print(f"\n╭─ {title} {'─'*(pad-len(title)-2)}")
    print(hdr)
    print("├"+"─"*(len(hdr)-2)+"┤")
    for r in rows:
        vals = " ".join(f"{r[c]:>7.2f}" for c in cols)
        print(f"│ {r['model']:<7} {r['task']:<5} {vals} │")
    print("╰"+"─"*(len(hdr)-2)+"╯")

# ─────────────── Mode A ───────────────


def run_mode_A(X_vec, clean, X_gold, silver, gold):
    res = []
    for task, col in [("keto", "silver_keto"), ("vegan", "silver_vegan")]:
        ys, yt = silver[col].values, gold[f"label_{task}"].values
        X_os, y_os = RandomOverSampler(random_state=42).fit_resample(X_vec, ys)
        show_balance(pd.DataFrame({col: y_os}),
                     f"Oversampled {task.capitalize()}")

        import time
        for name, base in tqdm(build_models(task).items(), desc=f"A/{task}"):
            t0 = time.time()

            if name == "Rule":
                mdl = base
                prob = mdl.predict_proba(clean)[:, 1]
            else:
                mdl = tune(name, base, X_os, y_os).fit(X_os, y_os)
                prob = (mdl.predict_proba(X_gold)[:, 1]
                        if hasattr(mdl, "predict_proba")
                        else mdl.decision_function(X_gold))

            prob = verify_with_rules(task, gold.clean, prob)
            r = pack(yt, prob) | {"model": name, "task": task}
            print(f"\n{name:<7} {task:<5} " + " ".join(f"{m}:{r[m]:>6.2f}" for m in ("ACC", "PREC", "REC", "F1", "ROC", "PR")),
                  f"⏱ {time.time() - t0:.1f}s")
            res.append(r)

    table("MODE A  (silver → gold)", res)
    return res


class RuleWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, score_fn):
        self.score_fn = score_fn

    def fit(self, X, y=None):
        return self  # no training needed

    def predict(self, X):
        return (self.score_fn(X) > 0.5).astype(int)

    def predict_proba(self, X):
        prob = np.clip(self.score_fn(X), 0, 1)
        return np.stack([1 - prob, prob], axis=1)


# ─────────────── Top-2 Ensemble (robust) ───────────────


def tune_threshold(y_true, probs):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1 = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.argmax(f1)
    return thresholds[optimal_idx]


def log_false_preds(task, texts, y_true, y_pred, model_mame="Some Model"):
    # False Positives: predicted 1 but actually 0
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    df_fp = pd.DataFrame({
        "Text": texts.iloc[fp_indices],
        "True_Label": y_true[fp_indices],
        "Predicted_Label": y_pred[fp_indices],
        "Error_Type": "False Positive",
        "Task": task
    })
    fp_path = f"false_positives_{task}_{model_mame}.csv"
    df_fp.to_csv(fp_path, index=False)
    log.info(
        f"""Logged {len(df_fp)} false positives to {fp_path} - on {model_mame}""")

    # False Negatives: predicted 0 but actually 1
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    df_fn = pd.DataFrame({
        "Text": texts.iloc[fn_indices],
        "True_Label": y_true[fn_indices],
        "Predicted_Label": y_pred[fn_indices],
        "Error_Type": "False Negative",
        "Task": task
    })
    fn_path = f"false_negatives_{task}_{model_mame}.csv"
    df_fn.to_csv(fn_path, index=False)
    log.info(
        f"""Logged {len(df_fn)} false negatives to {fn_path} - on {model_mame}""")


def best_ensemble(task, res, X_vec, clean, X_gold, silver, gold):
    """
    Tries ensembles with n = 1 to total available models (excluding 'Rule').
    Returns the best one based on F1 score.
    """
    # Count number of eligible models for ensemble (excluding 'Rule')
    model_names = [r["model"]
                   for r in res if r["task"] == task and r["model"] != "Rule"]
    max_n = len(set(model_names))

    best_score = -1
    best_result = None

    for n in range(1, max_n + 1):
        try:
            result = top_n(task, res, X_vec, clean, X_gold, silver, gold, n=n)
            if result["F1"] > best_score:
                best_score = result["F1"]
                best_result = result
        except Exception as e:
            print(f"[WARN] Ensemble n={n} failed: {e}")

    return best_result


def top_n(task, res, X_vec, clean, X_gold, silver, gold, n=3, use_saved_params=False, rule_weight=0):
    """
    Build an n-model ensemble on *task*:
    - Select top-n models based on combined metrics (PREC, REC, ROC, PR, F1, ACC)
    - Logs false predictions for each base model and final ensemble
    """

    from sklearn.ensemble import VotingClassifier
    import json

    if use_saved_params:
        with open("best_params.json") as f:
            saved_params = json.load(f).get(task, {})

    top_models = sorted(
        [r for r in res if r["task"] == task and r["model"] != "Rule"],
        key=lambda x: x["PREC"] + x["REC"] + x["ROC"] + x["PR"] + x["F1"] + x["ACC"],
        reverse=True
    )[:n]

    print(f"\n🔁 Trying ensemble of n = {n} on task = {task}...\n")

    estimators = []
    for r in top_models:
        name = r["model"]
        print(f"🔧 Preparing model: {name}")
        base = build_models(task)[name]

        if use_saved_params and name in saved_params:
            base.set_params(**saved_params[name])
        else:
            base = tune(name, base, X_vec, silver[f"silver_{task}"])

        base.fit(X_vec, silver[f"silver_{task}"])
        y_pred_i = base.predict(X_gold)
        y_true = gold[f"label_{task}"].values

        # Log false predictions per model
        log_false_preds(task, gold.clean, y_true, y_pred_i, model_mame=name)

        estimators.append((name, base))

    # Ensemble
    ens = VotingClassifier(estimators, voting="hard", n_jobs=-1)
    ens.fit(X_vec, silver[f"silver_{task}"])

    if hasattr(ens, "predict_proba"):
        prob = ens.predict_proba(X_gold)[:, 1]
    else:
        preds = np.column_stack([clf.predict(X_gold) for clf in ens.estimators_])
        prob = preds.mean(axis=1)

    prob = verify_with_rules(task, gold.clean, prob)
    y_true = gold[f"label_{task}"].values
    y_pred = (prob >= 0.5).astype(int)

    print(f"\n── False Predictions: Ensemble Top-{n} on {task} ──")
    log_false_preds(task, gold.clean, y_true, y_pred, model_mame=f"EnsembleTop{n}")

    return pack(y_true, prob) | {"model": f"Ens{n}", "task": task}

# ─────────────── main ───────────────


def main():
    download_raw()
    csv = parquet_to_csv()

    gold = pd.read_csv(CFG.data_dir/"ground_truth_sample.csv")
    gold["label_keto"] = gold.filter(regex="keto").iloc[:, 0].astype(int)
    gold["label_vegan"] = gold.filter(regex="vegan").iloc[:, 0].astype(int)
    gold["clean"] = gold.ingredients.fillna("").map(normalise)

    silver = build_silver(csv)
    show_balance(gold, "Gold")
    show_balance(silver, "Silver")

    vec = TfidfVectorizer(**CFG.vec_kwargs)
    X_silver = vec.fit_transform(silver.clean)
    X_gold = vec.transform(gold.clean)

    res = run_mode_A(X_silver, gold.clean, X_gold, silver, gold)
    res_ens = [
        # top_n("keto", res, X_silver, gold.clean, X_gold, silver, gold, n=4),
        # top_n("vegan", res, X_silver, gold.clean, X_gold, silver, gold, n=4),
        best_ensemble("keto", res, X_silver, gold.clean, X_gold, silver, gold),
        best_ensemble("vegan", res, X_silver,
                      gold.clean, X_gold, silver, gold),
        # ensemble_all("keto", res, X_silver, gold.clean, X_gold, silver, gold),
        # ensemble_all("vegan", res, X_silver, gold.clean, X_gold, silver, gold)
    ]
    table("MODE A Ensemble (Top-3 combined metrics)", res_ens)

    with open("best_params.json", "w") as fp:
        json.dump({k: v.get_params()
                  for k, v in BEST.items() if k != "Rule"}, fp, indent=2)


if __name__ == "__main__":
    main()
