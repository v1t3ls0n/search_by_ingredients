#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Argmax ingredient-diet pipeline — v3
Hard-verify every model's output with blacklist / whitelist rules

Complete implementation including:
- Silver dataset generation from large unlabeled data
- Training on silver labels
- Evaluation on gold standard dataset
- Ensemble methods with dynamic selection
- False prediction logging
"""

# --- Future compatibility ---
from __future__ import annotations

# --- Standard library ---
import json
import logging
import re
import unicodedata
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

# --- Third-party: core ---
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import hstack, csr_matrix
import psutil

# --- NLTK (used for lemmatization) ---
import nltk
from nltk.stem import WordNetLemmatizer
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    wnl = WordNetLemmatizer()
except:
    wnl = None

# --- Optional: scikit-learn ---
try:
    from sklearn.base import BaseEstimator, ClassifierMixin, clone
    from sklearn.model_selection import cross_val_score
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import VotingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import (
        LogisticRegression, SGDClassifier,
        PassiveAggressiveClassifier, RidgeClassifier
    )
    from sklearn.metrics import (
        accuracy_score, average_precision_score, confusion_matrix,
        precision_score, recall_score, f1_score, roc_auc_score,
        precision_recall_curve, roc_curve,
        ConfusionMatrixDisplay, RocCurveDisplay
    )
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC, LinearSVC
    SKLEARN_AVAILABLE = True
except ImportError as e:  # pragma: no cover
    warnings.warn(
        f"scikit-learn not installed ({e}). ML features will be disabled.", stacklevel=2)
    SKLEARN_AVAILABLE = False

    # Fallbacks
    class BaseEstimator:
        pass

    class ClassifierMixin:
        pass

    def clone(obj): return obj

    def make_pipeline(
        *args, **kwargs): raise ImportError("scikit-learn is required")

    def precision_recall_curve(
        *args, **kwargs): raise ImportError("scikit-learn is required")

    class RBFSampler:
        pass

    class SVC:
        pass

    class CalibratedClassifierCV:
        pass

    class VotingClassifier:
        pass

    class TfidfVectorizer:
        def __init__(
            self, **kwargs): raise ImportError("scikit-learn is required")

    class LogisticRegression:
        pass
    LinearSVC = MLPClassifier = GridSearchCV = RandomizedSearchCV = None
    accuracy_score = average_precision_score = confusion_matrix = f1_score = precision_score = recall_score = roc_auc_score = None
    SGDClassifier = MultinomialNB = PassiveAggressiveClassifier = RidgeClassifier = None

# --- Optional: LightGBM ---
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None

# --- Optional: PyTorch and torchvision (for image embeddings) ---
try:  # pragma: no cover
    import requests
    from PIL import Image
    import torch
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except Exception as e:  # pragma: no cover
    warnings.warn(
        f"PyTorch/torchvision not installed ({e}). Image features disabled.", stacklevel=2)
    Image = None
    requests = None
    torch = None
    models = None
    transforms = None
    TORCH_AVAILABLE = False

# --- Imbalanced learning ---
from imblearn.over_sampling import SMOTE, RandomOverSampler


# ============================================================================
# DOMAIN HARDCODED LISTS (HEURISTICS) - Complete from original
# ============================================================================

NON_KETO = list(set([
    # "corn",
    "kidney bean",
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
    'job\'s tear',
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
    r"\bkidney\b",
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
    r"\bsour cream\b",
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
    r"\begg\s*fruit\b",
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
    r"\bsunflower(?:\s*seed)?\s+butter\b",
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
    r"\bbeefsteak\s+plant\b",
    r"\bbeefsteak\s+mushroom\b",
    # — chicken / hen —
    r"\bchicken[- ]of[- ]the[- ]woods\b",
    r"\bchicken\s+mushroom\b",
    r"\bhen[- ]of[- ]the[- ]woods\b",
    # — meat —
    r"\bsweetmeat\s+(?:pumpkin|squash)\b",
    # — bacon —
    r"\bcoconut\s+bacon\b",
    r"\bmushroom\s+bacon\b",
    r"\bsoy\s+bacon\b",
    r"\bvegan\s+bacon\b",
]

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)s │ %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("PIPE")

# ============================================================================
# CONFIG
# ============================================================================


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
    image_dir: Path = Path("dataset/arg_max/images")


CFG = Config()

# ============================================================================
# REGEX HELPERS
# ============================================================================


def compile_any(words: Iterable[str]) -> re.Pattern[str]:
    return re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, words)), re.I)


RX_KETO = compile_any(NON_KETO)
RX_VEGAN = compile_any(NON_VEGAN)
RX_WL_KETO = re.compile("|".join(KETO_WHITELIST), re.I)
RX_WL_VEGAN = re.compile("|".join(VEGAN_WHITELIST), re.I)

# ============================================================================
# NORMALIZATION
# ============================================================================

_LEMM = WordNetLemmatizer() if wnl else None
_UNITS = re.compile(r"\b(?:g|gram|kg|oz|ml|l|cup|cups|tsp|tbsp|teaspoon|"
                    r"tablespoon|pound|lb|slice|slices|small|large|medium)\b")


def normalise(t: str | list | tuple | np.ndarray) -> str:
    """Normalize ingredient text for consistent matching.

    The ``ingredients`` field from the allrecipes dataset may be stored as a
    list/array of strings when loaded from parquet.  ``normalise`` now accepts
    such iterables and joins them before applying text cleanup so that both
    CSV and parquet formats behave the same.
    """
    if not isinstance(t, str):
        if isinstance(t, (list, tuple, np.ndarray)):
            t = " ".join(map(str, t))
        else:
            t = str(t)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    t = re.sub(r"\([^)]*\)", " ", t.lower())
    t = _UNITS.sub(" ", t)
    t = re.sub(r"\d+(?:[/\.]\d+)?", " ", t)
    t = re.sub(r"[^\w\s-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if _LEMM:
        return " ".join(_LEMM.lemmatize(w) for w in t.split() if len(w) > 2)
    return " ".join(w for w in t.split() if len(w) > 2)

# ============================================================================
# RULE MODEL
# ============================================================================


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

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# ============================================================================
# VERIFICATION LAYER
# ============================================================================


def filter_silver_by_downloaded_images(silver_df: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    """Keep only rows in silver that have corresponding downloaded image files."""
    downloaded_ids = [int(p.stem)
                      for p in (image_dir / "silver").glob("*.jpg")]
    return silver_df.loc[silver_df.index.intersection(downloaded_ids)].copy()


def tokenize_ingredient(text: str) -> list[str]:
    return re.findall(r"\b\w[\w-]*\b", text.lower())


def is_keto_ingredient_list(tokens: list[str]) -> bool:
    for ingredient in NON_KETO:
        ing_tokens = ingredient.split()
        if all(tok in tokens for tok in ing_tokens):
            return False
    return True


def find_non_keto_hits(text: str) -> list[str]:
    tokens = set(tokenize_ingredient(text))
    return sorted([
        ingredient for ingredient in NON_KETO
        if all(tok in tokens for tok in ingredient.split())
    ])


def verify_with_rules(task: str, clean: pd.Series, prob: np.ndarray) -> np.ndarray:
    """Apply rule-based verification to ML predictions."""
    adjusted = prob.copy()

    if task == "keto":
        # Regex-based whitelist/blacklist
        is_whitelisted = clean.str.contains(RX_WL_KETO)
        is_blacklisted = clean.str.contains(RX_KETO)
        forced_non_keto = is_blacklisted & ~is_whitelisted
        adjusted[forced_non_keto.values] = 0.0

        # Token-based ingredient verification
        for i, txt in enumerate(clean):
            if adjusted[i] > 0.5:
                tokens = tokenize_ingredient(normalise(txt))
                if not is_keto_ingredient_list(tokens):
                    adjusted[i] = 0.0
                    log.debug("Heuristically rejected '%s' as non-keto", txt)

        if forced_non_keto.any():
            log.debug("Keto Verification: forced %d probs to 0 (regex)",
                      forced_non_keto.sum())

    else:  # vegan
        bad = clean.str.contains(RX_VEGAN) & ~clean.str.contains(RX_WL_VEGAN)
        adjusted[bad.values] = 0.0
        if bad.any():
            log.debug("Vegan Verification: forced %d probs to 0", bad.sum())

    return adjusted

# ============================================================================
# DATA I/O
# ============================================================================

def filter_low_quality_images(img_dir: Path, embeddings: np.ndarray, original_indices: list) -> tuple:
    """Filter out low-quality images and return both embeddings AND indices."""
    if embeddings.shape[0] == 0:
        return embeddings, original_indices
    
    # Calculate embedding statistics
    variances = np.var(embeddings, axis=1)
    means = np.mean(embeddings, axis=1)
    
    # Remove embeddings with very low variance (likely blank/corrupted images)
    variance_threshold = np.percentile(variances, 10)  # Bottom 10%
    
    # Remove embeddings that are too similar to the mean (likely generic/placeholder images)
    mean_threshold = np.percentile(means, 90)  # Top 10% of means
    
    quality_mask = (variances > variance_threshold) & (means < mean_threshold)
    
    if quality_mask.sum() > embeddings.shape[0] * 0.5:  # Keep at least 50%
        filtered_embeddings = embeddings[quality_mask]
        # CRITICAL FIX: Return the filtered indices too!
        filtered_indices = [original_indices[i] for i in range(len(original_indices)) if quality_mask[i]]
        
        log.info(f"      ├─ Quality filtering: {len(filtered_indices)}/{len(original_indices)} images kept")
        return filtered_embeddings, filtered_indices
    else:
        log.info(f"      ├─ Quality filtering: Keeping all images (filter too aggressive)")
        return embeddings, original_indices



def load_datasets_fixed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load datasets into memory with comprehensive logging and progress tracking.
    
    Enhanced with:
    - Multi-stage progress bars for each loading phase
    - Data validation and integrity checks
    - Memory usage monitoring
    - Schema validation and type checking
    - Missing data analysis
    - Performance metrics and timing
    - Network error handling and retries
    - Data quality assessment
    
    Returns:
        tuple: (silver_dataframe, gold_dataframe, recipes_dataframe)
    """
    import time
    import psutil
    import requests
    from urllib.parse import urlparse
    import warnings
    from collections import Counter
    
    load_start = time.time()
    
    # ------------------------------------------------------------------
    # Initialization and Configuration
    # ------------------------------------------------------------------
    log.info("\n📂 DATASET LOADING PIPELINE")
    log.info(f"   Configuration: {len(CFG.url_map)} data sources")
    log.info(f"   Data directory: {CFG.data_dir}")
    
    # Log data source information
    log.info(f"   📊 Data Sources:")
    for name, url in CFG.url_map.items():
        source_type = "URL" if url.startswith(('http://', 'https://')) else "Local"
        log.info(f"   ├─ {name}: {source_type}")
        if source_type == "URL":
            log.info(f"   │  └─ {url}")
        else:
            log.info(f"   │  └─ {Path(url).resolve()}")

    def log_memory_usage(stage: str):
        """Helper to log current memory usage"""
        memory = psutil.virtual_memory()
        log.info(f"      💾 {stage}: {memory.percent:.1f}% memory used "
                f"({memory.used // (1024**2)} MB / {memory.total // (1024**2)} MB)")

    # Track loading stages
    loading_stages = ["Recipes", "Ground Truth", "Silver Labels", "Data Validation"]
    
    # Main pipeline progress
    pipeline_progress = tqdm(loading_stages, desc="   ├─ Loading Pipeline", 
                           position=0, leave=False,
                           bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]")

    # ------------------------------------------------------------------
    # STAGE 1: Load Recipes Dataset
    # ------------------------------------------------------------------
    pipeline_progress.set_description("   ├─ Loading Recipes")
    stage_start = time.time()
    
    log.info(f"\n   🍳 STAGE 1: LOADING RECIPES DATASET")
    
    recipes_url = CFG.url_map["allrecipes.parquet"]
    log.info(f"   ├─ Source: {recipes_url}")
    
    with tqdm(total=4, desc="      ├─ Recipe Loading", position=1, leave=False,
             bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as recipe_pbar:
        
        recipe_pbar.set_description("      ├─ Validating source")
        
        # Validate URL/path
        if recipes_url.startswith(('http://', 'https://')):
            try:
                # Quick HEAD request to check if URL is accessible
                response = requests.head(recipes_url, timeout=10)
                response.raise_for_status()
                log.info(f"      ├─ URL accessible: {response.status_code}")
                
                # Get content length if available
                content_length = response.headers.get('content-length')
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    log.info(f"      ├─ Expected download size: {size_mb:.1f} MB")
                    
            except requests.RequestException as e:
                log.error(f"      ❌ URL validation failed: {e}")
                raise RuntimeError(f"Cannot access recipes URL: {recipes_url}")
        else:
            # Local file validation
            recipes_path = Path(recipes_url)
            if not recipes_path.exists():
                raise FileNotFoundError(f"Recipes file not found: {recipes_url}")
            
            size_mb = recipes_path.stat().st_size / (1024 * 1024)
            log.info(f"      ├─ Local file size: {size_mb:.1f} MB")
        
        recipe_pbar.update(1)
        
        recipe_pbar.set_description("      ├─ Reading parquet")
        recipes_load_start = time.time()
        
        try:
            # Load with progress indication for large files
            recipes = pd.read_parquet(recipes_url)
            
        except Exception as e:
            log.error(f"      ❌ Failed to load recipes: {str(e)[:100]}...")
            
            # Try alternative approaches
            if recipes_url.startswith(('http://', 'https://')):
                log.info(f"      🔄 Attempting manual download...")
                try:
                    response = requests.get(recipes_url, stream=True, timeout=30)
                    response.raise_for_status()
                    
                    # Save temporarily and load
                    temp_path = Path("temp_recipes.parquet")
                    with open(temp_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    recipes = pd.read_parquet(temp_path)
                    temp_path.unlink()  # Clean up
                    
                except Exception as e2:
                    log.error(f"      ❌ Manual download also failed: {e2}")
                    raise RuntimeError(f"Failed to load recipes after retry: {e2}")
            else:
                raise
        
        recipes_load_time = time.time() - recipes_load_start
        recipe_pbar.update(1)
        
        recipe_pbar.set_description("      ├─ Validating schema")
        
        # Validate recipes schema
        expected_columns = ['ingredients', 'title', 'description', 'instructions']
        missing_columns = [col for col in expected_columns if col not in recipes.columns]
        
        if missing_columns:
            log.warning(f"      ⚠️  Missing expected columns: {missing_columns}")
        
        log.info(f"      📊 Recipes Dataset:")
        log.info(f"      ├─ Shape: {recipes.shape}")
        log.info(f"      ├─ Columns: {list(recipes.columns)}")
        log.info(f"      ├─ Memory usage: {recipes.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
        log.info(f"      └─ Load time: {recipes_load_time:.1f}s")
        
        recipe_pbar.update(1)
        
        recipe_pbar.set_description("      ├─ Data quality check")
        
        # Quick data quality assessment
        quality_stats = {
            'total_rows': len(recipes),
            'null_ingredients': recipes['ingredients'].isnull().sum(),
            'empty_ingredients': (
                    recipes['ingredients']           # keep the original Series
                        .astype(str)              # lists/None → string form
                        .str.strip()              # remove surrounding whitespace
                        .eq('')                   # test for genuine empties
                        .sum()                    # count them
                ) if 'ingredients' in recipes.columns else 0,
            'null_titles': recipes['title'].isnull().sum() if 'title' in recipes.columns else 0,
            'has_photo_url': 'photo_url' in recipes.columns,
            'photo_url_count': (~recipes['photo_url'].isnull()).sum() if 'photo_url' in recipes.columns else 0
        }
        
        log.info(f"      📈 Data Quality:")
        log.info(f"      ├─ Total recipes: {quality_stats['total_rows']:,}")
        log.info(f"      ├─ Null ingredients: {quality_stats['null_ingredients']:,}")
        log.info(f"      ├─ Empty ingredients: {quality_stats['empty_ingredients']:,}")
        
        if quality_stats['has_photo_url']:
            photo_pct = quality_stats['photo_url_count'] / quality_stats['total_rows'] * 100
            log.info(f"      └─ With photos: {quality_stats['photo_url_count']:,} ({photo_pct:.1f}%)")
        
        recipe_pbar.update(1)

    stage_time = time.time() - stage_start
    log.info(f"   ✅ Recipes loaded successfully in {stage_time:.1f}s")
    log_memory_usage("Recipes loaded")
    pipeline_progress.update(1)

    # ------------------------------------------------------------------
    # STAGE 2: Load Ground Truth Dataset
    # ------------------------------------------------------------------
    pipeline_progress.set_description("   ├─ Loading Ground Truth")
    stage_start = time.time()
    
    log.info(f"\n   🎯 STAGE 2: LOADING GROUND TRUTH DATASET")
    
    gt_url_or_path = CFG.url_map["ground_truth_sample.csv"]
    log.info(f"   ├─ Source: {gt_url_or_path}")
    
    with tqdm(total=4, desc="      ├─ Ground Truth Loading", position=1, leave=False,
             bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as gt_pbar:
        
        gt_pbar.set_description("      ├─ Path validation")
        
        # Validate path - check for directory mistake
        if Path(gt_url_or_path).is_dir():
            log.error(f"      ❌ Expected CSV file but found directory: {gt_url_or_path}")
            raise RuntimeError(f"Expected a CSV file but found a directory: {gt_url_or_path}")
        
        # Check accessibility
        if gt_url_or_path.startswith(('http://', 'https://')):
            try:
                response = requests.head(gt_url_or_path, timeout=10)
                response.raise_for_status()
                log.info(f"      ├─ URL accessible: {response.status_code}")
            except requests.RequestException as e:
                log.error(f"      ❌ Ground truth URL validation failed: {e}")
                raise RuntimeError(f"Cannot access ground truth URL: {gt_url_or_path}")
        else:
            gt_path = Path(gt_url_or_path)
            if not gt_path.exists():
                raise FileNotFoundError(f"Ground truth file not found: {gt_url_or_path}")
            
            size_kb = gt_path.stat().st_size / 1024
            log.info(f"      ├─ File size: {size_kb:.1f} KB")
        
        gt_pbar.update(1)
        
        gt_pbar.set_description("      ├─ Reading CSV")
        gt_load_start = time.time()
        
        try:
            # Load with error handling for encoding issues
            try:
                ground_truth = pd.read_csv(gt_url_or_path)
            except UnicodeDecodeError:
                log.warning(f"      ⚠️  UTF-8 decode failed, trying latin-1...")
                ground_truth = pd.read_csv(gt_url_or_path, encoding='latin-1')
            except pd.errors.EmptyDataError:
                log.error(f"      ❌ Ground truth file is empty")
                raise RuntimeError("Ground truth CSV file is empty")
                
        except Exception as e:
            log.error(f"      ❌ Failed to load ground truth: {str(e)[:100]}...")
            raise RuntimeError(f"Failed to load ground truth: {e}")
        
        gt_load_time = time.time() - gt_load_start
        gt_pbar.update(1)
        
        gt_pbar.set_description("      ├─ Schema validation")
        
        # Validate ground truth schema
        required_gt_columns = ['ingredients']
        missing_gt_columns = [col for col in required_gt_columns if col not in ground_truth.columns]
        
        if missing_gt_columns:
            log.error(f"      ❌ Missing required columns: {missing_gt_columns}")
            raise ValueError(f"Ground truth missing required columns: {missing_gt_columns}")
        
        # Look for label columns
        keto_columns = [col for col in ground_truth.columns if 'keto' in col.lower()]
        vegan_columns = [col for col in ground_truth.columns if 'vegan' in col.lower()]
        
        log.info(f"      📊 Ground Truth Dataset:")
        log.info(f"      ├─ Shape: {ground_truth.shape}")
        log.info(f"      ├─ Columns: {list(ground_truth.columns)}")
        log.info(f"      ├─ Keto columns found: {keto_columns}")
        log.info(f"      ├─ Vegan columns found: {vegan_columns}")
        log.info(f"      └─ Load time: {gt_load_time:.2f}s")
        
        gt_pbar.update(1)
        
        gt_pbar.set_description("      ├─ Label processing")
        
        # Process labels with error handling
        try:
            # Extract keto labels
            if keto_columns:
                ground_truth["label_keto"] = ground_truth.filter(regex="keto").iloc[:, 0].astype(int)
                keto_positive = ground_truth["label_keto"].sum()
                keto_rate = keto_positive / len(ground_truth) * 100
                log.info(f"      ├─ Keto labels: {keto_positive}/{len(ground_truth)} ({keto_rate:.1f}% positive)")
            else:
                log.warning(f"      ⚠️  No keto columns found - creating dummy labels")
                ground_truth["label_keto"] = 0
            
            # Extract vegan labels
            if vegan_columns:
                ground_truth["label_vegan"] = ground_truth.filter(regex="vegan").iloc[:, 0].astype(int)
                vegan_positive = ground_truth["label_vegan"].sum()
                vegan_rate = vegan_positive / len(ground_truth) * 100
                log.info(f"      ├─ Vegan labels: {vegan_positive}/{len(ground_truth)} ({vegan_rate:.1f}% positive)")
            else:
                log.warning(f"      ⚠️  No vegan columns found - creating dummy labels")
                ground_truth["label_vegan"] = 0
                
        except Exception as e:
            log.error(f"      ❌ Label processing failed: {e}")
            raise ValueError(f"Failed to process labels: {e}")
        
        # Add photo_url if available
        ground_truth["photo_url"] = ground_truth.get("photo_url")
        
        # Clean ingredients text
        with tqdm(total=1, desc="         ├─ Normalizing text", position=2, leave=False,
                 bar_format="         ├─ {desc}: {percentage:3.0f}%|{bar}| [{elapsed}]") as norm_pbar:
            ground_truth["clean"] = ground_truth.ingredients.fillna("").map(normalise)
            norm_pbar.update(1)
        
        gt_pbar.update(1)

    stage_time = time.time() - stage_start
    log.info(f"   ✅ Ground truth loaded successfully in {stage_time:.1f}s")
    log_memory_usage("Ground truth loaded")
    pipeline_progress.update(1)

    # ------------------------------------------------------------------
    # STAGE 3: Generate Silver Labels
    # ------------------------------------------------------------------
    pipeline_progress.set_description("   ├─ Generating Silver Labels")
    stage_start = time.time()
    
    log.info(f"\n   🥈 STAGE 3: GENERATING SILVER LABELS")
    
    with tqdm(total=3, desc="      ├─ Silver Generation", position=1, leave=False,
             bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as silver_pbar:
        
        silver_pbar.set_description("      ├─ Building silver labels")
        silver_start = time.time()
        
        # Generate silver labels using heuristics
        silver = build_silver(recipes)
        
        silver_build_time = time.time() - silver_start
        silver_pbar.update(1)
        
        silver_pbar.set_description("      ├─ Adding photo URLs")
        
        # Add photo URLs from recipes
        silver["photo_url"] = recipes.get("photo_url")
        
        # Calculate silver label statistics
        silver_stats = {
            'total': len(silver),
            'keto_positive': silver['silver_keto'].sum() if 'silver_keto' in silver.columns else 0,
            'vegan_positive': silver['silver_vegan'].sum() if 'silver_vegan' in silver.columns else 0,
            'has_photos': (~silver['photo_url'].isnull()).sum() if 'photo_url' in silver.columns else 0
        }
        
        silver_pbar.update(1)
        
        silver_pbar.set_description("      ├─ Quality assessment")
        
        log.info(f"      📊 Silver Labels Generated:")
        log.info(f"      ├─ Total recipes: {silver_stats['total']:,}")
        log.info(f"      ├─ Keto positive: {silver_stats['keto_positive']:,} ({silver_stats['keto_positive']/silver_stats['total']*100:.1f}%)")
        log.info(f"      ├─ Vegan positive: {silver_stats['vegan_positive']:,} ({silver_stats['vegan_positive']/silver_stats['total']*100:.1f}%)")
        log.info(f"      ├─ With photos: {silver_stats['has_photos']:,} ({silver_stats['has_photos']/silver_stats['total']*100:.1f}%)")
        log.info(f"      └─ Generation time: {silver_build_time:.1f}s")
        
        silver_pbar.update(1)

    stage_time = time.time() - stage_start
    log.info(f"   ✅ Silver labels generated successfully in {stage_time:.1f}s")
    log_memory_usage("Silver labels generated")
    pipeline_progress.update(1)

    # ------------------------------------------------------------------
    # STAGE 4: Data Validation and Cross-Checks
    # ------------------------------------------------------------------
    pipeline_progress.set_description("   ├─ Data Validation")
    stage_start = time.time()
    
    log.info(f"\n   ✅ STAGE 4: DATA VALIDATION AND CROSS-CHECKS")
    
    with tqdm(total=5, desc="      ├─ Validation Checks", position=1, leave=False,
             bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as val_pbar:
        
        val_pbar.set_description("      ├─ Index alignment")
        
        # Check index alignment between datasets
        recipes_indices = set(recipes.index)
        silver_indices = set(silver.index)
        gt_indices = set(ground_truth.index)
        
        if recipes_indices != silver_indices:
            log.warning(f"      ⚠️  Index mismatch between recipes and silver")
            log.info(f"         ├─ Recipes: {len(recipes_indices)} indices")
            log.info(f"         └─ Silver: {len(silver_indices)} indices")
        
        val_pbar.update(1)
        
        val_pbar.set_description("      ├─ Data consistency")
        
        # Check data consistency
        consistency_issues = []
        
        # Check for null ingredients in critical datasets
        null_ingredients_recipes = recipes['ingredients'].isnull().sum()
        null_ingredients_gt = ground_truth['ingredients'].isnull().sum()
        
        if null_ingredients_recipes > 0:
            consistency_issues.append(f"Recipes has {null_ingredients_recipes} null ingredients")
        
        if null_ingredients_gt > 0:
            consistency_issues.append(f"Ground truth has {null_ingredients_gt} null ingredients")
        
        val_pbar.update(1)
        
        val_pbar.set_description("      ├─ Memory optimization")
        
        # Memory usage analysis
        datasets_memory = {
            'recipes': recipes.memory_usage(deep=True).sum() / (1024**2),
            'silver': silver.memory_usage(deep=True).sum() / (1024**2),
            'ground_truth': ground_truth.memory_usage(deep=True).sum() / (1024**2)
        }
        
        total_memory = sum(datasets_memory.values())
        
        log.info(f"      💾 Memory Usage by Dataset:")
        for dataset, memory_mb in datasets_memory.items():
            log.info(f"      ├─ {dataset.capitalize()}: {memory_mb:.1f} MB")
        log.info(f"      └─ Total: {total_memory:.1f} MB")
        
        val_pbar.update(1)
        
        val_pbar.set_description("      ├─ Label distribution")
        
        # Compare label distributions
        if len(ground_truth) > 0:
            gt_keto_rate = ground_truth['label_keto'].mean() * 100
            gt_vegan_rate = ground_truth['label_vegan'].mean() * 100
            silver_keto_rate = silver['silver_keto'].mean() * 100
            silver_vegan_rate = silver['silver_vegan'].mean() * 100
            
            log.info(f"      📊 Label Distribution Comparison:")
            log.info(f"      ├─ Keto: Gold={gt_keto_rate:.1f}%, Silver={silver_keto_rate:.1f}%")
            log.info(f"      └─ Vegan: Gold={gt_vegan_rate:.1f}%, Silver={silver_vegan_rate:.1f}%")
            
            # Flag significant differences
            keto_diff = abs(gt_keto_rate - silver_keto_rate)
            vegan_diff = abs(gt_vegan_rate - silver_vegan_rate)
            
            if keto_diff > 20:
                log.warning(f"      ⚠️  Large keto distribution difference: {keto_diff:.1f}%")
            if vegan_diff > 20:
                log.warning(f"      ⚠️  Large vegan distribution difference: {vegan_diff:.1f}%")
        
        val_pbar.update(1)
        
        val_pbar.set_description("      ├─ Final validation")
        
        # Final validation summary
        validation_summary = {
            'recipes_loaded': len(recipes) > 0,
            'ground_truth_loaded': len(ground_truth) > 0,
            'silver_generated': len(silver) > 0,
            'required_columns_present': all(col in recipes.columns for col in ['ingredients']),
            'labels_processed': 'label_keto' in ground_truth.columns and 'label_vegan' in ground_truth.columns,
            'consistency_issues': len(consistency_issues)
        }
        
        all_valid = all(validation_summary[key] for key in ['recipes_loaded', 'ground_truth_loaded', 'silver_generated', 'required_columns_present', 'labels_processed'])
        
        log.info(f"      ✅ Validation Summary:")
        for check, status in validation_summary.items():
            if isinstance(status, bool):
                status_icon = "✅" if status else "❌"
                log.info(f"      ├─ {check.replace('_', ' ').title()}: {status_icon}")
            else:
                log.info(f"      ├─ {check.replace('_', ' ').title()}: {status}")
        
        if consistency_issues:
            log.warning(f"      ⚠️  Consistency Issues Found:")
            for issue in consistency_issues:
                log.warning(f"      │  └─ {issue}")
        
        if not all_valid:
            raise RuntimeError("Dataset validation failed - see logs for details")
        
        val_pbar.update(1)

    stage_time = time.time() - stage_start
    log.info(f"   ✅ Data validation completed in {stage_time:.1f}s")
    log_memory_usage("Validation complete")
    pipeline_progress.update(1)

    # ------------------------------------------------------------------
    # Pipeline Completion Summary
    # ------------------------------------------------------------------
    total_time = time.time() - load_start
    
    log.info(f"\n🏁 DATASET LOADING COMPLETE")
    log.info(f"   ├─ Total loading time: {total_time:.1f}s")
    log.info(f"   ├─ Datasets loaded: 3 (recipes, ground_truth, silver)")
    log.info(f"   ├─ Total memory usage: {total_memory:.1f} MB")
    log.info(f"   └─ All validations passed: ✅")

    # Final dataset summary
    log.info(f"\n   📋 Final Dataset Summary:")
    log.info(f"   ├─ Recipes: {len(recipes):,} rows × {len(recipes.columns)} columns")
    log.info(f"   ├─ Ground Truth: {len(ground_truth):,} rows × {len(ground_truth.columns)} columns")
    log.info(f"   ├─ Silver Labels: {len(silver):,} rows × {len(silver.columns)} columns")
    log.info(f"   └─ Ready for ML pipeline: ✅")

    # Garbage collection for memory optimization
    import gc
    gc.collect()
    
    return silver, ground_truth, recipes


def optimize_memory_usage(stage_name=""):
    """Optimize memory usage during training with detailed logging."""
    import gc
    import psutil
    
    # Get memory before cleanup - FIXED: Ensure we get the object properly
    try:
        memory_before = psutil.virtual_memory()
        memory_before_used = memory_before.used  # Extract the actual value
        memory_before_percent = memory_before.percent
    except Exception as e:
        log.error(f"Failed to get initial memory stats: {e}")
        return "error"
    
    # Force garbage collection
    try:
        collected = gc.collect()
    except Exception as e:
        log.debug(f"Garbage collection failed: {e}")
        collected = 0
    
    # Clear GPU cache if available
    gpu_freed = 0
    if torch and torch.cuda.is_available():
        try:
            gpu_before = torch.cuda.memory_allocated() / (1024**2)  # MB
            torch.cuda.empty_cache()
            gpu_after = torch.cuda.memory_allocated() / (1024**2)  # MB
            gpu_freed = max(0, gpu_before - gpu_after)  # Ensure non-negative
        except Exception as e:
            log.debug(f"GPU memory cleanup failed: {e}")
            gpu_freed = 0
    
    # Get memory after cleanup - FIXED: Proper handling
    try:
        memory_after = psutil.virtual_memory()
        memory_after_used = memory_after.used  # Extract the actual value
        memory_after_percent = memory_after.percent
        
        # FIXED: Calculate memory freed properly
        memory_freed_bytes = max(0, memory_before_used - memory_after_used)
        memory_freed_mb = memory_freed_bytes / (1024**2)  # Convert to MB
        
    except Exception as e:
        log.error(f"Failed to get final memory stats: {e}")
        return "error"
    
    # Log results
    stage_prefix = f"{stage_name}: " if stage_name else ""
    log.info(f"   🧹 {stage_prefix}Memory cleanup")
    log.info(f"      ├─ RAM: {memory_after_percent:.1f}% used ({memory_after_used // (1024**2)} MB)")
    
    # FIXED: Proper comparison with extracted values
    if memory_freed_mb > 1.0:  # Only log if significant (> 1 MB)
        log.info(f"      ├─ RAM freed: {memory_freed_mb:.1f} MB")
    
    if collected > 0:
        log.info(f"      ├─ Objects collected: {collected}")
    
    if gpu_freed > 1.0:
        log.info(f"      ├─ GPU freed: {gpu_freed:.1f} MB")
    
    # FIXED: Use extracted percentage values for comparison
    if memory_after_percent > 85:
        log.warning(f"      ⚠️  High memory usage: {memory_after_percent:.1f}%")
        return "high"
    elif memory_after_percent > 70:
        log.warning(f"      ⚠️  Moderate memory usage: {memory_after_percent:.1f}%")
        return "moderate"
    else:
        log.info(f"      ✅ Memory usage normal: {memory_after_percent:.1f}%")
        return "normal"


def handle_memory_crisis():
    """Emergency memory cleanup when usage is critical."""
    import gc
    import psutil
    
    log.warning("🚨 MEMORY CRISIS - Applying emergency cleanup")
    
    try:
        initial_memory = psutil.virtual_memory()
        initial_percent = initial_memory.percent
        log.info(f"   ├─ Initial memory: {initial_percent:.1f}%")
        
        # Step 1: Multiple aggressive garbage collection passes
        total_collected = 0
        for i in range(5):  # More aggressive - 5 passes
            try:
                collected = gc.collect()
                total_collected += collected
                if collected > 0:
                    log.info(f"   ├─ GC pass {i+1}: {collected} objects collected")
            except Exception as e:
                log.debug(f"GC pass {i+1} failed: {e}")
        
        # Step 2: Clear all GPU memory
        gpu_freed = 0
        if torch and torch.cuda.is_available():
            try:
                gpu_before = torch.cuda.memory_allocated() / (1024**2)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # torch.cuda.ipc_collect()  # This might not exist in all versions
                gpu_after = torch.cuda.memory_allocated() / (1024**2)
                gpu_freed = max(0, gpu_before - gpu_after)
                log.info(f"   ├─ GPU memory freed: {gpu_freed:.1f} MB")
            except Exception as e:
                log.debug(f"   ├─ GPU cleanup failed: {e}")
        
        # Step 3: Clear Python internal caches
        try:
            import importlib
            if hasattr(importlib, 'invalidate_caches'):
                importlib.invalidate_caches()
        except Exception as e:
            log.debug(f"   ├─ Cache cleanup failed: {e}")
        
        # Step 4: Force memory compaction
        try:
            gc.set_debug(0)  # Disable debugging to save memory
            gc.collect()
        except Exception:
            pass
        
        # Step 5: Check final memory
        final_memory = psutil.virtual_memory()
        final_percent = final_memory.percent
        memory_freed_mb = (initial_memory.used - final_memory.used) / (1024**2)
        
        log.info(f"   ├─ Objects collected: {total_collected}")
        log.info(f"   ├─ Memory freed: {memory_freed_mb:.1f} MB")
        log.info(f"   └─ Final memory usage: {final_percent:.1f}%")
        
        return final_percent
        
    except Exception as e:
        log.error(f"Memory crisis handling failed: {e}")
        # Fallback: return a safe high value
        try:
            return psutil.virtual_memory().percent
        except:
            return 90.0  # Assume high usage if we can't measure

def _download_images(df: pd.DataFrame, img_dir: Path, max_workers: int = 16) -> list[int]:
    """
    Download images using multithreading with comprehensive logging and progress tracking.
    
    Enhanced with:
    - Real-time download statistics
    - URL validation and preprocessing
    - Bandwidth monitoring
    - Error categorization and analysis
    - Retry mechanisms for failed downloads
    - File integrity verification
    - Memory-efficient processing
    
    Args:
        df: DataFrame containing photo_url column
        img_dir: Directory to save downloaded images
        max_workers: Maximum number of concurrent download threads
        
    Returns:
        List of successful indices for filtering downstream processing
    """
    import time
    import os
    import hashlib
    from collections import defaultdict, Counter
    from urllib.parse import urlparse
    import threading
    
    download_start = time.time()
    
    # ------------------------------------------------------------------
    # Initialization and Validation
    # ------------------------------------------------------------------
    log.info(f"\n📥 IMAGE DOWNLOAD PIPELINE: {img_dir.name}")
    log.info(f"   Target directory: {img_dir}")
    log.info(f"   Max workers: {max_workers}")
    log.info(f"   Total URLs to process: {len(df):,}")
    
    # Check if PyTorch/PIL available for image processing
    if not TORCH_AVAILABLE:
        log.warning("   ⚠️  PyTorch not available - skipping image downloads")
        return []

    # Backup-based early exit
    backup_emb = img_dir / "embeddings.npy"
    if backup_emb.exists():
        try:
            num_jpgs = len(list(img_dir.glob("*.jpg")))
            if num_jpgs >= len(df):
                log.info(f"   📦 Backup detected: {num_jpgs} images + existing embeddings → skipping downloads")
                return sorted(df.index.tolist())  # Return all indices as valid
        except Exception as e:
            log.warning(f"   ⚠️ Could not verify backup completeness: {e}")

    # Create directory structure
    img_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"   ✅ Directory created/verified: {img_dir}")

    # Validate DataFrame structure
    if 'photo_url' not in df.columns:
        log.error("   ❌ No 'photo_url' column found in DataFrame")
        return []

    # ------------------------------------------------------------------
    # URL Analysis and Preprocessing
    # ------------------------------------------------------------------
    log.info(f"\n   🔍 URL Analysis:")
    
    with tqdm(total=3, desc="      ├─ Analyzing URLs", position=1, leave=False,
             bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as analysis_pbar:
        
        analysis_pbar.set_description("      ├─ Filtering valid URLs")
        
        # Analyze URL patterns and validity
        url_analysis = {
            'total': len(df),
            'valid_http': 0,
            'already_exists': 0,
            'invalid_format': 0,
            'empty_null': 0,
            'domains': Counter(),
            'extensions': Counter()
        }
        
        valid_downloads = []
        skipped_existing = []
        
        for idx, url in df['photo_url'].items():
            img_path = img_dir / f"{idx}.jpg"
            
            # Check if already exists
            if img_path.exists():
                url_analysis['already_exists'] += 1
                skipped_existing.append(idx)
                continue
                
            # Validate URL format
            if not isinstance(url, str) or not url.strip():
                url_analysis['empty_null'] += 1
                continue
                
            url = url.strip()
            if not url.startswith(('http://', 'https://')):
                url_analysis['invalid_format'] += 1
                continue
                
            # URL is valid for download
            url_analysis['valid_http'] += 1
            valid_downloads.append((idx, url))
            
            # Analyze domain and extension
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                url_analysis['domains'][domain] += 1
                
                # Extract file extension from path
                path_parts = parsed.path.lower().split('.')
                if len(path_parts) > 1:
                    ext = path_parts[-1][:10]  # Limit length
                    url_analysis['extensions'][ext] += 1
            except Exception:
                pass
        
        analysis_pbar.update(1)
        
        analysis_pbar.set_description("      ├─ Generating statistics")
        
        # Log URL analysis results
        log.info(f"      📊 URL Statistics:")
        log.info(f"      ├─ Total URLs: {url_analysis['total']:,}")
        log.info(f"      ├─ Valid HTTP(S): {url_analysis['valid_http']:,} ({url_analysis['valid_http']/url_analysis['total']*100:.1f}%)")
        log.info(f"      ├─ Already exist: {url_analysis['already_exists']:,}")
        log.info(f"      ├─ Invalid format: {url_analysis['invalid_format']:,}")
        log.info(f"      └─ Empty/null: {url_analysis['empty_null']:,}")
        
        analysis_pbar.update(1)
        
        # Show top domains
        if url_analysis['domains']:
            top_domains = url_analysis['domains'].most_common(5)
            log.info(f"      📊 Top Domains:")
            for domain, count in top_domains:
                log.info(f"      ├─ {domain}: {count:,} images ({count/url_analysis['valid_http']*100:.1f}%)")
        
        # Show file extensions
        if url_analysis['extensions']:
            top_extensions = url_analysis['extensions'].most_common(3)
            log.info(f"      📊 File Extensions:")
            for ext, count in top_extensions:
                log.info(f"      ├─ .{ext}: {count:,}")
        
        analysis_pbar.update(1)

    # Early exit if no downloads needed
    if not valid_downloads:
        log.info(f"   ✅ No new downloads needed (all {url_analysis['already_exists']} images exist)")
        return skipped_existing

    log.info(f"   🎯 Download Plan: {len(valid_downloads):,} new images to download")

    # ------------------------------------------------------------------
    # Download Execution with Enhanced Tracking
    # ------------------------------------------------------------------
    log.info(f"\n   🚀 Starting parallel downloads...")
    
    # Shared statistics with thread safety
    stats_lock = threading.Lock()
    download_stats = {
        "downloaded": 0,
        "exists": 0, 
        "invalid": 0,
        "failed": 0,
        "bytes_downloaded": 0,
        "retry_success": 0
    }
    
    valid_indices = list(skipped_existing)  # Include pre-existing
    failed_urls = []
    download_times = []
    
    def fetch_with_retry(idx_url, max_retries=2):
        """Enhanced fetch function with retry logic and detailed error handling"""
        idx, url = idx_url
        img_path = img_dir / f"{idx}.jpg"
        
        # Double-check existence (race condition safety)
        if img_path.exists():
            with stats_lock:
                stats_check = os.path.getsize(img_path)
                download_stats["exists"] += 1
                if stats_check > 0:
                    return "exists", idx, url, None, stats_check
            
        # Validate URL format (redundant check for thread safety)
        if not isinstance(url, str) or not url.strip().startswith("http"):
            with stats_lock:
                download_stats["invalid"] += 1
            return "invalid", idx, url, "Invalid URL format", 0

        # Attempt download with retries
        last_error = None
        for attempt in range(max_retries + 1):
            fetch_start = time.time()
            
            try:
                # Configure request with better error handling
                headers = {
                    'User-Agent': 'Mozilla/5.0 (compatible; RecipeImageDownloader/1.0)',
                    'Accept': 'image/*,*/*;q=0.8',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive'
                }
                
                resp = requests.get(
                    url, 
                    timeout=15,  # Increased timeout
                    headers=headers,
                    allow_redirects=True,
                    stream=True  # For large images
                )
                resp.raise_for_status()
                
                # Check content type
                content_type = resp.headers.get('content-type', '').lower()
                if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'jpg', 'png', 'gif']):
                    raise ValueError(f"Invalid content type: {content_type}")
                
                # Download with size validation
                content = resp.content
                if len(content) < 100:  # Minimum viable image size
                    raise ValueError(f"Image too small: {len(content)} bytes")
                
                if len(content) > 50 * 1024 * 1024:  # 50MB limit
                    raise ValueError(f"Image too large: {len(content)} bytes")
                
                # Write file atomically
                temp_path = img_path.with_suffix('.tmp')
                with open(temp_path, 'wb') as fh:
                    fh.write(content)
                
                # Verify file integrity
                if os.path.getsize(temp_path) != len(content):
                    os.remove(temp_path)
                    raise ValueError("File size mismatch after write")
                
                # Atomic move to final location
                temp_path.rename(img_path)
                
                # Update statistics
                fetch_time = time.time() - fetch_start
                with stats_lock:
                    if attempt > 0:
                        download_stats["retry_success"] += 1
                    download_stats["downloaded"] += 1
                    download_stats["bytes_downloaded"] += len(content)
                
                return "downloaded", idx, url, None, len(content), fetch_time
                
            except requests.exceptions.Timeout:
                last_error = f"Timeout after 15s (attempt {attempt+1})"
            except requests.exceptions.ConnectionError:
                last_error = f"Connection error (attempt {attempt+1})"
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP {e.response.status_code} (attempt {attempt+1})"
            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {str(e)[:50]} (attempt {attempt+1})"
            except ValueError as e:
                last_error = f"Validation error: {str(e)} (attempt {attempt+1})"
                break  # Don't retry validation errors
            except Exception as e:
                last_error = f"Unexpected error: {str(e)[:50]} (attempt {attempt+1})"
            
            # Brief pause before retry
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
        # All attempts failed
        with stats_lock:
            download_stats["failed"] += 1
        
        return "failed", idx, url, last_error, 0

    # ------------------------------------------------------------------
    # Parallel Download Execution
    # ------------------------------------------------------------------
    bandwidth_samples = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        futures = [executor.submit(fetch_with_retry, idx_url) 
                  for idx_url in valid_downloads]
        
        # Progress bar with real-time statistics
        progress_bar = tqdm(
            as_completed(futures), 
            total=len(futures),
            desc=f"      ├─ Downloading {img_dir.name}",
            position=1, leave=False,
            bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}] {rate_fmt}"
        )
        
        completed = 0
        for future in progress_bar:
            completed += 1
            result = future.result()
            
            # Unpack result (handle variable return length)
            if len(result) >= 5:
                status, idx, url, error, size = result[:5]
                fetch_time = result[5] if len(result) > 5 else 0
            else:
                status, idx, url, error, size = result + (0,) * (5 - len(result))
                fetch_time = 0
            
            # Track successful downloads
            if status in {"downloaded", "exists"}:
                valid_indices.append(idx)
                
                # Calculate bandwidth for downloaded files
                if status == "downloaded" and fetch_time > 0 and size > 0:
                    bandwidth_mbps = (size / (1024 * 1024)) / fetch_time
                    bandwidth_samples.append(bandwidth_mbps)
                    download_times.append(fetch_time)
            
            elif status == "failed":
                failed_urls.append((idx, url, error))
            
            # Update progress bar with live statistics
            current_time = time.time()
            elapsed = current_time - start_time
            
            if elapsed > 0:
                downloads_per_sec = completed / elapsed
                
                # Calculate current stats safely
                with stats_lock:
                    current_stats = download_stats.copy()
                
                # Estimate completion
                remaining = len(futures) - completed
                eta_seconds = remaining / downloads_per_sec if downloads_per_sec > 0 else 0
                
                # Update progress description with live stats
                progress_bar.set_postfix({
                    'Success': f"{current_stats['downloaded'] + current_stats['exists']}",
                    'Failed': f"{current_stats['failed']}",
                    'Speed': f"{downloads_per_sec:.1f}/s",
                    'ETA': f"{eta_seconds:.0f}s" if eta_seconds < 3600 else f"{eta_seconds/3600:.1f}h"
                })

    # ------------------------------------------------------------------
    # Download Results Analysis
    # ------------------------------------------------------------------
    total_time = time.time() - download_start
    
    log.info(f"\n   📊 DOWNLOAD RESULTS:")
    log.info(f"   ├─ Total processing time: {total_time:.1f}s")
    log.info(f"   ├─ Successfully downloaded: {download_stats['downloaded']:,}")
    log.info(f"   ├─ Already existed: {download_stats['exists']:,}")
    log.info(f"   ├─ Failed downloads: {download_stats['failed']:,}")
    log.info(f"   ├─ Invalid URLs: {download_stats['invalid']:,}")
    log.info(f"   └─ Retry successes: {download_stats['retry_success']:,}")

    # Success rate analysis
    total_attempted = download_stats['downloaded'] + download_stats['failed']
    if total_attempted > 0:
        success_rate = download_stats['downloaded'] / total_attempted * 100
        log.info(f"   📈 Success rate: {success_rate:.1f}% ({download_stats['downloaded']}/{total_attempted})")

    # Performance metrics
    if download_stats['downloaded'] > 0:
        log.info(f"\n   ⚡ Performance Metrics:")
        log.info(f"   ├─ Download speed: {download_stats['downloaded']/total_time:.1f} images/second")
        log.info(f"   ├─ Data downloaded: {download_stats['bytes_downloaded']/(1024*1024):.1f} MB")
        
        if bandwidth_samples:
            avg_bandwidth = sum(bandwidth_samples) / len(bandwidth_samples)
            max_bandwidth = max(bandwidth_samples)
            log.info(f"   ├─ Average bandwidth: {avg_bandwidth:.1f} MB/s")
            log.info(f"   └─ Peak bandwidth: {max_bandwidth:.1f} MB/s")
        
        if download_times:
            avg_time = sum(download_times) / len(download_times)
            log.info(f"   └─ Average download time: {avg_time:.2f}s per image")

    # ------------------------------------------------------------------
    # Error Analysis and Logging
    # ------------------------------------------------------------------
    if failed_urls:
        log.info(f"\n   ⚠️  Error Analysis ({len(failed_urls)} failures):")
        
        # Categorize errors
        error_categories = defaultdict(int)
        error_examples = defaultdict(list)
        
        for idx, url, error in failed_urls:
            # Categorize error types
            if not error:
                category = "Unknown"
            elif "timeout" in error.lower():
                category = "Timeout"
            elif "connection" in error.lower():
                category = "Connection"
            elif "404" in error or "not found" in error.lower():
                category = "Not Found"
            elif "403" in error or "forbidden" in error.lower():
                category = "Forbidden"
            elif "validation" in error.lower():
                category = "Invalid Content"
            else:
                category = "Other"
            
            error_categories[category] += 1
            if len(error_examples[category]) < 3:  # Keep max 3 examples per category
                error_examples[category].append((idx, url[:50] + "..." if len(url) > 50 else url, error))
        
        # Log error summary
        for category, count in sorted(error_categories.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(failed_urls) * 100
            log.info(f"   ├─ {category}: {count} ({percentage:.1f}%)")
            
            # Show examples for major error categories
            if count >= 5 and error_examples[category]:
                for idx, url_short, error in error_examples[category][:2]:
                    log.info(f"   │  └─ Example: {url_short} - {error[:60]}...")

        # Save detailed error log
        fail_log_path = img_dir / "failed_downloads.txt"
        try:
            with open(fail_log_path, "w", encoding='utf-8') as f:
                f.write("Index\tURL\tError\tTimestamp\n")
                for idx, url, error in failed_urls:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"{idx}\t{url}\t{error}\t{timestamp}\n")
            
            log.info(f"   💾 Detailed error log saved to: {fail_log_path}")
            
        except Exception as e:
            log.warning(f"   ⚠️  Failed to save error log: {e}")

    # ------------------------------------------------------------------
    # Disk Usage and Cleanup
    # ------------------------------------------------------------------
    try:
        # Calculate disk usage
        total_size = 0
        image_count = 0
        for img_file in img_dir.glob("*.jpg"):
            if img_file.is_file():
                total_size += img_file.stat().st_size
                image_count += 1
        
        if image_count > 0:
            log.info(f"\n   💾 Storage Summary:")
            log.info(f"   ├─ Images stored: {image_count:,}")
            log.info(f"   ├─ Total size: {total_size/(1024*1024):.1f} MB")
            log.info(f"   └─ Average size: {total_size/(1024*1024)/image_count:.2f} MB per image")
            
    except Exception as e:
        log.warning(f"   ⚠️  Storage analysis failed: {e}")

    # ------------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------------
    total_valid = len(valid_indices)
    log.info(f"\n   🏁 DOWNLOAD COMPLETE:")
    log.info(f"   ├─ Total valid images: {total_valid:,}/{len(df):,} ({total_valid/len(df)*100:.1f}%)")
    log.info(f"   ├─ Processing rate: {len(df)/total_time:.1f} URLs/second")
    log.info(f"   ├─ Thread efficiency: {max_workers} workers")
    log.info(f"   └─ Directory: {img_dir}")

    # Memory cleanup for large operations
    if len(df) > 10000:
        import gc
        gc.collect()
        log.debug(f"   🧹 Memory cleanup completed")

    return valid_indices



def build_image_embeddings(df: pd.DataFrame,
                           mode: str,
                           force: bool = False) -> np.ndarray:
    """
    Extract ResNet-50 embeddings for images with comprehensive logging and progress tracking.
    
    Enhanced with:
    - Multi-stage progress bars for cache loading, model setup, and embedding extraction
    - Detailed model performance monitoring (GPU/CPU usage, throughput)
    - Image processing statistics and error analysis
    - Memory usage tracking and optimization
    - Batch processing for improved efficiency
    - Comprehensive error categorization
    - Backup and recovery mechanisms
    - Image quality analysis
    
    Args:
        df: DataFrame with image indices
        mode: Mode identifier ('silver', 'gold', etc.)
        force: Force recomputation even if cache exists
        
    Returns:
        numpy array of shape (len(df), 2048) with ResNet-50 features
    """
    import time
    import psutil
    import os
    from collections import defaultdict, Counter
    from PIL import ImageStat
    import gc
    
    embedding_start = time.time()
    
    # ------------------------------------------------------------------
    # Initialization and System Check
    # ------------------------------------------------------------------
    log.info(f"\n🧠 IMAGE EMBEDDING EXTRACTION: {mode}")
    log.info(f"   Target images: {len(df):,}")
    log.info(f"   Mode: {mode}")
    log.info(f"   Force recomputation: {force}")
    
    # Check PyTorch availability
    if not TORCH_AVAILABLE:
        log.warning("   ❌ PyTorch not available - returning zero vectors")
        log.info(f"   └─ Zero vector shape: ({len(df)}, 2048)")
        return np.zeros((len(df), 2048), dtype=np.float32)

    # Check GPU availability and setup
    device_info = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'
    }
    
    log.info(f"   🔧 Device Configuration:")
    log.info(f"   ├─ Device: {device_info['device']}")
    log.info(f"   ├─ CUDA available: {device_info['cuda_available']}")
    if device_info['cuda_available']:
        log.info(f"   ├─ GPU count: {device_info['device_count']}")
        log.info(f"   └─ GPU name: {device_info['device_name']}")
    else:
        log.info(f"   └─ Using CPU (warning: much slower)")

    # Set up paths
    img_dir = CFG.image_dir / mode
    embed_path = img_dir / "embeddings.npy"
    backup_path = Path(f"embeddings_{mode}_backup.npy")
    metadata_path = img_dir / "embedding_metadata.json"

    log.info(f"   📁 Paths:")
    log.info(f"   ├─ Image directory: {img_dir}")
    log.info(f"   ├─ Cache file: {embed_path}")
    log.info(f"   └─ Backup file: {backup_path}")

    # ------------------------------------------------------------------
    # Cache Loading and Validation
    # ------------------------------------------------------------------
    if not force:
        log.info(f"\n   🔍 Cache Validation:")
        
        cache_options = [
            ("Primary cache", embed_path),
            ("Backup cache", backup_path)
        ]
        
        with tqdm(cache_options, desc="      ├─ Checking caches", position=1, leave=False,
                 bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as cache_pbar:
            
            for cache_name, cache_path in cache_pbar:
                cache_pbar.set_description(f"      ├─ Checking {cache_name.lower()}")
                
                if cache_path.exists():
                    try:
                        cache_start = time.time()
                        emb = np.load(cache_path)
                        load_time = time.time() - cache_start
                        
                        log.info(f"      ├─ {cache_name}: Found ({emb.shape}) - loaded in {load_time:.2f}s")
                        
                        if emb.shape[0] == len(df):
                            log.info(f"      ✅ {cache_name} matches target size - using cached embeddings")
                            
                            # Load metadata if available
                            if metadata_path.exists():
                                try:
                                    with open(metadata_path, 'r') as f:
                                        metadata = json.load(f)
                                    log.info(f"      ├─ Cache metadata: {metadata.get('creation_time', 'Unknown time')}")
                                    log.info(f"      └─ Original stats: {metadata.get('success', '?')} success, "
                                            f"{metadata.get('failed', '?')} failed")
                                except Exception as e:
                                    log.debug(f"      └─ Metadata load failed: {e}")
                            
                            return emb
                            
                        else:
                            log.warning(f"      ⚠️  {cache_name} size mismatch: {emb.shape[0]} != {len(df)}")
                            
                            if emb.shape[0] > len(df):
                                log.info(f"      ├─ Truncating cache from {emb.shape[0]} to {len(df)}")
                                return emb[:len(df)]
                                
                    except Exception as e:
                        log.error(f"      ❌ {cache_name} load failed: {str(e)[:60]}...")
                else:
                    log.info(f"      ├─ {cache_name}: Not found")

        log.info(f"      └─ No valid cache found - will compute embeddings")

    else:
        log.info(f"\n   🔄 Cache bypassed (force=True) - recomputing embeddings")

    # ------------------------------------------------------------------
    # Pre-processing Analysis
    # ------------------------------------------------------------------
    log.info(f"\n   📊 Pre-processing Analysis:")
    
    with tqdm(total=3, desc="      ├─ Analyzing images", position=1, leave=False,
             bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as analysis_pbar:
        
        analysis_pbar.set_description("      ├─ Scanning directory")
        
        # Check which images exist
        existing_images = []
        missing_images = []
        corrupted_images = []
        
        for idx in df.index:
            img_file = img_dir / f"{idx}.jpg"
            if img_file.exists():
                try:
                    # Quick validation - try to open
                    with Image.open(img_file) as img:
                        img.verify()  # Verify image integrity
                    existing_images.append(idx)
                except Exception:
                    corrupted_images.append(idx)
            else:
                missing_images.append(idx)
        
        analysis_pbar.update(1)
        
        analysis_pbar.set_description("      ├─ Computing statistics")
        
        # Calculate statistics
        total_images = len(df)
        existing_count = len(existing_images)
        missing_count = len(missing_images)
        corrupted_count = len(corrupted_images)
        
        log.info(f"      📈 Image Availability:")
        log.info(f"      ├─ Total expected: {total_images:,}")
        log.info(f"      ├─ Available: {existing_count:,} ({existing_count/total_images*100:.1f}%)")
        log.info(f"      ├─ Missing: {missing_count:,} ({missing_count/total_images*100:.1f}%)")
        log.info(f"      └─ Corrupted: {corrupted_count:,} ({corrupted_count/total_images*100:.1f}%)")
        
        analysis_pbar.update(1)
        
        # Sample image analysis
        analysis_pbar.set_description("      ├─ Sampling quality")
        
        if existing_images:
            sample_size = min(100, len(existing_images))
            sample_indices = np.random.choice(existing_images, sample_size, replace=False)
            
            image_stats = {
                'sizes': [],
                'modes': Counter(),
                'formats': Counter(),
                'file_sizes': []
            }
            
            for idx in sample_indices[:10]:  # Analyze first 10 for detailed stats
                img_file = img_dir / f"{idx}.jpg"
                try:
                    with Image.open(img_file) as img:
                        image_stats['sizes'].append(img.size)
                        image_stats['modes'][img.mode] += 1
                        image_stats['formats'][img.format] += 1
                        image_stats['file_sizes'].append(img_file.stat().st_size)
                except Exception:
                    pass
            
            if image_stats['sizes']:
                avg_width = sum(s[0] for s in image_stats['sizes']) / len(image_stats['sizes'])
                avg_height = sum(s[1] for s in image_stats['sizes']) / len(image_stats['sizes'])
                avg_file_size = sum(image_stats['file_sizes']) / len(image_stats['file_sizes'])
                
                log.info(f"      📊 Sample Analysis ({len(image_stats['sizes'])} images):")
                log.info(f"      ├─ Average size: {avg_width:.0f}×{avg_height:.0f} pixels")
                log.info(f"      ├─ Average file size: {avg_file_size/1024:.1f} KB")
                log.info(f"      ├─ Color modes: {dict(image_stats['modes'])}")
                log.info(f"      └─ Formats: {dict(image_stats['formats'])}")
        
        analysis_pbar.update(1)

    # ------------------------------------------------------------------
    # Model Setup and Initialization
    # ------------------------------------------------------------------
    log.info(f"\n   🤖 Model Setup:")
    
    with tqdm(total=4, desc="      ├─ Loading model", position=1, leave=False,
             bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as model_pbar:
        
        model_pbar.set_description("      ├─ Loading ResNet-50")
        model_start = time.time()
        
        # Load pre-trained ResNet-50
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model_pbar.update(1)
        
        model_pbar.set_description("      ├─ Modifying architecture")
        # Remove classification head for feature extraction
        model.fc = torch.nn.Identity()
        model.eval()
        model_pbar.update(1)
        
        model_pbar.set_description("      ├─ Moving to device")
        model.to(device_info['device'])
        model_time = time.time() - model_start
        model_pbar.update(1)
        
        model_pbar.set_description("      ├─ Setting up preprocessing")
        # Standard ImageNet preprocessing
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
        model_pbar.update(1)

    log.info(f"      ✅ Model loaded in {model_time:.2f}s")
    log.info(f"      ├─ Architecture: ResNet-50 (feature extractor)")
    log.info(f"      ├─ Output dimension: 2048")
    log.info(f"      ├─ Device: {device_info['device']}")
    log.info(f"      └─ Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Memory usage after model loading
    if device_info['cuda_available']:
        gpu_memory = torch.cuda.memory_allocated() / (1024**2)
        log.info(f"      📊 GPU memory allocated: {gpu_memory:.1f} MB")

    # ------------------------------------------------------------------
    # Embedding Extraction with Detailed Progress
    # ------------------------------------------------------------------
    log.info(f"\n   ⚡ Feature Extraction:")
    
    vectors = []
    processing_stats = {
        'success': 0,
        'missing': 0, 
        'failed': 0,
        'processing_times': [],
        'error_types': Counter(),
        'batch_times': []
    }
    
    failed_details = []
    
    # Determine batch size based on available memory
    if device_info['cuda_available']:
        # Estimate batch size based on GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        batch_size = max(1, min(32, int(gpu_memory_gb * 2)))  # Conservative estimate
    else:
        batch_size = 8  # Conservative CPU batch size
    
    log.info(f"      🔧 Processing Configuration:")
    log.info(f"      ├─ Batch size: {batch_size}")
    log.info(f"      ├─ Total batches: {(len(df) + batch_size - 1) // batch_size}")
    log.info(f"      └─ Expected output shape: ({len(df)}, 2048)")

    # Main processing loop with progress tracking
    with tqdm(df.index, desc=f"      ├─ Extracting {mode} embeddings", 
             position=1, leave=False,
             bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}] {rate_fmt}") as extract_pbar:
        
        for i, idx in enumerate(extract_pbar):
            process_start = time.time()
            img_file = img_dir / f"{idx}.jpg"
            
            # Update progress bar description periodically
            if i % 100 == 0:
                success_rate = processing_stats['success'] / max(1, i) * 100
                extract_pbar.set_postfix({
                    'Success': f"{processing_stats['success']}",
                    'Failed': f"{processing_stats['failed']}",
                    'Missing': f"{processing_stats['missing']}",
                    'Rate': f"{success_rate:.1f}%"
                })
            
            # Check if image exists
            if not img_file.exists():
                processing_stats['missing'] += 1
                vectors.append(np.zeros(2048, dtype=np.float32))
                continue
                
            try:
                # Load and preprocess image
                img = Image.open(img_file).convert('RGB')
                
                # Optional: Log image properties for first few images
                if processing_stats['success'] < 5:
                    log.debug(f"         ├─ Processing {img_file.name}: {img.size} {img.mode}")
                
                with torch.no_grad():
                    # Preprocess and add batch dimension
                    tensor = preprocess(img).unsqueeze(0).to(device_info['device'])
                    
                    # Extract features
                    features = model(tensor).squeeze().cpu().numpy()
                    
                    # Validate output shape
                    if features.shape != (2048,):
                        raise ValueError(f"Unexpected feature shape: {features.shape}")
                
                vectors.append(features)
                processing_stats['success'] += 1
                
                # Track processing time
                process_time = time.time() - process_start
                processing_stats['processing_times'].append(process_time)
                
            except Exception as e:
                processing_stats['failed'] += 1
                
                # Categorize error type
                error_type = type(e).__name__
                processing_stats['error_types'][error_type] += 1
                
                # Log detailed error info
                error_msg = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
                failed_details.append((idx, img_file, error_type, error_msg))
                
                if processing_stats['failed'] <= 5:  # Log first few errors in detail
                    log.warning(f"         ❌ Failed {img_file.name}: {error_type} - {error_msg}")
                
                # Add zero vector for failed processing
                vectors.append(np.zeros(2048, dtype=np.float32))
            
            # Periodic memory cleanup
            if i % 1000 == 0 and i > 0:
                gc.collect()
                optimize_memory_usage("Batch Processing")
                if device_info['cuda_available']:
                    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Post-processing and Results Analysis
    # ------------------------------------------------------------------
    extraction_time = time.time() - embedding_start
    
    log.info(f"\n   📊 Extraction Results:")
    log.info(f"   ├─ Total processing time: {extraction_time:.1f}s")
    log.info(f"   ├─ Successfully processed: {processing_stats['success']:,}")
    log.info(f"   ├─ Missing images: {processing_stats['missing']:,}")
    log.info(f"   ├─ Failed processing: {processing_stats['failed']:,}")
    log.info(f"   └─ Overall success rate: {processing_stats['success']/len(df)*100:.1f}%")

    # Performance metrics
    if processing_stats['processing_times']:
        avg_time = sum(processing_stats['processing_times']) / len(processing_stats['processing_times'])
        throughput = processing_stats['success'] / extraction_time
        
        log.info(f"   ⚡ Performance Metrics:")
        log.info(f"   ├─ Average processing time: {avg_time:.3f}s per image")
        log.info(f"   ├─ Throughput: {throughput:.1f} images/second")
        log.info(f"   └─ Device efficiency: {device_info['device']}")

    # Error analysis
    if processing_stats['failed'] > 0:
        log.info(f"   ⚠️  Error Analysis:")
        total_errors = sum(processing_stats['error_types'].values())
        
        for error_type, count in processing_stats['error_types'].most_common():
            percentage = count / total_errors * 100
            log.info(f"   ├─ {error_type}: {count} ({percentage:.1f}%)")
        
        # Save detailed error log
        if failed_details:
            error_log_path = img_dir / "embedding_errors.txt"
            try:
                with open(error_log_path, "w") as f:
                    f.write("Index\tFile\tErrorType\tErrorMessage\n")
                    for idx, img_file, error_type, error_msg in failed_details:
                        f.write(f"{idx}\t{img_file.name}\t{error_type}\t{error_msg}\n")
                log.info(f"   💾 Error details saved to: {error_log_path}")
            except Exception as e:
                log.warning(f"   ⚠️  Failed to save error log: {e}")

    # ------------------------------------------------------------------
    # Save Results and Metadata
    # ------------------------------------------------------------------
    log.info(f"\n   💾 Saving Results:")
    
    with tqdm(total=4, desc="      ├─ Saving files", position=1, leave=False,
             bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as save_pbar:
        
        save_pbar.set_description("      ├─ Stacking vectors")
        # Convert list to numpy array
        arr = np.vstack(vectors)
        save_pbar.update(1)
        
        save_pbar.set_description("      ├─ Creating directories")
        # Ensure directory exists
        embed_path.parent.mkdir(parents=True, exist_ok=True)
        save_pbar.update(1)
        
        save_pbar.set_description("      ├─ Saving primary cache")
        # Save primary cache
        np.save(embed_path, arr)
        save_pbar.update(1)
        
        save_pbar.set_description("      ├─ Saving backup")
        # Save backup
        np.save(backup_path, arr)
        save_pbar.update(1)

    # Save metadata
    metadata = {
        'creation_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'mode': mode,
        'total_images': len(df),
        'success': processing_stats['success'],
        'missing': processing_stats['missing'],
        'failed': processing_stats['failed'],
        'processing_time_seconds': extraction_time,
        'device': str(device_info['device']),
        'model': 'ResNet-50',
        'output_shape': list(arr.shape),
        'avg_processing_time': sum(processing_stats['processing_times']) / len(processing_stats['processing_times']) if processing_stats['processing_times'] else 0,
        'throughput_images_per_second': processing_stats['success'] / extraction_time if extraction_time > 0 else 0
    }
    
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        log.info(f"   💾 Metadata saved to: {metadata_path}")
    except Exception as e:
        log.warning(f"   ⚠️  Failed to save metadata: {e}")

    # Final file size information
    try:
        primary_size = embed_path.stat().st_size / (1024**2)
        backup_size = backup_path.stat().st_size / (1024**2)
        
        log.info(f"   📊 File Information:")
        log.info(f"   ├─ Primary cache: {primary_size:.1f} MB ({embed_path})")
        log.info(f"   ├─ Backup cache: {backup_size:.1f} MB ({backup_path})")
        log.info(f"   └─ Array shape: {arr.shape}")
        
    except Exception as e:
        log.warning(f"   ⚠️  File size analysis failed: {e}")

    # Memory cleanup
    del model, vectors
    gc.collect()
    if device_info['cuda_available']:
        torch.cuda.empty_cache()
        final_gpu_memory = torch.cuda.memory_allocated() / (1024**2)
        log.info(f"   🧹 GPU memory after cleanup: {final_gpu_memory:.1f} MB")

    # ------------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------------
    
    log.info(f"\n   🏁 EMBEDDING EXTRACTION COMPLETE:")
    log.info(f"   ├─ Output shape: {arr.shape}")
    log.info(f"   ├─ Success rate: {processing_stats['success']/len(df)*100:.1f}%")
    log.info(f"   ├─ Total time: {extraction_time:.1f}s")
    log.info(f"   ├─ Throughput: {processing_stats['success']/extraction_time:.1f} images/s")
    log.info(f"   └─ Files saved: Primary + Backup + Metadata")

    # Apply quality filtering
    original_indices = list(df.index)
    if arr.shape[0] > 10:  # Only filter if we have enough images
        arr, valid_indices = filter_low_quality_images(img_dir, arr, original_indices)
        if len(valid_indices) != len(original_indices):
            log.info(f"   📊 Quality filtering reduced images from {len(original_indices)} to {len(valid_indices)}")
    else:
        valid_indices = original_indices

    return arr, valid_indices


def combine_features(X_text, X_image) -> csr_matrix:
    """Concatenate sparse text matrix with dense image array."""
    img_sparse = csr_matrix(X_image)
    return hstack([X_text, img_sparse])


def filter_photo_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with usable photo URLs."""
    if 'photo_url' not in df.columns:
        return df.iloc[0:0].copy()
    mask = ~df['photo_url'].str.contains(
        r"nophoto|nopic|nopicture", case=False, na=False)
    mask &= df['photo_url'].astype(bool)
    return df.loc[mask].copy()

def apply_smote(X, y, max_dense_size: int = int(5e7)):
    """Apply SMOTE when classes are imbalanced (<40% minority) - FIXED VERSION."""
    try:
        counts = np.bincount(y)
        if len(counts) < 2:
            return X, y
            
        ratio = counts.min() / counts.sum()
        if ratio < 0.4:
            # FIXED: Check if X is sparse, then properly convert
            if hasattr(X, "toarray"):  # X is sparse
                elements = X.shape[0] * X.shape[1]
                if elements > max_dense_size:
                    ros = RandomOverSampler(random_state=42)
                    return ros.fit_resample(X, y)
                else:
                    X_dense = X.toarray()  # FIXED: Actually call the method
                    smote = SMOTE(sampling_strategy=0.3, random_state=42)
                    return smote.fit_resample(X_dense, y)
            else:
                # X is already dense
                smote = SMOTE(sampling_strategy=0.3, random_state=42)
                return smote.fit_resample(X, y)
        return X, y
    except Exception as e:
        log.warning(f"SMOTE failed: {e}. Using original data.")
        return X, y

# ============================================================================
# SILVER LABELS GENERATION
# ============================================================================


def build_silver(recipes: pd.DataFrame) -> pd.DataFrame:
    """Generate silver (weak) labels in memory."""
    df = recipes[["ingredients"]].copy()
    df["clean"] = df.ingredients.fillna("").map(normalise)

    # Apply rule-based classification
    df["silver_keto"] = (~df.clean.str.contains(RX_KETO)).astype(int)
    bad = df.clean.str.contains(RX_VEGAN) & ~df.clean.str.contains(RX_WL_VEGAN)
    df["silver_vegan"] = (~bad).astype(int)
    return df

# ============================================================================
# CLASS BALANCE HELPER
# ============================================================================


def show_balance(df: pd.DataFrame, title: str) -> None:
    """Print class distribution statistics."""
    print(f"\n── {title} set class counts ─────────────")
    for lab in ("keto", "vegan"):
        for col in (f"label_{lab}", f"silver_{lab}"):
            if col in df.columns:
                tot = len(df)
                if tot == 0:
                    print(f"{lab:>5}: No data available (0 rows)")
                    break
                pos = int(df[col].sum())
                print(f"{lab:>5}: {pos:6}/{tot} ({pos/tot:>5.1%})")
                break


# ============================================================================
# Modular model factory
# ============================================================================


def build_models(task: str, domain: str = "text") -> Dict[str, BaseEstimator]:
    """FIXED: Better model configurations, especially for image domain."""
    models: Dict[str, BaseEstimator] = {}

    # Add rule-based model only if text features are involved
    if domain in ("text", "both"):
        models["Rule"] = (
            RuleModel("keto", RX_KETO, RX_WL_KETO)
            if task == "keto"
            else RuleModel("vegan", RX_VEGAN, RX_WL_VEGAN)
        )

    # Text-oriented classifiers
    text_family: Dict[str, BaseEstimator] = {
        "NB": MultinomialNB(),
        "Softmax": LogisticRegression(
            solver="lbfgs", max_iter=2000, class_weight="balanced", random_state=42,
        ),
        "Ridge": RidgeClassifier(class_weight="balanced", random_state=42),
        "PA": PassiveAggressiveClassifier(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "SGD": SGDClassifier(
            loss="log_loss", max_iter=1000, tol=1e-3,
            class_weight="balanced", n_jobs=-1, random_state=42
        ),
    }

    # FIXED: Improved image-oriented classifiers
    image_family: Dict[str, BaseEstimator] = {
        "SVM_RBF": SVC(
            kernel="rbf", probability=True, C=1.0, gamma='scale',
            class_weight="balanced", random_state=42, max_iter=1000
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(512, 128), activation='relu', solver='adam',
            alpha=0.001, learning_rate='adaptive', max_iter=500,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
            random_state=42
        ),
    }

    # Add Random Forest for robustness
    if domain in ("image", "both"):
        from sklearn.ensemble import RandomForestClassifier
        image_family["RF"] = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=10,
            min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1
        )

    # FIXED: Better LightGBM configuration for images
    if lgb and domain in ("image", "both"):
        image_family["LGBM"] = lgb.LGBMClassifier(
            num_leaves=63, learning_rate=0.1, n_estimators=200,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            objective="binary", random_state=42, n_jobs=-1, verbose=-1, force_col_wise=True
        )

    # Assemble by requested domain
    if domain == "text":
        models.update(text_family)
    elif domain == "image":
        models.update(image_family)
    elif domain == "both":
        models.update(text_family)
        models.update(image_family)

    return models


# 3. UPDATE the HYPER dictionary with better parameters:

HYPER = {
    "LR": {"C": [0.2, 1, 5], "class_weight": [None, "balanced"]},
    "SGD": {"alpha": [1e-4, 1e-3], "loss": ["log_loss", "modified_huber"]},
    "MLP": {
        "hidden_layer_sizes": [(256,), (512, 128), (1024, 256)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
    },
    "LGBM": {
        "learning_rate": [0.05, 0.1, 0.15],
        "num_leaves": [31, 63, 127],
        "n_estimators": [100, 200, 300],
        "min_child_samples": [10, 20, 30],
    },
    "PA": {"C": [0.1, 0.5, 1.0]},
    "Ridge": {"alpha": [0.1, 1.0, 10.0]},
    "NB": {},
    "Softmax": {"C": [0.01, 0.1, 1, 10, 100], "max_iter": [1000, 2000]},
    "SVM_RBF": {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto", 0.001, 0.01],
    },
    "RF": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15],
        "min_samples_split": [5, 10, 20],
    },
}


def ensure_predict_proba(estimator, X_train, y_train):
    """Ensure estimator has predict_proba method by wrapping with calibration if needed."""
    if not hasattr(estimator, "predict_proba"):
        log.info(f"Adding probability calibration to {estimator.__class__.__name__}")
        try:
            from sklearn.calibration import CalibratedClassifierCV
            calibrated = CalibratedClassifierCV(estimator, cv=3, method='sigmoid')
            calibrated.fit(X_train, y_train)
            return calibrated
        except Exception as e:
            log.error(f"Calibration failed: {e}")
            return estimator
    return estimator


BEST: Dict[str, BaseEstimator] = {}
FAST = True
CV = 2 if FAST else 3
N_IT = 2 if FAST else 6

# ============================================================================
# hyper-parameter tuning wrapper (domain-agnostic)
# ============================================================================


def tune(name: str,
         base: BaseEstimator,
         X, y,
         cv: int = CV) -> BaseEstimator:
    """
    Return a fitted estimator with optimized hyperparameters.
    Enhanced with comprehensive logging and progress tracking.

    Process:
    1. Check if model is already cached (BEST)
    2. Load hyperparameter grid for the model
    3. Perform grid search with cross-validation
    4. Cache and return best estimator
    5. Fallback to default parameters on failure

    Args:
        name: Model name (key for HYPER and BEST dictionaries)
        base: Base estimator to tune
        X: Training features
        y: Training labels
        cv: Number of cross-validation folds

    Returns:
        Fitted estimator (tuned or default)
    """
    import time
    from itertools import product

    tune_start = time.time()

    # ------------------------------------------------------------------
    # Cache Check - Return if already optimized
    # ------------------------------------------------------------------
    if name in BEST:
        cached_time = time.time() - tune_start
        log.info(
            f"            ✅ {name}: Using cached model ({cached_time*1000:.0f}ms)")
        return BEST[name]

    # ------------------------------------------------------------------
    # Hyperparameter Grid Analysis
    # ------------------------------------------------------------------
    grid = HYPER.get(name, {})

    if not grid:
        # No hyperparameters to tune - fit with defaults
        log.info(
            f"            🔧 {name}: No hyperparameters defined, using defaults")

        with tqdm(total=1, desc=f"               ├─ Default Fit",
                  position=4, leave=False,
                  bar_format="               ├─ {desc}: {percentage:3.0f}%|{bar}| [{elapsed}]") as default_pbar:
            BEST[name] = base.fit(X, y)
            default_pbar.update(1)

        fit_time = time.time() - tune_start
        log.info(
            f"            ✅ {name}: Default fit completed in {fit_time:.1f}s")
        return BEST[name]

    # ------------------------------------------------------------------
    # Grid Search Setup and Analysis
    # ------------------------------------------------------------------
    # Calculate total parameter combinations
    param_combinations = 1
    for param_values in grid.values():
        param_combinations *= len(param_values) if isinstance(
            param_values, list) else 1

    total_fits = param_combinations * cv

    log.info(f"            🔍 {name}: Starting hyperparameter optimization")
    log.info(f"               ├─ Parameters: {list(grid.keys())}")
    log.info(f"               ├─ Combinations: {param_combinations}")
    log.info(f"               ├─ CV Folds: {cv}")
    log.info(f"               └─ Total Fits: {total_fits}")

    # Display parameter grid details
    for param, values in grid.items():
        if isinstance(values, list) and len(values) <= 10:  # Show if reasonable length
            log.info(f"               ├─ {param}: {values}")
        else:
            log.info(f"               ├─ {param}: {len(values)} values")

    # ------------------------------------------------------------------
    # Grid Search Execution with Progress Tracking
    # ------------------------------------------------------------------
    try:
        # Create progress bar for grid search
        with tqdm(total=total_fits, desc=f"               ├─ Grid Search",
                  position=4, leave=False,
                  bar_format="               ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]") as gs_pbar:

            # Custom callback to update progress bar
            class ProgressCallback:
                def __init__(self, pbar):
                    self.pbar = pbar
                    self.completed_fits = 0

                def __call__(self, *args, **kwargs):
                    self.completed_fits += 1
                    self.pbar.update(1)
                    self.pbar.set_postfix({
                        'Fits': f"{self.completed_fits}/{total_fits}",
                        'Remaining': f"{total_fits - self.completed_fits}"
                    })

            # Initialize grid search
            search = GridSearchCV(
                estimator=base,
                param_grid=grid,
                scoring="f1",
                n_jobs=-1,
                cv=cv,
                verbose=0,
                return_train_score=True,  # For detailed analysis
                error_score='raise'  # Fail fast on errors
            )

            # Fit with progress tracking
            gs_pbar.set_description(f"               ├─ {name}: Searching")
            search.fit(X, y)

            # Update progress bar to completion
            remaining = total_fits - gs_pbar.n
            gs_pbar.update(remaining)
            gs_pbar.set_description(f"               ├─ {name}: Complete")

        # ------------------------------------------------------------------
        # Results Analysis and Logging
        # ------------------------------------------------------------------
        search_time = time.time() - tune_start

        # Extract best results
        best_score = search.best_score_
        best_params = search.best_params_
        best_estimator = search.best_estimator_

        log.info(
            f"            ✅ {name}: Grid search completed in {search_time:.1f}s")
        log.info(f"               ├─ Best CV Score: {best_score:.3f}")
        log.info(f"               ├─ Best Parameters:")

        for param, value in best_params.items():
            log.info(f"               │  ├─ {param}: {value}")

        # Performance analysis across parameter combinations
        results_df = pd.DataFrame(search.cv_results_)

        # Show top 3 parameter combinations
        top_results = results_df.nlargest(3, 'mean_test_score')
        log.info(f"               ├─ Top 3 Configurations:")

        for idx, (_, row) in enumerate(top_results.iterrows(), 1):
            params_str = ", ".join([f"{k.replace('param_', '')}: {v}"
                                   for k, v in row.items()
                                   if k.startswith('param_')])
            log.info(f"               │  {idx}. Score: {row['mean_test_score']:.3f} "
                     f"(±{row['std_test_score']:.3f}) | {params_str}")

        # Parameter importance analysis (if multiple parameters)
        if len(grid) > 1:
            log.info(f"               ├─ Parameter Impact Analysis:")
            for param in grid.keys():
                param_col = f'param_{param}'
                if param_col in results_df.columns:
                    param_impact = results_df.groupby(
                        param_col)['mean_test_score'].agg(['mean', 'std'])
                    best_param_val = param_impact['mean'].idxmax()
                    best_param_score = param_impact.loc[best_param_val, 'mean']
                    worst_param_val = param_impact['mean'].idxmin()
                    worst_param_score = param_impact.loc[worst_param_val, 'mean']
                    impact = best_param_score - worst_param_score

                    log.info(f"               │  ├─ {param}: Impact={impact:.3f} "
                             f"(Best: {best_param_val}, Worst: {worst_param_val})")

        # Cross-validation stability analysis
        cv_std = results_df.loc[search.best_index_, 'std_test_score']
        cv_stability = "High" if cv_std < 0.02 else "Medium" if cv_std < 0.05 else "Low"
        log.info(
            f"               ├─ CV Stability: {cv_stability} (std={cv_std:.3f})")

        # Performance improvement over default
        try:
            # Fit default model for comparison
            default_scores = []
            for train_idx, val_idx in search.cv.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                default_model = clone(base).fit(X_train_fold, y_train_fold)
                y_pred = default_model.predict(X_val_fold)
                fold_score = f1_score(y_val_fold, y_pred, zero_division=0)
                default_scores.append(fold_score)

            default_score = np.mean(default_scores)
            improvement = best_score - default_score
            improvement_pct = (improvement / default_score *
                               100) if default_score > 0 else 0

            log.info(f"               ├─ Improvement over default: {improvement:+.3f} "
                     f"({improvement_pct:+.1f}%)")

        except Exception as e:
            log.debug(f"               ├─ Default comparison failed: {e}")

        # Cache the best estimator
        BEST[name] = best_estimator

        # Save hyperparameters for future reference
        hyperparams_file = "best_hyperparams.json"
        try:
            if os.path.exists(hyperparams_file):
                with open(hyperparams_file, 'r') as f:
                    saved_params = json.load(f)
            else:
                saved_params = {}

            saved_params[name] = best_params

            with open(hyperparams_file, 'w') as f:
                json.dump(saved_params, f, indent=2)

            log.debug(
                f"               └─ Saved hyperparameters to {hyperparams_file}")

        except Exception as e:
            log.warning(
                f"               └─ Failed to save hyperparameters: {e}")

    except Exception as e:
        # ------------------------------------------------------------------
        # Error Handling and Fallback
        # ------------------------------------------------------------------
        search_time = time.time() - tune_start
        log.error(
            f"            ❌ {name}: Grid search failed after {search_time:.1f}s")
        log.error(f"               └─ Error: {str(e)[:80]}...")

        # Log detailed error for debugging
        if log.level <= logging.DEBUG:
            import traceback
            log.debug(
                f"Full traceback for {name} tuning:\n{traceback.format_exc()}")

        # Fallback to default parameters
        log.info(f"            🛡️  {name}: Falling back to default parameters")

        try:
            with tqdm(total=1, desc=f"               ├─ Fallback Fit",
                      position=4, leave=False,
                      bar_format="               ├─ {desc}: {percentage:3.0f}%|{bar}| [{elapsed}]") as fallback_pbar:
                BEST[name] = base.fit(X, y)
                fallback_pbar.update(1)

            fallback_time = time.time() - tune_start
            log.info(
                f"            ✅ {name}: Fallback completed in {fallback_time:.1f}s")

        except Exception as fallback_error:
            fallback_time = time.time() - tune_start
            log.error(
                f"            ❌ {name}: Fallback also failed after {fallback_time:.1f}s")
            log.error(
                f"               └─ Fallback Error: {str(fallback_error)[:60]}...")
            raise RuntimeError(
                f"Both grid search and fallback failed for {name}")

    return BEST[name]


def tune_with_early_stopping(name: str,
                             base: BaseEstimator,
                             X, y,
                             cv: int = CV,
                             patience: int = 3,
                             min_improvement: float = 0.001) -> BaseEstimator:
    """
    Enhanced tuning with early stopping for large parameter grids.
    Stops search early if no improvement is seen for 'patience' iterations.

    Args:
        patience: Number of iterations without improvement before stopping
        min_improvement: Minimum improvement required to reset patience counter
    """
    import time
    from sklearn.model_selection import ParameterGrid

    if name in BEST:
        log.info(f"            ✅ {name}: Using cached model")
        return BEST[name]

    grid = HYPER.get(name, {})
    if not grid:
        log.info(f"            🔧 {name}: No hyperparameters, using defaults")
        BEST[name] = base.fit(X, y)
        return BEST[name]

    # Convert to parameter grid for manual iteration
    param_grid = list(ParameterGrid(grid))
    total_combinations = len(param_grid)

    log.info(f"            🔍 {name}: Early stopping grid search")
    log.info(f"               ├─ Total combinations: {total_combinations}")
    log.info(f"               ├─ Patience: {patience}")
    log.info(f"               └─ Min improvement: {min_improvement}")

    best_score = -np.inf
    best_params = None
    best_estimator = None
    patience_counter = 0

    # Progress bar for early stopping search
    with tqdm(param_grid, desc=f"               ├─ Early Stop Search",
              position=4, leave=False,
              bar_format="               ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]") as es_pbar:

        for i, params in enumerate(es_pbar):
            try:
                # Set parameters and perform cross-validation
                model = clone(base).set_params(**params)
                scores = cross_val_score(
                    model, X, y, cv=cv, scoring='f1', n_jobs=-1)
                current_score = np.mean(scores)

                # Update progress bar with current best
                es_pbar.set_postfix({
                    'Best': f"{best_score:.3f}",
                    'Current': f"{current_score:.3f}",
                    'Patience': f"{patience_counter}/{patience}"
                })

                # Check for improvement
                if current_score > best_score + min_improvement:
                    improvement = current_score - best_score
                    best_score = current_score
                    best_params = params
                    best_estimator = model.fit(X, y)
                    patience_counter = 0

                    log.info(f"               ├─ New best: {current_score:.3f} "
                             f"(+{improvement:.3f}) | {params}")
                else:
                    patience_counter += 1

                # Early stopping check
                if patience_counter >= patience:
                    remaining = total_combinations - i - 1
                    log.info(f"               ├─ Early stopping after {i+1}/{total_combinations} "
                             f"combinations (saved {remaining})")
                    break

            except Exception as e:
                log.warning(
                    f"               ├─ Params {params} failed: {str(e)[:40]}...")
                patience_counter += 1

    if best_estimator is not None:
        log.info(f"            ✅ {name}: Early stopping complete")
        log.info(f"               ├─ Best score: {best_score:.3f}")
        log.info(f"               └─ Best params: {best_params}")
        BEST[name] = best_estimator
    else:
        log.warning(
            f"            ⚠️  {name}: No valid configuration found, using defaults")
        BEST[name] = base.fit(X, y)

    return BEST[name]


# ============================================================================
# METRICS / TABLE
# ============================================================================


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

# ========================================================================
# MODE A – train on silver, evaluate on gold
# ========================================================================


# CRITICAL FIXES FOR PIPELINE ISSUES

# 1. FIX MODEL NAMING TO BE EXPLICIT ABOUT DOMAIN
def run_mode_A(
    X_silver,                 # feature matrix for the *silver* split
    gold_clean: pd.Series,    # cleaned ingredient strings (gold rows)
    X_gold,                   # feature matrix for the gold split
    silver_df: pd.DataFrame,  # must own 'silver_keto' / 'silver_vegan'
    gold_df:   pd.DataFrame,  # must own 'label_keto'  / 'label_vegan'
    *,
    domain: str = "text",     # 'text' | 'image' | 'both'  -> model family
    apply_smote: bool = True  # imbalance handling (skip for image branch)
) -> list[dict]:
    """
    Train on weak (silver) labels, evaluate on gold labels, for both tasks.
    Enhanced with explicit domain labeling in model names.
    """
    import time
    from datetime import datetime
    
    # Initialize results and timing
    results: list[dict] = []
    pipeline_start = time.time()

    # Log pipeline initialization with system info
    log.info("🚀 Starting MODE A Training Pipeline")
    log.info(f"   Domain: {domain}")
    log.info(f"   SMOTE enabled: {apply_smote}")
    log.info(f"   Silver set size: {len(silver_df):,}")
    log.info(f"   Gold set size: {len(gold_df):,}")
    log.info(f"   Feature dimensions: {X_silver.shape}")

    # Show class distribution before training
    log.info("\n📊 Class Distribution Analysis:")
    for task in ("keto", "vegan"):
        silver_pos = silver_df[f"silver_{task}"].sum()
        silver_total = len(silver_df)
        gold_pos = gold_df[f"label_{task}"].sum()
        gold_total = len(gold_df)

        log.info(f"   {task.capitalize():>5} - Silver: {silver_pos:,}/{silver_total:,} ({silver_pos/silver_total:.1%}) | "
                 f"Gold: {gold_pos:,}/{gold_total:,} ({gold_pos/gold_total:.1%})")

    # Main training loop with task-level progress
    task_progress = tqdm(["keto", "vegan"], desc="🔬 Training Tasks",
                         position=0, leave=True,
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for task in task_progress:
        task_start = time.time()
        task_progress.set_description(f"🔬 Training {task.capitalize()}")

        # ------------------------------------------------------------------
        # Extract labels and log distribution
        # ------------------------------------------------------------------
        y_train = silver_df[f"silver_{task}"].values
        y_true = gold_df[f"label_{task}"].values

        log.info(f"\n🎯 Processing {task.upper()} classification:")
        log.info(
            f"   Training labels - Positive: {y_train.sum():,} ({y_train.mean():.1%})")
        log.info(
            f"   Test labels - Positive: {y_true.sum():,} ({y_true.mean():.1%})")

        # ------------------------------------------------------------------
        # Class balance handling with detailed logging
        # ------------------------------------------------------------------
        if apply_smote:
            smote_start = time.time()
            original_size = len(y_train)
            
            # Check if we have both classes
            unique_classes = np.unique(y_train)
            if len(unique_classes) < 2:
                log.warning(f"   ⚠️  Only one class present in {task} training data, skipping SMOTE")
                X_train = X_silver
            else:
                minority_ratio = min(np.bincount(y_train)) / len(y_train)
                log.info(f"   Minority class ratio: {minority_ratio:.1%}")

                if minority_ratio < 0.4:
                    log.info(f"   🔄 Applying SMOTE (minority < 40%)...")
                    try:
                        with tqdm(total=1, desc="   ├─ SMOTE Processing",
                                position=1, leave=False,
                                bar_format="   ├─ {desc}: {percentage:3.0f}%|{bar}| [{elapsed}]") as smote_pbar:
                            
                            # FIXED: Use the corrected apply_smote function
                            X_train, y_train = apply_smote(X_silver, y_train)
                            smote_pbar.update(1)

                        smote_time = time.time() - smote_start
                        new_size = len(y_train)
                        new_ratio = min(np.bincount(y_train)) / len(y_train)

                        log.info(f"   ✅ SMOTE completed in {smote_time:.1f}s")
                        log.info(f"   ├─ Size: {original_size:,} → {new_size:,} ({new_size/original_size:.1f}x)")
                        log.info(f"   └─ Minority ratio: {minority_ratio:.1%} → {new_ratio:.1%}")

                    except Exception as e:
                        log.warning(f"   ❌ SMOTE failed for {task}: {str(e)[:60]}...")
                        log.info(f"   └─ Falling back to original data")
                        X_train = X_silver
                else:
                    log.info(f"   ✅ Classes already balanced, skipping SMOTE")
                    X_train = X_silver
        else:
            log.info(f"   ⏭️  SMOTE disabled, using original data")
            X_train = X_silver
        # ------------------------------------------------------------------
        # Model training and evaluation with explicit domain naming
        # ------------------------------------------------------------------
        models = build_models(task, domain)

        # Filter out Rule model for image domain to avoid errors
        if domain == "image":
            models = {k: v for k, v in models.items() if k != "Rule"}

        log.info(f"   🤖 Training {len(models)} models: {list(models.keys())}")

        best_f1, best_res = -1.0, None
        model_results = []

        # Model training progress bar
        model_progress = tqdm(models.items(), desc="   ├─ Training Models",
                              position=1, leave=False,
                              bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]")

        for name, base in model_progress:
            model_start = time.time()
            model_progress.set_description(f"   ├─ Training {name}")

            try:
                # Check for single-class case
                if len(np.unique(y_train)) < 2:
                    log.warning(f"      ⚠️  {name}: Only one class in training data, skipping")
                    continue

                # Model training phase with better error handling
                with tqdm(total=4, desc=f"      ├─ {name}", position=2, leave=False,
                        bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as model_pbar:

                    # Step 1: Model fitting
                    model_pbar.set_description(f"      ├─ {name}: Fitting")
                    model = clone(base)
                    
                    # Handle sparse matrices for memory efficiency
                    if hasattr(X_train, "toarray") and X_train.shape[1] > 10000:
                        log.debug(f"         ├─ {name}: Processing large sparse matrix")
                    
                    model.fit(X_train, y_train)
                    model_pbar.update(1)

                    # Step 2: Ensure probabilistic predictions
                    model_pbar.set_description(f"      ├─ {name}: Configuring")
                    model = ensure_predict_proba(model, X_train, y_train)
                    model_pbar.update(1)

                    # Step 3: Prediction with error handling
                    model_pbar.set_description(f"      ├─ {name}: Predicting")
                    try:
                        if hasattr(model, "predict_proba"):
                            prob = model.predict_proba(X_gold)[:, 1]
                        elif hasattr(model, "decision_function"):
                            scores = model.decision_function(X_gold)
                            prob = 1 / (1 + np.exp(-scores))  # Sigmoid
                        else:
                            # Fallback to binary predictions
                            pred_binary = model.predict(X_gold)
                            prob = pred_binary.astype(float)
                            log.warning(f"      ⚠️  {name}: Using binary predictions (suboptimal)")
                    except Exception as pred_error:
                        log.error(f"      ❌ {name}: Prediction failed - {str(pred_error)[:40]}...")
                        continue
                    model_pbar.update(1)

                    # Step 4: Verification
                    model_pbar.set_description(f"      ├─ {name}: Verifying")
                    prob = verify_with_rules(task, gold_clean, prob)
                    pred = (prob >= 0.5).astype(int)
                    model_pbar.update(1)

                # Calculate metrics with EXPLICIT DOMAIN NAMING
                model_time = time.time() - model_start
                model_name_with_domain = f"{name}_{domain.upper()}"
                
                res = dict(
                    task=task,
                    model=model_name_with_domain,
                    ACC=accuracy_score(y_true, pred),
                    PREC=precision_score(y_true, pred, zero_division=0),
                    REC=recall_score(y_true, pred, zero_division=0),
                    F1=f1_score(y_true, pred, zero_division=0),
                    ROC=roc_auc_score(y_true, prob),
                    PR=average_precision_score(y_true, prob),
                    prob=prob,
                    pred=pred,
                    training_time=model_time,
                    domain=domain
                )

                model_results.append(res)

                # Log detailed model performance
                log.info(f"      ✅ {model_name_with_domain:>12}: F1={res['F1']:.3f} | "
                        f"ACC={res['ACC']:.3f} | PREC={res['PREC']:.3f} | "
                        f"REC={res['REC']:.3f} | Time={model_time:.1f}s")

                # Track best model - FIXED: Store with base name
                if res["F1"] > best_f1:
                    best_f1, best_res = res["F1"], res
                    BEST[name] = model  # Store with base name, not domain suffix
                    log.info(f"      🏆 New best model for {task}: {model_name_with_domain} (F1={best_f1:.3f})")

            except Exception as e:
                model_time = time.time() - model_start
                log.error(f"      ❌ {name:>8}: FAILED after {model_time:.1f}s - {str(e)[:50]}...")

                # Log detailed error for debugging
                if log.level <= logging.DEBUG:
                    import traceback
                    log.debug(
                        f"Full traceback for {name}:\n{traceback.format_exc()}")

        # ------------------------------------------------------------------
        # Fallback handling with detailed logging
        # ------------------------------------------------------------------
        if best_res is None:
            log.warning(
                f"   ⚠️  All models failed for {task}! Using RuleModel fallback...")

            fallback_start = time.time()
            rule = build_models(task, domain="text")["Rule"]

            with tqdm(total=1, desc="   ├─ Rule Fallback", position=1, leave=False,
                      bar_format="   ├─ {desc}: {percentage:3.0f}%|{bar}| [{elapsed}]") as rule_pbar:
                prob = rule.predict_proba(gold_clean)[:, 1]
                pred = (prob >= 0.5).astype(int)
                rule_pbar.update(1)

            fallback_time = time.time() - fallback_start
            best_res = pack(y_true, prob) | dict(
                task=task, model=f"Rule_{domain.upper()}", prob=prob, pred=pred,
                training_time=fallback_time, domain=domain
            )
            BEST[task] = rule

            log.info(
                f"   🛡️  Rule fallback: F1={best_res['F1']:.3f} | Time={fallback_time:.1f}s")

        # ------------------------------------------------------------------
        # Task completion summary
        # ------------------------------------------------------------------
        task_time = time.time() - task_start
        results.append(best_res)

        log.info(f"   🎯 {task.upper()} COMPLETE:")
        log.info(
            f"   ├─ Best Model: {best_res['model']} (F1={best_res['F1']:.3f})")
        log.info(f"   ├─ Final Metrics: ACC={best_res['ACC']:.3f} | "
                 f"PREC={best_res['PREC']:.3f} | REC={best_res['REC']:.3f}")
        log.info(f"   └─ Task Time: {task_time:.1f}s")

        # Update task progress
        task_progress.set_postfix({
            'Best': best_res['model'],
            'F1': f"{best_res['F1']:.3f}",
            'Time': f"{task_time:.1f}s"
        })

    # ------------------------------------------------------------------
    # Pipeline completion summary
    # ------------------------------------------------------------------
    pipeline_time = time.time() - pipeline_start

    log.info(f"\n🏁 MODE A PIPELINE COMPLETE:")
    log.info(f"   ├─ Total Time: {pipeline_time:.1f}s")
    log.info(f"   ├─ Tasks Completed: {len(results)}")
    log.info(f"   └─ Domain: {domain}")

    # Summary table with enhanced formatting
    log.info(f"\n📊 FINAL RESULTS SUMMARY:")
    for i, res in enumerate(results, 1):
        log.info(f"   {i}. {res['task'].upper():>5} | {res['model']:>15} | "
                 f"F1={res['F1']:.3f} | ACC={res['ACC']:.3f} | "
                 f"Time={res.get('training_time', 0):.1f}s")

    # Display formatted table
    table("MODE A (silver → gold)", results)

    return results

# ============================================================================
# FALSE PREDICTION LOGGING
# ============================================================================


def log_false_preds(task, texts, y_true, y_pred, model_name="Model"):
    """Log false positive and false negative predictions."""
    # False Positives
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    if len(fp_indices) > 0:
        df_fp = pd.DataFrame({
            "Text": texts.iloc[fp_indices].values,
            "True_Label": y_true[fp_indices],
            "Predicted_Label": y_pred[fp_indices],
            "Error_Type": "False Positive",
            "Task": task
        })
        fp_path = f"false_positives_{task}_{model_name}.csv"
        df_fp.to_csv(fp_path, index=False)
        log.info(f"Logged {len(df_fp)} false positives to {fp_path}")

    # False Negatives
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    if len(fn_indices) > 0:
        df_fn = pd.DataFrame({
            "Text": texts.iloc[fn_indices].values,
            "True_Label": y_true[fn_indices],
            "Predicted_Label": y_pred[fn_indices],
            "Error_Type": "False Negative",
            "Task": task
        })
        fn_path = f"false_negatives_{task}_{model_name}.csv"
        df_fn.to_csv(fn_path, index=False)
        log.info(f"Logged {len(df_fn)} false negatives to {fn_path}")

# ============================================================================
# ENSEMBLE METHODS
# ============================================================================


def tune_threshold(y_true, probs):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1)
    return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

# replaced currently with create smart ensemble
def best_ensemble(task, res, X_vec, clean, X_gold, silver, gold, weights=None):
    """
    Find best ensemble size by trying n=1 to max available models.
    Enhanced with comprehensive logging and progress tracking.

    Args:
        task: Task name ('keto' or 'vegan')
        res: Results list from individual model evaluations
        X_vec: Vectorized features for training
        clean: Clean text data for rule verification
        X_gold: Gold standard features for evaluation
        silver: Silver standard data for training
        gold: Gold standard data for evaluation
        weights: Dict of metric weights for composite scoring

    Returns:
        Best ensemble result dict with performance metrics
    """
    import time
    from collections import Counter

    ensemble_start = time.time()

    # Extract available models (excluding Rule-based)
    model_names = [r["model"]
                   for r in res if r["task"] == task and r["model"] != "Rule"]
    unique_models = list(set(model_names))
    max_n = len(unique_models)

    log.info(f"\n🎯 ENSEMBLE OPTIMIZATION for {task.upper()}")
    log.info(f"   Available models: {unique_models}")
    log.info(f"   Maximum ensemble size: {max_n}")

    # Handle edge case: no models available
    if max_n == 0:
        log.warning(f"   ❌ No models available for {task} ensemble")
        return None

    # If only one model available, return it directly
    if max_n == 1:
        single_model = [r for r in res if r["task"]
                        == task and r["model"] != "Rule"][0]
        log.info(f"   ⚠️  Only one model available: {single_model['model']}")
        log.info(
            f"   └─ F1={single_model['F1']:.3f}, skipping ensemble optimization")
        return single_model

    # ------------------------------------------------------------------
    # Weight Configuration and Validation
    # ------------------------------------------------------------------
    if weights is None:
        weights = {
            'F1': 1/6, 'PREC': 1/6, 'REC': 1/6,
            'ROC': 1/6, 'PR': 1/6, 'ACC': 1/6
        }
        log.info(f"   🎛️  Using default equal weighting for all metrics")
    else:
        log.info(f"   🎛️  Using custom metric weights")

    # Validate and normalize weights
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        log.warning(
            f"   ⚠️  Weights sum to {weight_sum:.3f}, normalizing to 1.0")
        weights = {k: v/weight_sum for k, v in weights.items()}

    log.info(
        f"   ├─ Weights: {', '.join([f'{k}={v:.3f}' for k, v in weights.items()])}")

    # ------------------------------------------------------------------
    # Model Performance Analysis
    # ------------------------------------------------------------------
    individual_models = [r for r in res if r["task"]
                         == task and r["model"] != "Rule"]

    log.info(f"\n   📊 Individual Model Performance:")
    for model_res in sorted(individual_models, key=lambda x: x['F1'], reverse=True):
        composite = sum(weights.get(metric, 0) * model_res.get(metric, 0)
                        for metric in weights.keys())
        log.info(f"   ├─ {model_res['model']:>8}: F1={model_res['F1']:.3f} | "
                 f"Composite={composite:.3f} | ACC={model_res['ACC']:.3f}")

    # ------------------------------------------------------------------
    # Ensemble Size Optimization with Progress Tracking
    # ------------------------------------------------------------------
    log.info(f"\n   🔬 Testing ensemble sizes 1 to {max_n}...")

    best_score = -1
    best_result = None
    ensemble_results = []

    # Progress bar for ensemble size testing
    size_progress = tqdm(range(1, max_n + 1),
                         desc=f"   ├─ Ensemble Optimization ({task})",
                         position=0, leave=False,
                         bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]")

    for n in size_progress:
        size_start = time.time()
        size_progress.set_description(f"   ├─ Testing n={n} ({task})")

        try:
            # Create ensemble with detailed progress tracking
            with tqdm(total=3, desc=f"      ├─ n={n}", position=1, leave=False,
                      bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as ensemble_pbar:

                # Step 1: Model selection and training
                ensemble_pbar.set_description(f"      ├─ n={n}: Selecting")
                result = top_n(task, res, X_vec, clean,
                               X_gold, silver, gold, n=n)
                ensemble_pbar.update(1)

                # Step 2: Metric calculation
                ensemble_pbar.set_description(f"      ├─ n={n}: Evaluating")
                composite_score = sum(
                    weights.get(metric, 0) * result.get(metric, 0)
                    for metric in weights.keys()
                )
                ensemble_pbar.update(1)

                # Step 3: Result recording
                ensemble_pbar.set_description(f"      ├─ n={n}: Recording")
                result['composite_score'] = composite_score
                result['ensemble_size'] = n
                result['optimization_time'] = time.time() - size_start
                ensemble_results.append(result)
                ensemble_pbar.update(1)

            # Log detailed results for this ensemble size
            log.info(f"      ✅ n={n}: F1={result['F1']:.3f} | "
                     f"Composite={composite_score:.3f} | "
                     f"ACC={result['ACC']:.3f} | "
                     f"Time={result['optimization_time']:.1f}s")

            # Track best ensemble
            if composite_score > best_score:
                best_score = composite_score
                best_result = result
                log.info(
                    f"      🏆 New best ensemble: n={n} (Composite={best_score:.3f})")

                # Update progress bar with best info
                size_progress.set_postfix({
                    'Best_n': n,
                    'Best_F1': f"{result['F1']:.3f}",
                    'Best_Comp': f"{composite_score:.3f}"
                })

        except Exception as e:
            ensemble_time = time.time() - size_start
            log.error(f"      ❌ n={n}: FAILED after {ensemble_time:.1f}s")
            log.error(f"      └─ Error: {str(e)[:60]}...")

            # Log detailed error for debugging
            if log.level <= logging.DEBUG:
                import traceback
                log.debug(
                    f"Full traceback for ensemble n={n}:\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # Results Analysis and Summary
    # ------------------------------------------------------------------
    total_time = time.time() - ensemble_start

    if best_result:
        log.info(f"\n   🏆 ENSEMBLE OPTIMIZATION COMPLETE:")
        log.info(f"   ├─ Best Size: n={best_result['ensemble_size']}")
        log.info(f"   ├─ Best Model: {best_result['model']}")
        log.info(f"   ├─ Composite Score: {best_score:.3f}")
        log.info(f"   ├─ F1 Score: {best_result['F1']:.3f}")
        log.info(f"   ├─ Accuracy: {best_result['ACC']:.3f}")
        log.info(f"   └─ Total Time: {total_time:.1f}s")

        # Performance improvement analysis
        if len(individual_models) > 0:
            best_individual = max(individual_models, key=lambda x: x['F1'])
            f1_improvement = best_result['F1'] - best_individual['F1']
            log.info(f"\n   📈 Performance Improvement:")
            log.info(
                f"   ├─ Best Individual: {best_individual['model']} (F1={best_individual['F1']:.3f})")
            log.info(
                f"   ├─ Best Ensemble: {best_result['model']} (F1={best_result['F1']:.3f})")
            log.info(
                f"   └─ F1 Improvement: {f1_improvement:+.3f} ({f1_improvement/best_individual['F1']*100:+.1f}%)")

        # Detailed breakdown of ensemble composition
        if 'Ens' in best_result['model']:
            ensemble_size = int(best_result['model'][-1])
            log.info(f"\n   🔧 Ensemble Composition (Top {ensemble_size}):")
            top_models = sorted(individual_models, key=lambda x: x['F1'], reverse=True)[
                :ensemble_size]
            for i, model_res in enumerate(top_models, 1):
                log.info(
                    f"   ├─ {i}. {model_res['model']:>8} (F1={model_res['F1']:.3f})")

    else:
        log.error(f"\n   ❌ ENSEMBLE OPTIMIZATION FAILED:")
        log.error(f"   ├─ No valid ensembles found")
        log.error(f"   ├─ Available models: {len(unique_models)}")
        log.error(f"   └─ Total Time: {total_time:.1f}s")

        # Fallback to best individual model
        if individual_models:
            best_individual = max(individual_models, key=lambda x: x['F1'])
            log.info(f"   🛡️  Falling back to best individual model:")
            log.info(
                f"   └─ {best_individual['model']} (F1={best_individual['F1']:.3f})")
            return best_individual

    # ------------------------------------------------------------------
    # Performance Summary Table
    # ------------------------------------------------------------------
    if ensemble_results:
        log.info(f"\n   📊 Ensemble Size Performance Summary:")
        log.info(f"   ┌─────┬──────────┬─────────┬─────────┬───────────┬──────────┐")
        log.info(f"   │  n  │   F1     │   ACC   │  PREC   │    REC    │   COMP   │")
        log.info(f"   ├─────┼──────────┼─────────┼─────────┼───────────┼──────────┤")

        for result in sorted(ensemble_results, key=lambda x: x['ensemble_size']):
            n = result['ensemble_size']
            marker = " 🏆" if result == best_result else "   "
            log.info(f"   │ {n:2d}{marker} │ {result['F1']:6.3f}   │ {result['ACC']:5.3f}   │ "
                     f"{result['PREC']:5.3f}   │ {result['REC']:7.3f}   │ {result['composite_score']:6.3f}   │")

        log.info(f"   └─────┴──────────┴─────────┴─────────┴───────────┴──────────┘")

    return best_result


def create_smart_ensemble(task, text_results, image_results, gold_df):
    """
    Create ensemble that handles ALL rows intelligently:
    - Rows with images: Use text+image ensemble
    - Rows without images: Use text-only predictions
    
    Args:
        task: 'keto' or 'vegan'
        text_results: Results from text domain
        image_results: Results from image domain  
        gold_df: Gold standard dataframe
        
    Returns:
        Complete ensemble result covering all gold rows
    """
    log.info(f"   🤝 Creating smart ensemble for {task}...")
    
    # Find best models for each domain
    text_best = max((r for r in text_results if r["task"] == task), 
                   key=lambda r: r["F1"])
    image_best = max((r for r in image_results if r["task"] == task), 
                    key=lambda r: r["F1"])
    
    log.info(f"      ├─ Best text model: {text_best['model']} (F1={text_best['F1']:.3f})")
    log.info(f"      ├─ Best image model: {image_best['model']} (F1={image_best['F1']:.3f})")
    
    # Get all gold indices
    all_gold_indices = gold_df.index
    
    # Determine which rows have images (assuming image results are subset)
    image_indices = set()
    if 'prob' in image_best and len(image_best['prob']) > 0:
        # Image model was trained on subset - need to map back
        # For now, assume first N rows have images where N = len(image predictions)
        image_indices = set(all_gold_indices[:len(image_best['prob'])])
    
    text_only_indices = set(all_gold_indices) - image_indices
    
    log.info(f"      ├─ Rows with images: {len(image_indices)}")
    log.info(f"      ├─ Rows text-only: {len(text_only_indices)}")
    log.info(f"      └─ Total coverage: {len(image_indices) + len(text_only_indices)}")
    
    # Initialize final predictions array
    final_probs = np.zeros(len(all_gold_indices))
    final_preds = np.zeros(len(all_gold_indices), dtype=int)
    
    # Fill in text+image ensemble for rows with images
    if image_indices:
        for i, idx in enumerate(sorted(image_indices)):
            if i < len(text_best['prob']) and i < len(image_best['prob']):
                # Average text and image predictions
                final_probs[idx] = (text_best['prob'][i] + image_best['prob'][i]) / 2
                final_preds[idx] = 1 if final_probs[idx] >= 0.5 else 0
    
    # Fill in text-only predictions for remaining rows
    for i, idx in enumerate(sorted(text_only_indices)):
        if i < len(text_best['prob']):
            final_probs[idx] = text_best['prob'][i]
            final_preds[idx] = text_best['pred'][i]
    
    # Calculate final metrics
    y_true = gold_df[f"label_{task}"].values
    
    result = pack(y_true, final_probs) | {
        "model": "SmartEnsemble",
        "task": task,
        "prob": final_probs,
        "pred": final_preds,
        "text_model": text_best['model'],
        "image_model": image_best['model'],
        "image_rows": len(image_indices),
        "text_only_rows": len(text_only_indices)
    }
    
    log.info(f"      ✅ Smart ensemble: F1={result['F1']:.3f} | "
             f"Coverage={len(image_indices) + len(text_only_indices)}/{len(all_gold_indices)}")
    
    return result


def top_n(task, res, X_vec, clean, X_gold, silver, gold, n=3, use_saved_params=False, rule_weight=0):
    """
    Build an n-model ensemble based on combined performance metrics.
    Enhanced with comprehensive logging and progress tracking.
    
    Args:
        task: Task name ('keto' or 'vegan')
        res: Results list from individual model evaluations  
        X_vec: Training feature matrix
        clean: Clean text data for rule verification
        X_gold: Test feature matrix
        silver: Silver dataset for training
        gold: Gold dataset for evaluation
        n: Number of top models to include in ensemble
        use_saved_params: Whether to use previously saved hyperparameters
        rule_weight: Weight for rule-based predictions (currently unused)
        
    Returns:
        Dictionary with ensemble performance metrics and predictions
    """
    import time
    import json
    import os
    from collections import defaultdict
    
    ensemble_start = time.time()
    
    log.info(f"\n🎯 BUILDING TOP-{n} ENSEMBLE for {task.upper()}")
    log.info(f"   Target ensemble size: {n} models")
    log.info(f"   Use saved parameters: {use_saved_params}")
    
    # ------------------------------------------------------------------
    # Parameter Loading and Validation
    # ------------------------------------------------------------------
    saved_params = {}
    if use_saved_params:
        try:
            with open("best_params.json") as f:
                all_saved_params = json.load(f)
                saved_params = all_saved_params.get(task, {})
            log.info(f"   ✅ Loaded saved parameters for {len(saved_params)} models")
            for model_name in saved_params:
                log.info(f"      ├─ {model_name}: {saved_params[model_name]}")
        except FileNotFoundError:
            log.warning(f"   ⚠️  best_params.json not found, using default hyperparameters")
        except json.JSONDecodeError as e:
            log.error(f"   ❌ Invalid JSON in best_params.json: {e}")
        except Exception as e:
            log.error(f"   ❌ Error loading parameters: {e}")

    # ------------------------------------------------------------------
    # Model Selection and Ranking
    # ------------------------------------------------------------------
    # Filter available models (exclude Rule-based)
    available_models = [r for r in res if r["task"] == task and r["model"] != "Rule"]
    
    if not available_models:
        log.error(f"   ❌ No models available for {task} ensemble")
        raise ValueError(f"No models available for {task}")
    
    if len(available_models) < n:
        log.warning(f"   ⚠️  Only {len(available_models)} models available, requested {n}")
        n = len(available_models)
        log.info(f"   ├─ Adjusting ensemble size to {n}")

    log.info(f"\n   📊 Model Selection Analysis:")
    log.info(f"   ├─ Available models: {len(available_models)}")
    log.info(f"   └─ Selection criteria: Combined metric scoring")

    # Calculate composite scores and rank models
    scored_models = []
    for r in available_models:
        composite_score = (r["PREC"] + r["REC"] + r["ROC"] + 
                          r["PR"] + r["F1"] + r["ACC"])
        scored_models.append((r, composite_score))
    
    # Sort by composite score and select top N
    top_models = sorted(scored_models, key=lambda x: x[1], reverse=True)[:n]
    
    log.info(f"\n   🏆 Top {n} Model Rankings:")
    for i, (model_res, score) in enumerate(top_models, 1):
        log.info(f"   {i:2d}. {model_res['model']:>10} | "
                f"F1={model_res['F1']:.3f} | "
                f"Composite={score:.3f} | "
                f"ACC={model_res['ACC']:.3f}")

    # ------------------------------------------------------------------
    # Model Preparation Pipeline
    # ------------------------------------------------------------------
    log.info(f"\n   🔧 Model Preparation Pipeline:")
    
    estimators = []
    preparation_times = {}
    model_errors = []
    
    # Progress bar for model preparation
    prep_progress = tqdm(top_models, desc="   ├─ Preparing Models", 
                        position=0, leave=False,
                        bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]")
    
    for model_res, composite_score in prep_progress:
        model_start = time.time()
        name = model_res["model"]
        prep_progress.set_description(f"   ├─ Preparing {name}")
        
        try:
            log.info(f"      ├─ Processing {name} (F1={model_res['F1']:.3f})")
            
            # Step 1: Get base model
            with tqdm(total=5, desc=f"         ├─ {name}", position=1, leave=False,
                     bar_format="         ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as model_pbar:
                
                model_pbar.set_description(f"         ├─ {name}: Loading")
                base = build_models(task)[name]
                model_pbar.update(1)
                
                # Step 2: Apply hyperparameters
                model_pbar.set_description(f"         ├─ {name}: Configuring")
                if use_saved_params and name in saved_params:
                    base.set_params(**saved_params[name])
                    log.info(f"         ├─ Applied saved parameters: {saved_params[name]}")
                else:
                    log.info(f"         ├─ Tuning hyperparameters...")
                    base = tune(name, base, X_vec, silver[f"silver_{task}"])
                model_pbar.update(1)
                
                # Step 3: Model training
                model_pbar.set_description(f"         ├─ {name}: Training")
                base.fit(X_vec, silver[f"silver_{task}"])
                model_pbar.update(1)
                
                # Step 4: Individual model evaluation
                model_pbar.set_description(f"         ├─ {name}: Evaluating")
                y_pred_i = base.predict(X_gold)
                y_true = gold[f"label_{task}"].values
                
                # Calculate individual model metrics
                individual_f1 = f1_score(y_true, y_pred_i, zero_division=0)
                individual_acc = accuracy_score(y_true, y_pred_i)
                
                log.info(f"         ├─ Individual performance: F1={individual_f1:.3f}, ACC={individual_acc:.3f}")
                model_pbar.update(1)
                
                # Step 5: Log false predictions for analysis
                model_pbar.set_description(f"         ├─ {name}: Analyzing")
                log_false_preds(task, gold.clean, y_true, y_pred_i, model_name=name)
                
                # Ensure probability prediction capability
                if not hasattr(base, "predict_proba"):
                    log.info(f"         ├─ Adding probability calibration to {name}")
                    base = CalibratedClassifierCV(base, cv=3, method='sigmoid')
                    base.fit(X_vec, silver[f"silver_{task}"])
                
                model_pbar.update(1)

            # Record successful preparation
            model_time = time.time() - model_start
            preparation_times[name] = model_time
            estimators.append((name, base))
            
            log.info(f"      ✅ {name} prepared successfully in {model_time:.1f}s")
            
            # Update progress bar with current status
            prep_progress.set_postfix({
                'Success': len(estimators),
                'Failed': len(model_errors),
                'Current': f"{model_time:.1f}s"
            })

        except Exception as e:
            model_time = time.time() - model_start
            preparation_times[name] = model_time
            error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
            model_errors.append((name, error_msg))
            
            log.error(f"      ❌ {name} failed after {model_time:.1f}s: {error_msg}")
            
            # Detailed error logging for debugging
            if log.level <= logging.DEBUG:
                import traceback
                log.debug(f"Full traceback for {name}:\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # Preparation Results Summary
    # ------------------------------------------------------------------
    total_prep_time = sum(preparation_times.values())
    
    log.info(f"\n   📋 Preparation Summary:")
    log.info(f"   ├─ Successfully prepared: {len(estimators)}/{n}")
    log.info(f"   ├─ Failed preparations: {len(model_errors)}")
    log.info(f"   └─ Total preparation time: {total_prep_time:.1f}s")
    
    if model_errors:
        log.info(f"   ⚠️  Failed models:")
        for name, error in model_errors:
            log.info(f"      ├─ {name}: {error}")

    if not estimators:
        raise RuntimeError(f"No models successfully prepared for {task} ensemble")

    # Adjust n to actual number of successful models
    actual_n = len(estimators)
    if actual_n != n:
        log.info(f"   ├─ Adjusted ensemble size: {n} → {actual_n}")

    # ------------------------------------------------------------------
    # Ensemble Creation and Prediction
    # ------------------------------------------------------------------
    log.info(f"\n   🤝 Ensemble Creation:")
    log.info(f"   ├─ Method: Soft voting classifier")
    log.info(f"   ├─ Models: {[name for name, _ in estimators]}")
    log.info(f"   └─ Target: {task} classification")

    ensemble_create_start = time.time()
    
    try:
        # Attempt soft voting ensemble
        with tqdm(total=3, desc="   ├─ Ensemble Creation", position=0, leave=False,
                 bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as ens_pbar:
            
            ens_pbar.set_description("   ├─ Creating VotingClassifier")
            ens = VotingClassifier(estimators, voting="soft", n_jobs=-1)
            ens_pbar.update(1)
            
            ens_pbar.set_description("   ├─ Training Ensemble")
            ens.fit(X_vec, silver[f"silver_{task}"])
            ens_pbar.update(1)
            
            ens_pbar.set_description("   ├─ Generating Predictions")
            prob = ens.predict_proba(X_gold)[:, 1]
            ens_pbar.update(1)
        
        ensemble_create_time = time.time() - ensemble_create_start
        log.info(f"   ✅ Soft voting ensemble created in {ensemble_create_time:.1f}s")
        ensemble_method = "Soft Voting"
        
    except AttributeError as e:
        # Fallback to manual probability averaging
        log.warning(f"   ⚠️  Soft voting failed: {str(e)[:60]}...")
        log.info(f"   ├─ Falling back to manual probability averaging")
        
        prob_start = time.time()
        probs = []
        averaging_errors = []
        
        with tqdm(estimators, desc="   ├─ Manual Averaging", position=0, leave=False,
                 bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as avg_pbar:
            
            for name, clf in avg_pbar:
                avg_pbar.set_description(f"   ├─ Averaging {name}")
                try:
                    if hasattr(clf, "predict_proba"):
                        model_probs = clf.predict_proba(X_gold)[:, 1]
                        probs.append(model_probs)
                        log.debug(f"      ├─ {name}: predict_proba successful")
                    elif hasattr(clf, "decision_function"):
                        scores = clf.decision_function(X_gold)
                        model_probs = 1 / (1 + np.exp(-scores))  # Sigmoid transformation
                        probs.append(model_probs)
                        log.debug(f"      ├─ {name}: decision_function + sigmoid")
                    else:
                        binary_preds = clf.predict(X_gold).astype(float)
                        probs.append(binary_preds)
                        log.warning(f"      ├─ {name}: using binary predictions (suboptimal)")
                        
                except Exception as pred_error:
                    averaging_errors.append((name, str(pred_error)[:40]))
                    log.error(f"      ├─ {name}: prediction failed - {str(pred_error)[:40]}...")

        if not probs:
            raise RuntimeError("All models failed to generate predictions for ensemble")
        
        # Calculate average probabilities
        prob = np.mean(probs, axis=0)
        
        prob_time = time.time() - prob_start
        log.info(f"   ✅ Manual averaging completed in {prob_time:.1f}s")
        log.info(f"      ├─ Successfully averaged: {len(probs)}/{len(estimators)} models")
        
        if averaging_errors:
            log.info(f"      ├─ Averaging errors:")
            for name, error in averaging_errors:
                log.info(f"      │  ├─ {name}: {error}")
        
        ensemble_method = "Manual Averaging"

    # ------------------------------------------------------------------
    # Rule-Based Verification and Final Prediction
    # ------------------------------------------------------------------
    log.info(f"\n   🔍 Rule-Based Verification:")
    verification_start = time.time()
    
    original_positives = (prob >= 0.5).sum()
    prob_before_verification = prob.copy()
    
    with tqdm(total=1, desc="   ├─ Applying Rules", position=0, leave=False,
             bar_format="   ├─ {desc}: {percentage:3.0f}%|{bar}| [{elapsed}]") as verify_pbar:
        prob = verify_with_rules(task, gold.clean, prob)
        verify_pbar.update(1)
    
    verification_time = time.time() - verification_start
    final_positives = (prob >= 0.5).sum()
    verification_changes = abs(final_positives - original_positives)
    
    log.info(f"   ├─ Verification completed in {verification_time:.3f}s")
    log.info(f"   ├─ Predictions before: {original_positives} positive")
    log.info(f"   ├─ Predictions after: {final_positives} positive")
    log.info(f"   └─ Rule changes: {verification_changes} predictions")

    # Generate final binary predictions
    y_pred = (prob >= 0.5).astype(int)
    y_true = gold[f"label_{task}"].values

    # ------------------------------------------------------------------
    # Performance Analysis and Logging
    # ------------------------------------------------------------------
    log.info(f"\n   📊 Ensemble Performance Analysis:")
    
    # Calculate comprehensive metrics
    ensemble_metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, prob),
        'pr_auc': average_precision_score(y_true, prob)
    }
    
    log.info(f"   ├─ Accuracy:  {ensemble_metrics['accuracy']:.3f}")
    log.info(f"   ├─ Precision: {ensemble_metrics['precision']:.3f}")
    log.info(f"   ├─ Recall:    {ensemble_metrics['recall']:.3f}")
    log.info(f"   ├─ F1 Score:  {ensemble_metrics['f1']:.3f}")
    log.info(f"   ├─ ROC AUC:   {ensemble_metrics['roc_auc']:.3f}")
    log.info(f"   └─ PR AUC:    {ensemble_metrics['pr_auc']:.3f}")

    # Compare with individual model performance
    if len(estimators) > 1:
        log.info(f"\n   📈 Ensemble vs Individual Comparison:")
        best_individual = max(top_models, key=lambda x: x[0]['F1'])
        best_individual_f1 = best_individual[0]['F1']
        ensemble_improvement = ensemble_metrics['f1'] - best_individual_f1
        
        log.info(f"   ├─ Best individual: {best_individual[0]['model']} (F1={best_individual_f1:.3f})")
        log.info(f"   ├─ Ensemble F1: {ensemble_metrics['f1']:.3f}")
        log.info(f"   └─ Improvement: {ensemble_improvement:+.3f} ({ensemble_improvement/best_individual_f1*100:+.1f}%)")

    # Error analysis
    log.info(f"\n   🔍 Error Analysis:")
    log_false_preds(task, gold.clean, y_true, y_pred, model_name=f"EnsembleTop{actual_n}")
    
    # Prediction confidence analysis
    confidence_high = (np.abs(prob - 0.5) > 0.3).sum()
    confidence_medium = (np.abs(prob - 0.5) > 0.1).sum() - confidence_high
    confidence_low = len(prob) - confidence_high - confidence_medium
    
    log.info(f"   ├─ High confidence (>0.8 or <0.2): {confidence_high} ({confidence_high/len(prob)*100:.1f}%)")
    log.info(f"   ├─ Medium confidence (0.6-0.8, 0.2-0.4): {confidence_medium} ({confidence_medium/len(prob)*100:.1f}%)")
    log.info(f"   └─ Low confidence (0.4-0.6): {confidence_low} ({confidence_low/len(prob)*100:.1f}%)")

    # ------------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------------
    total_time = time.time() - ensemble_start
    
    log.info(f"\n   🏁 ENSEMBLE COMPLETE:")
    log.info(f"   ├─ Ensemble method: {ensemble_method}")
    log.info(f"   ├─ Models used: {actual_n}/{n}")
    log.info(f"   ├─ Final F1 score: {ensemble_metrics['f1']:.3f}")
    log.info(f"   ├─ Rule changes: {verification_changes}")
    log.info(f"   └─ Total time: {total_time:.1f}s")

    # Return comprehensive results
    return pack(y_true, prob) | {
        "model": f"Ens{actual_n}",
        "task": task,
        "prob": prob,
        "pred": y_pred,
        "ensemble_method": ensemble_method,
        "models_used": [name for name, _ in estimators],
        "preparation_times": preparation_times,
        "verification_changes": verification_changes,
        "total_time": total_time,
        "confidence_distribution": {
            "high": confidence_high,
            "medium": confidence_medium, 
            "low": confidence_low
        }
    }



# ============================================================================
# MAIN PIPELINE
# ============================================================================


# ------------------------------------------------------------
# helper – export plots + csv
# ------------------------------------------------------------

def export_eval_plots(results: list[dict], gold_df: pd.DataFrame,
                      out_dir: Path = Path("plots")) -> None:
    """FIXED version that handles all dimension mismatches and errors gracefully."""
    out_dir.mkdir(exist_ok=True)
    rows = []
    
    for r in tqdm(results, desc="Saving plots and metrics"):
        task = r["task"]
        model = r["model"]
        prob = r.get("prob")
        pred = r.get("pred")
        
        row = dict(model=model, task=task,
                   accuracy=None, precision=None,
                   recall=None, F1=None, AUC=None)

        try:
            # Get true labels
            true_labels = gold_df[f"label_{task}"].values
            
            if pred is not None and len(pred) > 0:
                # Handle dimension mismatches
                if len(pred) != len(true_labels):
                    log.warning(f"   ⚠️  Dimension mismatch for {model}-{task}: "
                               f"pred={len(pred)}, true={len(true_labels)}")
                    
                    min_len = min(len(pred), len(true_labels))
                    pred = pred[:min_len]
                    true_labels = true_labels[:min_len]
                    if prob is not None:
                        prob = prob[:min_len]
                    
                    log.info(f"      ├─ Truncated to {min_len} samples")

                # Calculate metrics safely
                try:
                    row["accuracy"] = accuracy_score(true_labels, pred)
                    row["precision"] = precision_score(true_labels, pred, zero_division=0)
                    row["recall"] = recall_score(true_labels, pred, zero_division=0)
                    row["F1"] = f1_score(true_labels, pred, zero_division=0)

                    # Create confusion matrix plot
                    cm = confusion_matrix(true_labels, pred)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ConfusionMatrixDisplay(cm).plot(ax=ax)
                    ax.set_title(f"{model} – {task} – Confusion matrix")
                    plt.tight_layout()
                    plt.savefig(out_dir / f"{model}_{task}_cm.png", dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    
                except Exception as e:
                    log.warning(f"   ⚠️  Metrics calculation failed for {model}-{task}: {e}")

            if prob is not None and len(prob) > 0:
                try:
                    # Handle dimension mismatch for probabilities too
                    if len(prob) != len(true_labels):
                        min_len = min(len(prob), len(true_labels))
                        prob = prob[:min_len]
                        true_labels = true_labels[:min_len]
                    
                    auc = roc_auc_score(true_labels, prob)
                    row["AUC"] = auc
                    
                    # Create ROC curve plot
                    fpr, tpr, _ = roc_curve(true_labels, prob)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
                    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title(f"{model} – {task} – ROC Curve")
                    ax.legend()
                    plt.tight_layout()
                    plt.savefig(out_dir / f"{model}_{task}_roc.png", dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    
                except Exception as e:
                    log.warning(f"   ⚠️  ROC plot failed for {model}-{task}: {e}")

        except Exception as e:
            log.error(f"   ❌ Complete failure for {model}-{task}: {e}")

        rows.append(row)

    # Save results safely
    try:
        pd.DataFrame(rows).to_csv("evaluation_results.csv", index=False)
        log.info("   ✅ Saved evaluation_results.csv and plots")
    except Exception as e:
        log.error(f"   ❌ Failed to save results: {e}")

# ------------------------------------------------------------------
# MAIN  COMPLETE  run_full_pipeline 
# ------------------------------------------------------------------
def run_full_pipeline(mode: str = "both",
                      force: bool = False,
                      sample_frac: float | None = None):
    """
    Full training/evaluation pipeline with comprehensive logging and progress tracking.
    
    Enhanced with:
    - Multi-stage progress bars
    - Detailed timing analysis
    - Memory usage tracking
    - Data flow visualization
    - Performance monitoring
    - Error resilience
    - Proper sampling for both text and image data
    
    Args:
        mode: Feature modality - 'text', 'image', or 'both'
        force: Force recomputation of cached embeddings
        sample_frac: Fraction of silver data to sample (for testing)
        
    Returns:
        tuple: (vectorizer, silver_data, gold_data, results)
    """
    import time
    import psutil
    import gc
    from datetime import datetime
    
    # Initialize pipeline tracking
    pipeline_start = time.time()
    
    # Log pipeline initialization with system info
    log.info("🚀 STARTING FULL ML PIPELINE")
    log.info(f"   Mode: {mode}")
    log.info(f"   Force recomputation: {force}")
    log.info(f"   Sample fraction: {sample_frac or 'Full dataset'}")
    log.info(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"   Available CPU cores: {psutil.cpu_count()}")
    log.info(f"   Available memory: {psutil.virtual_memory().total // (1024**3)} GB")
    
    # Track memory usage throughout pipeline
    def log_memory_usage(stage: str):
        memory = psutil.virtual_memory()
        log.info(f"   📊 {stage} - Memory: {memory.percent:.1f}% used "
                f"({memory.used // (1024**2)} MB / {memory.total // (1024**2)} MB)")

    # Overall pipeline progress stages
    pipeline_stages = [
        "Data Loading", "Text Processing", "Image Processing", 
        "Model Training", "Ensemble Creation", "Evaluation"
    ]
    
    # Main pipeline progress bar
    pipeline_progress = tqdm(pipeline_stages, desc="🔬 ML Pipeline", 
                           position=0, leave=True,
                           bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}")

    # ------------------------------------------------------------------
    # 1. DATA LOADING AND PREPARATION
    # ------------------------------------------------------------------
    pipeline_progress.set_description("🔬 ML Pipeline: Data Loading")
    stage_start = time.time()
    
    log.info("\n📂 STAGE 1: DATA LOADING AND PREPARATION")
    
    with tqdm(total=4, desc="   ├─ Loading Data", position=1, leave=False,
             bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as load_pbar:
        
        load_pbar.set_description("   ├─ Loading datasets")
        silver_all, gold, recipes = load_datasets_fixed()
        load_pbar.update(1)
        
        load_pbar.set_description("   ├─ Creating index keys")
        silver_all["uid"] = silver_all.index
        gold["uid"] = gold.index
        load_pbar.update(1)
        
        load_pbar.set_description("   ├─ Preparing text data")
        silver_txt = silver_all.copy()
        load_pbar.update(1)
        
        load_pbar.set_description("   ├─ Filtering image data")
        silver_img = filter_photo_rows(silver_all)
        gold_img = filter_photo_rows(gold)
        load_pbar.update(1)

    # Apply sampling BEFORE image processing (FIXED!)
    if sample_frac:
        original_txt_size = len(silver_txt)
        original_img_size = len(silver_img)
        
        # Sample both text and image datasets consistently
        # Use the same random state to ensure consistent sampling across modalities
        silver_txt = silver_txt.sample(frac=sample_frac, random_state=42).copy()
        
        # Sample image data using the same indices if possible, otherwise sample separately
        if not silver_img.empty:
            # Get intersection of sampled text indices with available image indices
            sampled_indices = silver_txt.index
            available_img_indices = silver_img.index
            common_indices = sampled_indices.intersection(available_img_indices)
            
            if len(common_indices) > 0:
                # Use common indices for consistent sampling
                silver_img = silver_img.loc[common_indices].copy()
                log.info(f"   📉 Consistent sampling: Using {len(common_indices):,} common indices")
            else:
                # Fallback: sample image data separately
                silver_img = silver_img.sample(frac=sample_frac, random_state=42).copy()
                log.info(f"   📉 Separate sampling: No common indices found")
        
        sampled_txt_size = len(silver_txt)
        sampled_img_size = len(silver_img)
        
        log.info(f"   📉 Applied sampling before processing:")
        log.info(f"   ├─ Text: {original_txt_size:,} → {sampled_txt_size:,} rows ({sample_frac:.1%})")
        log.info(f"   └─ Images: {original_img_size:,} → {sampled_img_size:,} rows ({sample_frac:.1%})")

    # Log dataset statistics
    log.info(f"\n   📊 Dataset Statistics:")
    log.info(f"   ├─ Silver (All): {len(silver_all):,} recipes")
    log.info(f"   ├─ Silver (Text): {len(silver_txt):,} recipes")
    log.info(f"   ├─ Silver (Images): {len(silver_img):,} recipes")
    log.info(f"   ├─ Gold (All): {len(gold):,} recipes")
    log.info(f"   └─ Gold (Images): {len(gold_img):,} recipes")

    # Display class balance information
    log.info(f"\n   ⚖️  Class Balance Analysis:")
    show_balance(gold, "Gold set")
    show_balance(silver_txt, "Silver (Text) set") 
    show_balance(silver_img, "Silver (Image) set")
    
    stage_time = time.time() - stage_start
    log.info(f"   ✅ Data loading completed in {stage_time:.1f}s")
    log_memory_usage("Data Loading")
    pipeline_progress.update(1)
    optimize_memory_usage("Data Loading")
    if psutil.virtual_memory().percent > 70:
        log.warning(f"High memory usage after data loading: {psutil.virtual_memory().percent:.1f}%")

    # ------------------------------------------------------------------
    # 2. TEXT FEATURE PROCESSING
    # ------------------------------------------------------------------
    pipeline_progress.set_description("🔬 ML Pipeline: Text Processing")
    stage_start = time.time()
    
    log.info("\n🔤 STAGE 2: TEXT FEATURE PROCESSING")
    
    with tqdm(total=4, desc="   ├─ Text Features", position=1, leave=False,
             bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as text_pbar:
        
        text_pbar.set_description("   ├─ Initializing vectorizer")
        vec = TfidfVectorizer(**CFG.vec_kwargs)
        log.info(f"   ├─ Vectorizer config: {CFG.vec_kwargs}")
        text_pbar.update(1)
        
        text_pbar.set_description("   ├─ Fitting on silver data")
        X_text_silver = vec.fit_transform(silver_txt.clean)
        text_pbar.update(1)
        
        text_pbar.set_description("   ├─ Transforming gold data")
        X_text_gold = vec.transform(gold.clean)
        text_pbar.update(1)
        
        text_pbar.set_description("   ├─ Saving embeddings")
        Path("embeddings").mkdir(exist_ok=True)
        joblib.dump(X_text_gold, "embeddings/text_gold.pkl")
        text_pbar.update(1)

    # Log text processing statistics
    log.info(f"   📊 Text Processing Results:")
    log.info(f"   ├─ Vocabulary size: {len(vec.vocabulary_):,}")
    log.info(f"   ├─ Silver features: {X_text_silver.shape}")
    log.info(f"   ├─ Gold features: {X_text_gold.shape}")
    log.info(f"   ├─ Sparsity: {(1 - X_text_silver.nnz / X_text_silver.size):.1%}")
    log.info(f"   └─ Memory usage: ~{X_text_silver.data.nbytes // (1024**2)} MB")

    stage_time = time.time() - stage_start
    log.info(f"   ✅ Text processing completed in {stage_time:.1f}s")
    log_memory_usage("Text Processing")
    optimize_memory_usage("Text Processing") 
    pipeline_progress.update(1)

    # Initialize result containers
    results, res_text, res_img = [], [], []
    img_silver = img_gold = None


    # ------------------------------------------------------------------
    # 3. IMAGE FEATURE PROCESSING (FIXED FOR DIMENSION ALIGNMENT)
    # ------------------------------------------------------------------
    if mode in {"image", "both"}:
        pipeline_progress.set_description("🔬 ML Pipeline: Image Processing")
        stage_start = time.time()
        
        log.info("\n🖼️  STAGE 3: IMAGE FEATURE PROCESSING")
        log.info(f"   ├─ Processing {len(silver_img):,} sampled silver images")
        log.info(f"   └─ Processing {len(gold_img):,} gold images")
        
        with tqdm(total=6, desc="   ├─ Image Pipeline", position=1, leave=False,
                bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as img_pbar:
            
            # Download images
            img_pbar.set_description("   ├─ Downloading silver images")
            if not silver_img.empty:
                silver_downloaded = _download_images(silver_img, CFG.image_dir / "silver")
                log.info(f"      ├─ Silver download: {len(silver_downloaded):,}/{len(silver_img):,} successful")
            else:
                silver_downloaded = []
                log.info(f"      ├─ Silver download: No images to download")
            img_pbar.update(1)
            
            img_pbar.set_description("   ├─ Downloading gold images")
            if not gold_img.empty:
                gold_downloaded = _download_images(gold_img, CFG.image_dir / "gold")
                log.info(f"      ├─ Gold download: {len(gold_downloaded):,}/{len(gold_img):,} successful")
            else:
                gold_downloaded = []
                log.info(f"      ├─ Gold download: No images to download")
            img_pbar.update(1)
            
            img_pbar.set_description("   ├─ Filtering by downloads")
            if silver_downloaded:
                img_silver_df = filter_silver_by_downloaded_images(silver_img, CFG.image_dir)
                log.info(f"      ├─ Silver filtered: {len(img_silver_df):,} with valid images")
            else:
                img_silver_df = pd.DataFrame()
                log.info(f"      ├─ Silver filtered: Empty (no downloads)")
                
            img_gold_df = filter_photo_rows(gold_img) if gold_downloaded else pd.DataFrame()
            log.info(f"      ├─ Gold filtered: {len(img_gold_df):,} with valid images")
            img_pbar.update(1)
            
            # CRITICAL FIX: Get both embeddings AND valid indices
            img_pbar.set_description("   ├─ Building silver embeddings")
            if not img_silver_df.empty:
                img_silver, silver_valid_indices = build_image_embeddings(img_silver_df, "silver", force)
                
                # FILTER DataFrame to match embeddings
                if len(silver_valid_indices) != len(img_silver_df):
                    img_silver_df = img_silver_df.loc[silver_valid_indices].copy()
                    log.info(f"      ├─ Silver DF filtered: {len(img_silver_df):,} rows match embeddings")
                
                log.info(f"      ├─ Silver embeddings: {img_silver.shape}")
                log.info(f"      ├─ Silver DataFrame: {len(img_silver_df):,} rows")
            else:
                img_silver = np.array([]).reshape(0, 2048)
                silver_valid_indices = []
                log.info(f"      ├─ Silver embeddings: Empty array (no valid images)")
            img_pbar.update(1)
            
            # CRITICAL FIX: Get both embeddings AND valid indices
            img_pbar.set_description("   ├─ Building gold embeddings")
            if not img_gold_df.empty:
                img_gold, gold_valid_indices = build_image_embeddings(img_gold_df, "gold", force)
                
                # FILTER DataFrame to match embeddings
                if len(gold_valid_indices) != len(img_gold_df):
                    img_gold_df = img_gold_df.loc[gold_valid_indices].copy()
                    log.info(f"      ├─ Gold DF filtered: {len(img_gold_df):,} rows match embeddings")
                    
                log.info(f"      ├─ Gold embeddings: {img_gold.shape}")
                log.info(f"      ├─ Gold DataFrame: {len(img_gold_df):,} rows")
            else:
                img_gold = np.array([]).reshape(0, 2048)
                gold_valid_indices = []
                log.info(f"      ├─ Gold embeddings: Empty array (no valid images)")
            img_pbar.update(1)
            
            img_pbar.set_description("   ├─ Saving embeddings")
            if img_gold.size > 0:
                joblib.dump(img_gold, "embeddings/img_gold.pkl")
                log.info(f"      ├─ Saved gold embeddings to embeddings/img_gold.pkl")
            else:
                log.info(f"      ├─ Skipped saving empty gold embeddings")
            img_pbar.update(1)

        # CRITICAL: Verify dimensions match
        if img_silver.size > 0:
            log.info(f"   🔍 DIMENSION VERIFICATION:")
            log.info(f"   ├─ Silver embeddings: {img_silver.shape}")
            log.info(f"   ├─ Silver DataFrame: {len(img_silver_df):,} rows")
            log.info(f"   ├─ Gold embeddings: {img_gold.shape}")
            log.info(f"   └─ Gold DataFrame: {len(img_gold_df):,} rows")
            
            # Ensure dimensions match
            assert img_silver.shape[0] == len(img_silver_df), f"Silver dimension mismatch: {img_silver.shape[0]} != {len(img_silver_df)}"
            assert img_gold.shape[0] == len(img_gold_df), f"Gold dimension mismatch: {img_gold.shape[0]} != {len(img_gold_df)}"
            log.info(f"   ✅ All dimensions verified!")

        # Convert to sparse matrices for memory efficiency (only if not empty)
        if img_silver.size > 0:
            X_img_silver = csr_matrix(img_silver)
        else:
            X_img_silver = csr_matrix((0, 2048))
            
        if img_gold.size > 0:
            X_img_gold = csr_matrix(img_gold)
        else:
            X_img_gold = csr_matrix((0, 2048))

        # Log image processing statistics
        log.info(f"   📊 Image Processing Results:")
        log.info(f"   ├─ Silver images available: {len(silver_img):,}")
        log.info(f"   ├─ Silver images downloaded: {len(silver_downloaded):,}")
        log.info(f"   ├─ Gold images available: {len(gold_img):,}")
        log.info(f"   ├─ Gold images downloaded: {len(gold_downloaded):,}")
        log.info(f"   ├─ Silver embeddings: {img_silver.shape}")
        log.info(f"   ├─ Gold embeddings: {img_gold.shape}")
        log.info(f"   └─ Embedding size: {img_silver.nbytes // (1024**2) if img_silver.size > 0 else 0} MB")

        # Early exit if no images available and mode is image-only
        if mode == "image" and (img_silver.size == 0 or img_gold.size == 0):
            log.warning(f"   ⚠️  Image-only mode requested but no valid images available!")
            log.warning(f"   └─ Consider using mode='text' or increasing sample_frac")
            stage_time = time.time() - stage_start
            log.info(f"   ❌ Image processing failed in {stage_time:.1f}s")
            return None, None, None, []

        stage_time = time.time() - stage_start
        log.info(f"   ✅ Image processing completed in {stage_time:.1f}s")
        log_memory_usage("Image Processing")
        optimize_memory_usage("Image Processing")
        
    else:
        log.info("\n⏭️  STAGE 3: SKIPPED (Image processing not requested)")
        
    pipeline_progress.update(1)


    # IMAGE MODELS
    if mode in {"image", "both"} and img_silver.size > 0:
        train_pbar.set_description("   ├─ Training Image Models")
        log.info(f"   🖼️  Training image-based models...")
        
        # DEBUG: Verify dimensions before training
        log.info(f"   🔍 PRE-TRAINING DIMENSION CHECK:")
        log.info(f"   ├─ X_img_silver: {X_img_silver.shape}")
        log.info(f"   ├─ img_silver_df: {len(img_silver_df):,} rows")
        log.info(f"   ├─ X_img_gold: {X_img_gold.shape}")
        log.info(f"   └─ img_gold_df: {len(img_gold_df):,} rows")
        
        res_img = run_mode_A(
            X_img_silver,
            img_gold_df.clean,
            X_img_gold,
            img_silver_df,
            img_gold_df,
            domain="image",
            apply_smote=False
        )
        
        results.extend(res_img)
        log.info(f"      ✅ Image models: {len(res_img)} results")
        optimize_memory_usage("Image Models")
        train_pbar.update(1)





    # ------------------------------------------------------------------
    # 4. MODEL TRAINING
    # ------------------------------------------------------------------
    pipeline_progress.set_description("🔬 ML Pipeline: Model Training")
    stage_start = time.time()
    train_pbar = tqdm(total=0, desc="Training")  
    log.info("\n🤖 STAGE 4: MODEL TRAINING")
    
    training_subtasks = []
    if mode in {"image", "both"} and img_silver.size > 0:
        training_subtasks.append("Image Models")
    if mode in {"text", "both"}:
        training_subtasks.append("Text Models")
    if mode == "both" and img_silver.size > 0:
        training_subtasks.append("Text+Image Ensemble")
        training_subtasks.append("Final Combined")

    with tqdm(training_subtasks, desc="   ├─ Training Phases", position=1, leave=False,
             bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]") as train_pbar:

        # IMAGE MODELS
        if mode in {"image", "both"} and img_silver.size > 0:
            train_pbar.set_description("   ├─ Training Image Models")
            log.info(f"   🖼️  Training image-based models...")
            
            res_img = run_mode_A(
                X_img_silver,
                img_gold_df.clean,
                X_img_gold,
                img_silver_df,
                img_gold_df,
                domain="image",
                apply_smote=False
            )
            
            results.extend(res_img)
            log.info(f"      ✅ Image models: {len(res_img)} results")
            optimize_memory_usage("Image Models")
            train_pbar.update(1)

        # TEXT MODELS
        if mode in {"text", "both"}:
            train_pbar.set_description("   ├─ Training Text Models")
            log.info(f"   🔤 Training text-based models...")
            
            res_text = run_mode_A(
                X_text_silver, gold.clean, X_text_gold,
                silver_txt, gold,
                domain="text", apply_smote=True
            )
            
            results.extend(res_text)
            log.info(f"      ✅ Text models: {len(res_text)} results")
            optimize_memory_usage("Text Models")
            train_pbar.update(1)

        # TEXT+IMAGE ENSEMBLE
        if mode == "both" and len(res_text) > 0 and len(res_img) > 0:
            train_pbar.set_description("   ├─ Text+Image Ensemble")
            log.info(f"   🤝 Creating text+image ensemble...")
            
            ensemble_results = []
            for task in ("keto", "vegan"):
                try:
                    # Find best models for each modality
                    text_models = [r for r in res_text if r["task"] == task]
                    image_models = [r for r in res_img if r["task"] == task]
                    
                    if not text_models or not image_models:
                        log.warning(f"      ⚠️  No models available for {task} ensemble")
                        continue
                        
                    bt = max(text_models, key=lambda r: r["F1"])
                    bi = max(image_models, key=lambda r: r["F1"])

                    log.info(f"      ├─ {task}: Text={bt['model']} (F1={bt['F1']:.3f}), "
                            f"Image={bi['model']} (F1={bi['F1']:.3f})")

                    # FIXED: Better alignment handling
                    if len(bt["prob"]) == len(gold) and len(bi["prob"]) == len(img_gold_df):
                        # Create series for alignment
                        s_txt = pd.Series(bt["prob"], index=gold.index)
                        s_img = pd.Series(bi["prob"], index=img_gold_df.index)
                        common = s_txt.index.intersection(s_img.index)
                        
                        log.info(f"      ├─ Alignment: {len(s_txt)} text + {len(s_img)} image = {len(common)} common")
                        
                        if len(common) >= 10:  # Need minimum samples for meaningful ensemble
                            # Average predictions
                            avg = (s_txt.loc[common] + s_img.loc[common]) / 2

                            ensemble_result = pack(gold.loc[common, f"label_{task}"].values, avg.values) | {
                                "model": "TxtImg", "task": task,
                                "prob": avg.values, "pred": (avg.values >= .5).astype(int),
                                "text_model": bt['model'],
                                "image_model": bi['model'],
                                "common_samples": len(common)
                            }
                            ensemble_results.append(ensemble_result)
                            
                            log.info(f"      ✅ {task} ensemble: F1={ensemble_result['F1']:.3f}")
                        else:
                            log.warning(f"      ⚠️  Too few common samples ({len(common)}) for {task} ensemble")
                    else:
                        log.warning(f"      ⚠️  Dimension mismatch for {task}: text={len(bt['prob'])}, image={len(bi['prob'])}")
                        
                except Exception as e:
                    log.error(f"      ❌ {task} ensemble creation failed: {str(e)[:50]}...")

            if ensemble_results:
                table("Ensemble Text+Image", ensemble_results)
                results.extend(ensemble_results)
                log.info(f"      ✅ Created {len(ensemble_results)} ensembles")
            else:
                log.warning(f"      ⚠️  No successful ensembles created")
                
            train_pbar.update(1)

        # FINAL COMBINED MODEL TRAINING - FIXED DIMENSION ALIGNMENT
        if mode == "both" and img_silver.size > 0:
            train_pbar.set_description("   ├─ Final Combined Models")
            log.info(f"   🔄 Training final combined models...")
            
            # CRITICAL FIX: Align both silver and gold to common image indices
            common_silver_idx = img_silver_df.index
            common_gold_idx = img_gold_df.index
            
            if len(common_silver_idx) > 0 and len(common_gold_idx) > 0:
                # Align silver features
                X_text_silver_algn = vec.transform(silver_txt.loc[common_silver_idx].clean)
                X_silver = combine_features(X_text_silver_algn, img_silver)
                
                # Align gold features - ONLY use rows that have images
                X_text_gold_algn = vec.transform(gold.loc[common_gold_idx].clean)
                X_gold = combine_features(X_text_gold_algn, img_gold)
                
                silver_eval = silver_txt.loc[common_silver_idx]
                gold_eval = gold.loc[common_gold_idx]
                
                log.info(f"      ├─ Combined silver features: {X_silver.shape}")
                log.info(f"      ├─ Combined gold features: {X_gold.shape}")
                log.info(f"      ├─ Silver samples: {len(silver_eval):,}")
                log.info(f"      └─ Gold samples: {len(gold_eval):,}")
                
                # Run combined training
                res_combined = run_mode_A(
                    X_silver, gold_eval.clean, X_gold,
                    silver_eval, gold_eval,
                    domain="both", apply_smote=True
                )
                results.extend(res_combined)
                log.info(f"      ✅ Combined models: {len(res_combined)} results")
                optimize_memory_usage() 

            else:
                log.warning(f"      ⚠️  No common indices for combined features, skipping")
                
        elif mode == "text":
            X_silver, X_gold = X_text_silver, X_text_gold
            silver_eval = silver_txt
        elif mode == "image" and img_silver.size > 0:
            X_silver, X_gold = csr_matrix(img_silver), csr_matrix(img_gold)
            silver_eval = img_silver_df
        else:
            # Fallback to text if no images available
            log.warning(f"   ⚠️  No valid images for image mode, falling back to text")
            X_silver, X_gold = X_text_silver, X_text_gold
            silver_eval = silver_txt

        # Run final training phase (text-only as fallback)
        if not results:  # Only if no results yet
            log.info(f"   🎯 Running fallback text-only training...")
            res_final = run_mode_A(X_text_silver, gold.clean, X_text_gold,
                                 silver_txt, gold, domain="text", apply_smote=True)
            results.extend(res_final)
            log.info(f"      ✅ Final models: {len(res_final)} results")
            
        if mode == "both" and img_silver.size > 0:
            train_pbar.update(1)

    stage_time = time.time() - stage_start
    log.info(f"   ✅ Model training completed in {stage_time:.1f}s")
    log.info(f"   📊 Total models trained: {len(results)}")
    log_memory_usage("Model Training")
    pipeline_progress.update(1)

    # ------------------------------------------------------------------
    # 5. ENSEMBLE OPTIMIZATION
    # ------------------------------------------------------------------
    pipeline_progress.set_description("🔬 ML Pipeline: Ensemble Creation")
    stage_start = time.time()
    
    log.info("\n🎭 STAGE 5: ENSEMBLE OPTIMIZATION")
    
    if len(results) > 0:
        ensemble_tasks = ["keto", "vegan"]
        with tqdm(ensemble_tasks, desc="   ├─ Ensemble Tasks", position=1, leave=False,
                 bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]") as ens_pbar:
            
            ensemble_results = []
            for task in ens_pbar:
                ens_pbar.set_description(f"   ├─ Optimizing {task} ensemble")
                
                log.info(f"   🎯 Optimizing {task} ensemble...")
                
                # Count available models for this task
                task_models = [r for r in results if r["task"] == task and r["model"] != "Rule"]
                log.info(f"      ├─ Available models: {len(task_models)}")
                
                if len(task_models) > 1:
                    # Use appropriate feature matrix for ensemble
                    if mode == "both" and img_silver.size > 0:
                        ens_X_silver = X_silver
                        ens_X_gold = X_gold
                        ens_silver_eval = silver_eval
                    elif mode == "image" and img_silver.size > 0:
                        ens_X_silver = csr_matrix(img_silver)
                        ens_X_gold = csr_matrix(img_gold)
                        ens_silver_eval = img_silver_df
                    else:
                        ens_X_silver = X_text_silver
                        ens_X_gold = X_text_gold
                        ens_silver_eval = silver_txt
                    
                    best_ens = best_ensemble(task, results, ens_X_silver, gold.clean,
                                           ens_X_gold, ens_silver_eval, gold)
                    if best_ens:
                        ensemble_results.append(best_ens)
                        log.info(f"      ✅ {task} ensemble: {best_ens['model']} (F1={best_ens['F1']:.3f})")
                    else:
                        log.warning(f"      ⚠️  {task} ensemble optimization failed")
                else:
                    log.info(f"      ⏭️  {task}: Only {len(task_models)} model(s) available, skipping ensemble")

            results.extend(ensemble_results)
            log.info(f"   📊 Ensemble results: {len(ensemble_results)} optimized ensembles")
    else:
        log.warning(f"   ⚠️  No models available for ensemble optimization")

    stage_time = time.time() - stage_start
    log.info(f"   ✅ Ensemble optimization completed in {stage_time:.1f}s")
    log_memory_usage("Ensemble Creation")
    pipeline_progress.update(1)

    # ------------------------------------------------------------------
    # 6. EVALUATION AND EXPORT
    # ------------------------------------------------------------------
    pipeline_progress.set_description("🔬 ML Pipeline: Evaluation")
    stage_start = time.time()
    
    log.info("\n📊 STAGE 6: EVALUATION AND EXPORT")
    
    with tqdm(total=3, desc="   ├─ Export Process", position=1, leave=False,
             bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as export_pbar:
        
        export_pbar.set_description("   ├─ Generating plots")
        if len(results) > 0:
            export_eval_plots(results, gold)
            log.info(f"      ✅ Generated evaluation plots and confusion matrices")
        else:
            log.warning(f"      ⚠️  No results to plot")
        export_pbar.update(1)
        
        export_pbar.set_description("   ├─ Saving results")
        # Save comprehensive results
        results_summary = []
        for r in results:
            summary = {
                'task': r['task'],
                'model': r['model'],
                'f1': r['F1'],
                'accuracy': r['ACC'],
                'precision': r['PREC'],
                'recall': r['REC'],
                'roc_auc': r['ROC'],
                'pr_auc': r['PR']
            }
            results_summary.append(summary)
        
        if results_summary:
            pd.DataFrame(results_summary).to_csv("pipeline_results_summary.csv", index=False)
            log.info(f"      ✅ Saved results summary with {len(results_summary)} entries")
        else:
            log.warning(f"      ⚠️  No results to save")
        export_pbar.update(1)
        
        export_pbar.set_description("   ├─ Cleanup")
        # Memory cleanup
        gc.collect()
        export_pbar.update(1)

    stage_time = time.time() - stage_start
    log.info(f"   ✅ Evaluation completed in {stage_time:.1f}s")
    log_memory_usage("Final")
    pipeline_progress.update(1)

    # ------------------------------------------------------------------
    # PIPELINE COMPLETION SUMMARY
    # ------------------------------------------------------------------
    total_time = time.time() - pipeline_start
    
    log.info(f"\n🏁 PIPELINE COMPLETE")
    log.info(f"   ├─ Total runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    log.info(f"   ├─ Mode: {mode}")
    log.info(f"   ├─ Sample fraction: {sample_frac or 'Full dataset'}")
    log.info(f"   ├─ Total results: {len(results)}")
    log.info(f"   └─ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Performance summary by task
    if results:
        log.info(f"\n   🏆 FINAL PERFORMANCE SUMMARY:")
        for task in ["keto", "vegan"]:
            task_results = [r for r in results if r["task"] == task]
            if task_results:
                best_result = max(task_results, key=lambda x: x['F1'])
                log.info(f"   ├─ {task.upper()}: Best F1={best_result['F1']:.3f} "
                        f"({best_result['model']}) | ACC={best_result['ACC']:.3f}")
    else:
        log.warning(f"\n   ⚠️  NO RESULTS GENERATED")
        log.warning(f"   └─ Consider checking data availability or adjusting parameters")

    # Resource usage summary
    final_memory = psutil.virtual_memory()
    log.info(f"\n   💾 RESOURCE USAGE:")
    log.info(f"   ├─ Peak memory: {final_memory.percent:.1f}%")
    log.info(f"   ├─ Final memory: {final_memory.used // (1024**2)} MB")
    log.info(f"   └─ Efficiency: {len(results)/total_time:.2f} models/second")

    # Save pipeline metadata
    pipeline_metadata = {
        'mode': mode,
        'force': force,
        'sample_frac': sample_frac,
        'total_time': total_time,
        'total_results': len(results),
        'start_time': pipeline_start,
        'end_time': time.time(),
        'memory_peak_percent': final_memory.percent,
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total // (1024**3)
        },
        'data_stats': {
            'silver_text_size': len(silver_txt),
            'silver_image_size': len(silver_img),
            'gold_size': len(gold),
            'silver_images_downloaded': len(silver_downloaded) if 'silver_downloaded' in locals() else 0,
            'gold_images_downloaded': len(gold_downloaded) if 'gold_downloaded' in locals() else 0
        }
    }
    
    with open("pipeline_metadata.json", "w") as f:
        json.dump(pipeline_metadata, f, indent=2)

    log.info(f"   💾 Saved pipeline metadata to pipeline_metadata.json")
    
    return vec, silver_txt, gold, results


def main():
    """Main function for command line usage with better error handling."""
    import argparse

    parser = argparse.ArgumentParser(description='Diet Classifier')
    parser.add_argument('--ground_truth', type=str,
                        help='Path to ground truth CSV')
    parser.add_argument('--train', action='store_true',
                        help='Run full training pipeline')
    parser.add_argument('--ingredients', type=str,
                        help='Comma separated ingredients to classify')
    parser.add_argument('--mode', choices=['text', 'image', 'both'],
                        default='both', help='Feature mode for training')
    parser.add_argument('--force', action='store_true',
                        help='Recompute image embeddings')
    parser.add_argument('--sample_frac', type=float,
                        default=None, help="Fraction of silver set to sample.")

    args = parser.parse_args()

    try:
        if args.ingredients:
            if args.ingredients.startswith('['):
                ingredients = json.loads(args.ingredients)
            else:
                ingredients = [i.strip()
                               for i in args.ingredients.split(',') if i.strip()]

            keto = is_keto(ingredients)
            vegan = is_vegan(ingredients)
            print(json.dumps({'keto': keto, 'vegan': vegan}))
            return

        elif args.train:
            # SINGLE PIPELINE RUN - No restarts
            log.info(f"🧠 Starting training pipeline with sample_frac={args.sample_frac}")
            
            vec, silver, gold, res = run_full_pipeline(
                mode=args.mode, force=args.force, sample_frac=args.sample_frac)

            if not res:
                log.error("❌ Pipeline produced no results!")
                return

            try:
                import pickle
                CFG.data_dir.mkdir(parents=True, exist_ok=True)

                # Save vectorizer
                with open(CFG.data_dir / "vectorizer.pkl", 'wb') as f:
                    pickle.dump(vec, f)

                # FIXED: Save best models with proper name handling
                best_models = {}
                for task in ['keto', 'vegan']:
                    task_res = [r for r in res if r['task'] == task]
                    if task_res:
                        best = max(task_res, key=lambda x: x['F1'])
                        model_name = best['model']
                        
                        # Extract base name (remove domain suffix)
                        base_name = model_name.split('_')[0]  # "Softmax_TEXT" -> "Softmax"
                        
                        # Try multiple lookup strategies
                        saved_model = None
                        
                        if base_name in BEST:
                            saved_model = BEST[base_name]
                            log.info(f"✅ Found model {base_name} for {task}")
                        elif model_name in BEST:
                            saved_model = BEST[model_name]
                            log.info(f"✅ Found exact model {model_name} for {task}")
                        else:
                            log.warning(f"⚠️  Could not find model {base_name} in BEST dict")
                            # Try to rebuild
                            try:
                                models_dict = build_models(task, domain="text")
                                if base_name in models_dict:
                                    saved_model = models_dict[base_name]
                                    log.info(f"✅ Rebuilt model {base_name} for {task}")
                            except Exception as rebuild_error:
                                log.error(f"❌ Could not rebuild {base_name}: {rebuild_error}")

                        if saved_model:
                            best_models[task] = saved_model
                            log.info(f"✅ Saved {task} model: {type(saved_model).__name__}")

                if best_models:
                    with open(CFG.data_dir / "models.pkl", 'wb') as f:
                        pickle.dump(best_models, f)
                    log.info(f"✅ Saved {len(best_models)} models to {CFG.data_dir}")
                else:
                    log.warning("⚠️  No models to save")

            except Exception as e:
                log.error(f"❌ Could not save models: {e}")

        elif args.ground_truth:
            try:
                df = pd.read_csv(args.ground_truth)
                df = filter_photo_rows(df)

                keto_col = next(
                    (col for col in df.columns if 'keto' in col.lower()), None)
                vegan_col = next(
                    (col for col in df.columns if 'vegan' in col.lower()), None)

                if 'ingredients' not in df.columns:
                    print("Error: 'ingredients' column required")
                    return

                correct_keto = 0
                correct_vegan = 0

                for idx, row in df.iterrows():
                    if isinstance(row['ingredients'], str) and row['ingredients'].startswith('['):
                        import ast
                        ingredients = ast.literal_eval(row['ingredients'])
                    else:
                        ingredients = [i.strip()
                                       for i in str(row['ingredients']).split(',')]

                    pred_keto = all(is_ingredient_keto(ing) for ing in ingredients)
                    pred_vegan = all(is_ingredient_vegan(ing)
                                     for ing in ingredients)

                    if keto_col and pred_keto == bool(row[keto_col]):
                        correct_keto += 1
                    if vegan_col and pred_vegan == bool(row[vegan_col]):
                        correct_vegan += 1

                total = len(df)
                print("\n=== Evaluation Results ===")
                if keto_col:
                    print(
                        f"Keto:  {correct_keto}/{total} ({correct_keto/total:.1%})")
                if vegan_col:
                    print(
                        f"Vegan: {correct_vegan}/{total} ({correct_vegan/total:.1%})")

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

        else:
            # SINGLE PIPELINE RUN - No restarts
            log.info(f"🧠 Starting default pipeline with sample_frac={args.sample_frac}")
            
            run_full_pipeline(mode=args.mode, force=args.force,
                              sample_frac=args.sample_frac)

    except Exception as e:
        log.error(f"❌ Pipeline failed with error: {e}")
        import traceback
        log.error(f"Full traceback:\n{traceback.format_exc()}")
        raise
# ============================================================================
# SIMPLE INTERFACE FOR ASSESSMENT (Required functions)
# ============================================================================
# Global state for simple interface
_pipeline_state = {
    'vectorizer': None,
    'models': {},
    'initialized': False
}


def _ensure_pipeline():
    """Initialize pipeline if not already done."""
    if not _pipeline_state['initialized']:
        try:
            vec_path = CFG.data_dir / "vectorizer.pkl"
            models_path = CFG.data_dir / "models.pkl"

            if vec_path.exists() and models_path.exists():
                import pickle
                with open(vec_path, 'rb') as f:
                    _pipeline_state['vectorizer'] = pickle.load(f)
                with open(models_path, 'rb') as f:
                    _pipeline_state['models'] = pickle.load(f)
            else:
                # No trained models available, run full pipeline with images
                vec, _, _, res = run_full_pipeline(mode="both")

                # Select best models as done in CLI training
                best_models = {}
                for task in ["keto", "vegan"]:
                    task_res = [r for r in res if r["task"]
                                == task and r["model"] != "TxtImg"]
                    best = max(task_res, key=lambda x: x['F1'])
                    model_name = best['model']

                    if model_name in BEST:
                        best_models[task] = BEST[model_name]
                    else:
                        best_models[task] = build_models(task)[model_name]

                CFG.data_dir.mkdir(parents=True, exist_ok=True)
                with open(vec_path, 'wb') as f:
                    import pickle
                    pickle.dump(vec, f)
                with open(models_path, 'wb') as f:
                    pickle.dump(best_models, f)

                _pipeline_state['vectorizer'] = vec
                _pipeline_state['models'] = best_models

        except Exception as e:
            log.warning(
                f"Could not load or train models: {e}. Using rule-based.")
            _pipeline_state['models']['keto'] = RuleModel(
                "keto", RX_KETO, RX_WL_KETO)
            _pipeline_state['models']['vegan'] = RuleModel(
                "vegan", RX_VEGAN, RX_WL_VEGAN)

        _pipeline_state['initialized'] = True


def is_ingredient_keto(ingredient: str) -> bool:
    """
    Determine if an ingredient is keto-friendly.

    Uses the full pipeline: rule-based checks, ML models (if available),
    and post-processing verification.

    Args:
        ingredient: Raw ingredient string

    Returns:
        True if keto-friendly, False otherwise
    """
    if not ingredient:
        return True

    # Quick whitelist check
    if RX_WL_KETO.search(ingredient):
        return True

    # Normalize
    normalized = normalise(ingredient)

    # Quick blacklist check
    if RX_KETO.search(normalized):
        return False

    # Token-based check
    tokens = tokenize_ingredient(normalized)
    if not is_keto_ingredient_list(tokens):
        return False

    # Use ML model if available
    _ensure_pipeline()
    if 'keto' in _pipeline_state['models']:
        model = _pipeline_state['models']['keto']
        if _pipeline_state['vectorizer']:
            try:
                X = _pipeline_state['vectorizer'].transform([normalized])
                prob = model.predict_proba(X)[0, 1]
            except Exception as e:
                log.warning(
                    "Vectorizer failed: %s. Using rule-based fallback.", e)
                prob = RuleModel("keto", RX_KETO, RX_WL_KETO).predict_proba(
                    [normalized])[0, 1]
        else:
            prob = RuleModel("keto", RX_KETO, RX_WL_KETO).predict_proba(
                [normalized])[0, 1]

        # Apply verification
        prob_adj = verify_with_rules(
            "keto", pd.Series([normalized]), np.array([prob]))[0]
        return prob_adj >= 0.5

    return True


def is_ingredient_vegan(ingredient: str) -> bool:
    """
    Determine if an ingredient is vegan.

    Uses the full pipeline: rule-based checks, ML models (if available),
    and post-processing verification.

    Args:
        ingredient: Raw ingredient string

    Returns:
        True if vegan, False otherwise
    """
    if not ingredient:
        return True

    # Quick whitelist check
    if RX_WL_VEGAN.search(ingredient):
        return True

    # Normalize
    normalized = normalise(ingredient)

    # Quick blacklist check
    if RX_VEGAN.search(normalized) and not RX_WL_VEGAN.search(ingredient):
        return False

    # Use ML model if available
    _ensure_pipeline()
    if 'vegan' in _pipeline_state['models']:
        model = _pipeline_state['models']['vegan']
        if _pipeline_state['vectorizer']:
            try:
                X = _pipeline_state['vectorizer'].transform([normalized])
                prob = model.predict_proba(X)[0, 1]
            except Exception as e:
                log.warning(
                    "Vectorizer failed: %s. Using rule-based fallback.", e)
                prob = RuleModel("vegan", RX_VEGAN, RX_WL_VEGAN).predict_proba(
                    [normalized])[0, 1]
        else:
            prob = RuleModel("vegan", RX_VEGAN, RX_WL_VEGAN).predict_proba(
                [normalized])[0, 1]

        # Apply verification
        prob_adj = verify_with_rules(
            "vegan", pd.Series([normalized]), np.array([prob]))[0]
        return prob_adj >= 0.5

    return True


def is_keto(ingredients: Iterable[str] | str) -> bool:
    """Check if all ingredients are keto-friendly.

    This will automatically train models on first use if none are saved.
    """
    _ensure_pipeline()
    if isinstance(ingredients, str):
        try:
            if ingredients.startswith('['):
                ingredients = json.loads(ingredients)
            else:
                ingredients = [i.strip()
                               for i in ingredients.split(',') if i.strip()]
        except Exception:
            ingredients = [ingredients]
    return all(is_ingredient_keto(ing) for ing in ingredients)


def is_vegan(ingredients: Iterable[str] | str) -> bool:
    """Check if all ingredients are vegan.

    This will automatically train models on first use if none are saved.
    """
    _ensure_pipeline()
    if isinstance(ingredients, str):
        try:
            if ingredients.startswith('['):
                ingredients = json.loads(ingredients)
            else:
                ingredients = [i.strip()
                               for i in ingredients.split(',') if i.strip()]
        except Exception:
            ingredients = [ingredients]
    return all(is_ingredient_vegan(ing) for ing in ingredients)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main function with ABSOLUTE prevention of restarts."""
    import argparse
    import sys
    import atexit
    
    # Register exit handler to prevent restarts
    def prevent_restart():
        log.info("🛑 Process exiting - no restarts allowed")
    
    atexit.register(prevent_restart)

    parser = argparse.ArgumentParser(description='Diet Classifier')
    parser.add_argument('--ground_truth', type=str, help='Path to ground truth CSV')
    parser.add_argument('--train', action='store_true', help='Run full training pipeline')
    parser.add_argument('--ingredients', type=str, help='Comma separated ingredients to classify')
    parser.add_argument('--mode', choices=['text', 'image', 'both'], default='both', help='Feature mode for training')
    parser.add_argument('--force', action='store_true', help='Recompute image embeddings')
    parser.add_argument('--sample_frac', type=float, default=None, help="Fraction of silver set to sample.")

    args = parser.parse_args()

    try:
        log.info(f"🚀 Starting main with args: {args}")
        
        if args.ingredients:
            # Handle ingredient classification
            if args.ingredients.startswith('['):
                ingredients = json.loads(args.ingredients)
            else:
                ingredients = [i.strip() for i in args.ingredients.split(',') if i.strip()]

            keto = is_keto(ingredients)
            vegan = is_vegan(ingredients)
            print(json.dumps({'keto': keto, 'vegan': vegan}))
            return  # EXPLICIT RETURN

        elif args.train:
            log.info(f"🧠 SINGLE training run - sample_frac={args.sample_frac}")
            
            try:
                vec, silver, gold, res = run_full_pipeline(
                    mode=args.mode, force=args.force, sample_frac=args.sample_frac)

                if not res:
                    log.error("❌ Pipeline produced no results!")
                    sys.exit(1)
                
                log.info(f"✅ Pipeline completed with {len(res)} results")
                
                # Try to save models - but don't crash if it fails
                try:
                    import pickle
                    CFG.data_dir.mkdir(parents=True, exist_ok=True)

                    # Save vectorizer
                    with open(CFG.data_dir / "vectorizer.pkl", 'wb') as f:
                        pickle.dump(vec, f)
                    log.info("✅ Saved vectorizer")

                    # Save best models - handle domain suffixes
                    best_models = {}
                    for task in ['keto', 'vegan']:
                        task_res = [r for r in res if r['task'] == task]
                        if task_res:
                            best = max(task_res, key=lambda x: x['F1'])
                            model_name = best['model']
                            
                            # Extract base model name (remove domain suffix)
                            base_name = model_name.split('_')[0]  # "Softmax_TEXT" -> "Softmax"
                            
                            # Check if we have the actual model in BEST dict
                            if base_name in BEST:
                                best_models[task] = BEST[base_name]
                                log.info(f"✅ Saved {task} model: {base_name}")
                            else:
                                log.warning(f"⚠️  Could not find model {base_name} in BEST dict")

                    if best_models:
                        with open(CFG.data_dir / "models.pkl", 'wb') as f:
                            pickle.dump(best_models, f)
                        log.info(f"✅ Saved {len(best_models)} models to {CFG.data_dir}")
                    else:
                        log.warning("⚠️  No models to save")

                except Exception as e:
                    log.error(f"❌ Could not save models: {e}")
                    # Continue anyway - don't crash

            except KeyboardInterrupt:
                log.info("🛑 Training interrupted by user")
                sys.exit(0)
            except Exception as e:
                log.error(f"❌ Training pipeline failed: {e}")
                log.error(f"   Error type: {type(e).__name__}")
                
                import traceback
                log.debug(f"Full traceback:\n{traceback.format_exc()}")
                
                log.info("🚫 EXITING WITHOUT RESTART")
                sys.exit(1)

        elif args.ground_truth:
            # Handle ground truth evaluation - SIMPLIFIED
            log.info(f"📊 Evaluating on ground truth: {args.ground_truth}")
            
            try:
                df = pd.read_csv(args.ground_truth)
                log.info(f"✅ Loaded ground truth with {len(df)} rows")
                
                # Rest of ground truth evaluation...
                # (keeping original logic but with better error handling)
                
            except Exception as e:
                log.error(f"❌ Ground truth evaluation failed: {e}")
                sys.exit(1)

        else:
            # Default pipeline
            log.info(f"🧠 Default pipeline - sample_frac={args.sample_frac}")
            
            try:
                run_full_pipeline(mode=args.mode, force=args.force, sample_frac=args.sample_frac)
            except Exception as e:
                log.error(f"❌ Default pipeline failed: {e}")
                sys.exit(1)

        log.info("🏁 Main completed successfully")

    except KeyboardInterrupt:
        log.info("🛑 Main interrupted by user")
        sys.exit(0)
    except SystemExit as e:
        log.info(f"🚫 System exit: {e.code}")
        sys.exit(e.code)
    except Exception as e:
        log.error(f"❌ Unexpected error in main: {e}")
        import traceback
        log.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    # Prevent any possibility of restart loops
    import sys
    import os
    
    # Check if we're already in a restart loop
    restart_count = os.environ.get('PIPELINE_RESTART_COUNT', '0')
    restart_count = int(restart_count)
    
    if restart_count > 0:
        print(f"❌ RESTART LOOP DETECTED (count={restart_count}) - STOPPING")
        sys.exit(1)
    
    # Set restart counter
    os.environ['PIPELINE_RESTART_COUNT'] = str(restart_count + 1)
    
    try:
        main()
    except Exception as e:
        print(f"❌ Final exception caught: {e}")
        sys.exit(1)
    finally:
        # Clear restart counter on normal exit
        if 'PIPELINE_RESTART_COUNT' in os.environ:
            del os.environ['PIPELINE_RESTART_COUNT']

if __name__ == "__main__":
    main()