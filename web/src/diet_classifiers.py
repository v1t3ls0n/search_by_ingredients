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

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple
import warnings

# ---------------------------------------------------------------------------
# Optional scikit-learn imports
# ---------------------------------------------------------------------------
try:
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import precision_recall_curve
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.svm import SVC
    from sklearn.base import BaseEstimator, ClassifierMixin, clone
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import VotingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier
    SKLEARN_AVAILABLE = True
except ImportError as e:  # pragma: no cover - optional dependency
    warnings.warn(
        f"scikit-learn not installed ({e}). ML features will be disabled.",
        stacklevel=2,
    )
    SKLEARN_AVAILABLE = False

    class BaseEstimator:  # type: ignore
        pass

    class ClassifierMixin:  # type: ignore
        pass

    def clone(obj):  # type: ignore
        return obj

    def make_pipeline(*args, **kwargs):  # type: ignore
        raise ImportError("scikit-learn is required for this functionality")

    def precision_recall_curve(*args, **kwargs):  # type: ignore
        raise ImportError("scikit-learn is required for this functionality")

    class RBFSampler:  # type: ignore
        pass

    class SVC:  # type: ignore
        pass

    class CalibratedClassifierCV:  # type: ignore
        pass

    class VotingClassifier:  # type: ignore
        pass

    class TfidfVectorizer:  # type: ignore
        def __init__(self, **kwargs):
            raise ImportError("scikit-learn is required for this functionality")

    class LogisticRegression:  # type: ignore
        pass

    LinearSVC = MLPClassifier = GridSearchCV = RandomizedSearchCV = None  # type: ignore
    accuracy_score = average_precision_score = confusion_matrix = f1_score = precision_score = recall_score = roc_auc_score = None  # type: ignore
    SGDClassifier = MultinomialNB = PassiveAggressiveClassifier = RidgeClassifier = None  # type: ignore

import nltk
import json
import logging
import warnings
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from imblearn.over_sampling import SMOTE, RandomOverSampler

# Download NLTK data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    wnl = WordNetLemmatizer()
except:
    wnl = None

# Optional LightGBM
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None

# Optional PyTorch for image embeddings
try:  # pragma: no cover - optional dependency
    from PIL import Image
    import requests
    import torch
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except Exception as e:  # pragma: no cover - optional dependency
    warnings.warn(
        f"PyTorch/torchvision not installed ({e}). Image features disabled.",
        stacklevel=2,
    )
    Image = None  # type: ignore
    requests = None  # type: ignore
    torch = None  # type: ignore
    models = None  # type: ignore
    transforms = None  # type: ignore
    TORCH_AVAILABLE = False

from scipy.sparse import hstack, csr_matrix

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


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load datasets directly into memory without saving them."""
    log.info("Loading datasets into memory")
    recipes = pd.read_parquet(CFG.url_map["allrecipes.parquet"])
    ground_truth = pd.read_csv(CFG.url_map["ground_truth_sample.csv"])
    return recipes, ground_truth


def _download_images(df: pd.DataFrame, split: str) -> None:
    """Download images to the configured image directory."""
    if not TORCH_AVAILABLE:
        return

    img_dir = CFG.image_dir / split
    img_dir.mkdir(parents=True, exist_ok=True)

    if 'photo_url' not in df.columns:
        return

    downloaded = 0  # ← add counter

    for idx, url in df['photo_url'].items():
        f = img_dir / f"{idx}.jpg"
        if f.exists() or not isinstance(url, str) or not url:
            continue
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            with open(f, 'wb') as fh:
                fh.write(resp.content)
            downloaded += 1  # ← increment
        except Exception:
            continue

    log.info(f"Downloaded {downloaded} images to '{img_dir}'")

def build_image_embeddings(df: pd.DataFrame, split: str, force: bool = False) -> np.ndarray:
    """Return or compute image embeddings for the given dataframe."""
    if not TORCH_AVAILABLE:
        return np.zeros((len(df), 2048), dtype=np.float32)

    img_dir = CFG.image_dir / split
    embed_path = img_dir / "embeddings.npy"
    if embed_path.exists() and not force:
        return np.load(embed_path)

    _download_images(df, split)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    vectors = []
    for idx in df.index:
        img_file = img_dir / f"{idx}.jpg"
        if not img_file.exists():
            vectors.append(np.zeros(2048, dtype=np.float32))
            continue
        try:
            img = Image.open(img_file).convert('RGB')
            with torch.no_grad():
                t = preprocess(img).unsqueeze(0).to(device)
                vec = model(t).squeeze().cpu().numpy()
        except Exception:
            vec = np.zeros(2048, dtype=np.float32)
        vectors.append(vec)

    arr = np.vstack(vectors)
    np.save(embed_path, arr)
    return arr


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
    """Apply SMOTE when classes are imbalanced (<40% minority).

    For very large sparse matrices converting to dense can exhaust
    memory.  In such cases we fall back to ``RandomOverSampler`` which
    operates directly on sparse matrices without densifying them.
    ``max_dense_size`` controls the threshold (in number of elements)
    above which the fallback is used.
    """

    counts = np.bincount(y)
    ratio = counts.min() / counts.sum()
    if ratio < 0.4:
        if hasattr(X, "toarray"):
            elements = X.shape[0] * X.shape[1]
            if elements > max_dense_size:
                ros = RandomOverSampler(random_state=42)
                return ros.fit_resample(X, y)
            X = X.toarray()
        smote = SMOTE(random_state=42)
        return smote.fit_resample(X, y)
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
                pos, tot = int(df[col].sum()), len(df)
                print(f"{lab:>5}: {pos:6}/{tot} ({pos/tot:>5.1%})")
                break

# ============================================================================
# MODEL REGISTRY
# ============================================================================


def build_models(task: str) -> Dict[str, BaseEstimator]:
    """Build all available models for classification."""
    m = {
        "Rule": RuleModel("keto", RX_KETO, RX_WL_KETO) if task == "keto" else
        RuleModel("vegan", RX_VEGAN, RX_WL_VEGAN),
        "Softmax": LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "NB": MultinomialNB(),
        "PA": PassiveAggressiveClassifier(max_iter=1000, class_weight="balanced", random_state=42),
        "Ridge": RidgeClassifier(class_weight="balanced", random_state=42),
        "LR": LogisticRegression(solver="lbfgs", max_iter=1000),
        "SGD": SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, class_weight="balanced", n_jobs=-1),
    }

    if lgb:
        m["LGBM"] = lgb.LGBMClassifier(
            num_leaves=15,
            learning_rate=0.3,
            n_estimators=50,
            subsample=0.7,
            colsample_bytree=0.7,
            objective="binary",
            n_jobs=-1,
            random_state=42,
            verbose=-1,  # Suppress LightGBM info messages
            force_col_wise=True  # Avoid the overhead warning
        )

    return m


HYPER = {
    "LR": {"C": [0.2, 1, 5], "class_weight": [None, "balanced"]},
    "SGD": {"alpha": [1e-4, 1e-3]},
    "MLP": {"hidden_layer_sizes": [(40,), (80,), (80, 40)], "alpha": [1e-4, 1e-3]},
    "LGBM": {"learning_rate": [0.05, 0.1], "num_leaves": [31, 63], "n_estimators": [200, 400]},
    "PA": {"C": [0.1, 0.5, 1.0]},
    "Ridge": {"alpha": [0.1, 1.0, 10.0]},
    "NB": {"alpha": [0.5, 1.0, 1.5]},
    "Softmax": {"C": [0.05, 0.2, 1, 5, 10]}

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

    if name in ["NB", "PA", "Ridge", "SVC", "LR", "SGD", "Softmax"]:
        # Define the grid specific to each model
        if name == "SGD":
            grid = {"alpha": [1e-4, 1e-3]}
        elif name == "LR":
            grid = {"C": [0.1, 1, 10], "solver": ["liblinear"]}
        elif name == "Softmax":
            grid = {"C": [0.1, 1.0, 10.0]}  # regularization strength
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

# ============================================================================
# MODE A - TRAIN ON SILVER, EVALUATE ON GOLD
# ============================================================================


def run_mode_A(X_vec, clean, X_gold, silver, gold):
    """Train models on silver labels and evaluate on gold labels."""
    res = []
    for task, col in [("keto", "silver_keto"), ("vegan", "silver_vegan")]:
        ys, yt = silver[col].values, gold[f"label_{task}"].values
        X_os, y_os = apply_smote(X_vec, ys)
        log.info(f"[{task.upper()}] Silver image rows before SMOTE: {len(ys)}")
        log.info(f"[{task.upper()}] Silver image rows after  SMOTE: {len(y_os)}")
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

                # Get probabilities based on model type
                if hasattr(mdl, "predict_proba"):
                    prob = mdl.predict_proba(X_gold)[:, 1]
                elif hasattr(mdl, "decision_function"):
                    # For models with decision_function (like Ridge)
                    scores = mdl.decision_function(X_gold)
                    # Convert to probabilities using sigmoid
                    prob = 1 / (1 + np.exp(-scores))
                else:
                    # Fallback: use predictions as probabilities
                    prob = mdl.predict(X_gold).astype(float)

            prob = verify_with_rules(task, gold.clean, prob)
            r = pack(yt, prob) | {"model": name, "task": task, "prob": prob}
            print(f"\n{name:<7} {task:<5} " + " ".join(f"{m}:{r[m]:>6.2f}" for m in ("ACC", "PREC", "REC", "F1", "ROC", "PR")),
                  f"⏱ {time.time() - t0:.1f}s")
            res.append(r)

    table("MODE A  (silver → gold)", res)
    return res

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


def best_ensemble(task, res, X_vec, clean, X_gold, silver, gold, weights=None):
    """Find best ensemble size by trying n=1 to max available models.

    Args:
        task: Task name
        res: Results list
        X_vec: Vectorized features
        clean: Clean text data
        X_gold: Gold standard features
        silver: Silver standard data
        gold: Gold standard data
        weights: Dict of metric weights (default: equal weighting)
                e.g., {'F1': 0.3, 'PREC': 0.2, 'REC': 0.2, 'ROC': 0.1, 'PR': 0.1, 'ACC': 0.1}
    """
    model_names = [r["model"]
                   for r in res if r["task"] == task and r["model"] != "Rule"]
    max_n = len(set(model_names))

    # Default equal weighting for all metrics
    if weights is None:
        weights = {
            'F1': 1/6,
            'PREC': 1/6,
            'REC': 1/6,
            'ROC': 1/6,
            'PR': 1/6,
            'ACC': 1/6
        }

    # Validate weights sum to 1
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        print(
            f"[WARN] Weights don't sum to 1.0 ({sum(weights.values()):.3f}), normalizing...")
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}

    best_score = -1
    best_result = None

    print(f"\n🔍 Finding best ensemble for {task} using weighted metrics:")
    print(
        f"   Weights: {', '.join([f'{k}={v:.3f}' for k, v in weights.items()])}")

    for n in range(1, max_n + 1):
        try:
            result = top_n(task, res, X_vec, clean, X_gold, silver, gold, n=n)

            # Calculate weighted composite score
            composite_score = sum(
                weights.get(metric, 0) * result.get(metric, 0)
                for metric in weights.keys()
            )

            print(
                f"   n={n}: F1={result['F1']:.3f}, Composite={composite_score:.3f}")

            if composite_score > best_score:
                best_score = composite_score
                best_result = result
                best_result['composite_score'] = composite_score

        except Exception as e:
            print(f"[WARN] Ensemble n={n} failed: {e}")

    if best_result:
        print(
            f"✅ Best ensemble: n={best_result['model'][-1]} with composite score={best_score:.3f}")
    else:
        print("❌ No valid ensemble found")

    return best_result


def top_n(task, res, X_vec, clean, X_gold, silver, gold, n=3, use_saved_params=False, rule_weight=0):
    """Build an n-model ensemble based on combined performance metrics."""

    if use_saved_params:
        with open("best_params.json") as f:
            saved_params = json.load(f).get(task, {})

    # Updated sorting to use the same composite scoring approach
    top_models = sorted(
        [r for r in res if r["task"] == task and r["model"] != "Rule"],
        key=lambda x: x["PREC"] + x["REC"] +
        x["ROC"] + x["PR"] + x["F1"] + x["ACC"],
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
        log_false_preds(task, gold.clean, y_true, y_pred_i, model_name=name)

        # Wrap classifiers that don't have predict_proba
        if not hasattr(base, "predict_proba"):
            # Use CalibratedClassifierCV to add probability estimates
            base = CalibratedClassifierCV(base, cv=3, method='sigmoid')
            base.fit(X_vec, silver[f"silver_{task}"])

        estimators.append((name, base))

    # Create ensemble with soft voting
    try:
        ens = VotingClassifier(estimators, voting="soft", n_jobs=-1)
        ens.fit(X_vec, silver[f"silver_{task}"])
        prob = ens.predict_proba(X_gold)[:, 1]
    except AttributeError:
        # Fallback: average predictions from each model
        probs = []
        for name, clf in estimators:
            if hasattr(clf, "predict_proba"):
                probs.append(clf.predict_proba(X_gold)[:, 1])
            elif hasattr(clf, "decision_function"):
                # Convert decision function to probabilities
                scores = clf.decision_function(X_gold)
                # Sigmoid transformation
                probs.append(1 / (1 + np.exp(-scores)))
            else:
                # Last resort: use binary predictions
                probs.append(clf.predict(X_gold).astype(float))

        prob = np.mean(probs, axis=0)

    prob = verify_with_rules(task, gold.clean, prob)
    y_true = gold[f"label_{task}"].values
    y_pred = (prob >= 0.5).astype(int)

    print(f"\n── False Predictions: Ensemble Top-{n} on {task} ──")
    log_false_preds(task, gold.clean, y_true, y_pred,
                    model_name=f"EnsembleTop{n}")

    return pack(y_true, prob) | {"model": f"Ens{n}", "task": task}
# ============================================================================
# MAIN PIPELINE
# ============================================================================


def run_full_pipeline(mode: str = "both", force: bool = False):
    """Run the complete training and evaluation pipeline.
    Args:
        mode: 'text', 'image', or 'both'. Default 'both' trains both pipelines.
        force: Recompute image embeddings even if cached.
        mode: 'text', 'image', or 'both' to specify feature set.
        force: Recompute image embeddings even if cached.
    """
    
    # Load datasets in memory
    recipes, gold = load_datasets()
    gold["label_keto"] = gold.filter(regex="keto").iloc[:, 0].astype(int)
    gold["label_vegan"] = gold.filter(regex="vegan").iloc[:, 0].astype(int)
    gold["clean"] = gold.ingredients.fillna("").map(normalise)

    silver = build_silver(recipes)
    silver["photo_url"] = recipes.get("photo_url")
    gold["photo_url"] = gold.get("photo_url")

    show_balance(gold, "Gold")
    show_balance(silver, "Silver")

    vec = TfidfVectorizer(**CFG.vec_kwargs)
    X_text_silver = vec.fit_transform(silver.clean)
    X_text_gold = vec.transform(gold.clean)

    results = []
    res_text = []
    res_img = []

    if mode in {"text", "both"}:
        res_text = run_mode_A(X_text_silver, gold.clean, X_text_gold, silver, gold)
        results.extend(res_text)

    if mode in {"image", "both"}:
        img_silver_df = filter_photo_rows(silver)
        img_gold_df = filter_photo_rows(gold)
        img_silver = build_image_embeddings(img_silver_df, "silver", force=force)
        img_gold = build_image_embeddings(img_gold_df, "gold", force=force)
        X_img_silver = csr_matrix(img_silver)
        X_img_gold = csr_matrix(img_gold)
        res_img = run_mode_A(
            X_img_silver, img_gold_df.clean, X_img_gold, img_silver_df, img_gold_df)
        results.extend(res_img)

    if mode == "both" and res_text and res_img:
        ens = []
        for task in ["keto", "vegan"]:
            t = max([r for r in res_text if r["task"] == task],
                    key=lambda x: x["F1"])
            i = max([r for r in res_img if r["task"] == task],
                    key=lambda x: x["F1"])
            avg = (t["prob"] + i["prob"]) / 2
            r = pack(gold[f"label_{task}"].values, avg) | {
                "model": "TxtImg", "task": task}
            ens.append(r)
        table("Ensemble Text+Image", ens)
        results.extend(ens)

# Image embeddings if requested
    if mode in {"image", "both"}:
        img_silver = build_image_embeddings(recipes, "silver", force=force)
        img_gold = build_image_embeddings(gold, "gold", force=force)
    else:
        img_silver = img_gold = None

    if mode == "text":
        X_silver = X_text_silver
        X_gold = X_text_gold
    elif mode == "image":
        X_silver = csr_matrix(img_silver)
        X_gold = csr_matrix(img_gold)
    else:
        X_silver = combine_features(X_text_silver, img_silver)
        X_gold = combine_features(X_text_gold, img_gold)

    # Run mode A
    res = run_mode_A(X_silver, gold.clean, X_gold, silver, gold)

    # Find best ensembles
    res_ens = [
        best_ensemble("keto", res, X_silver, gold.clean, X_gold, silver, gold),
        best_ensemble("vegan", res, X_silver,
                      gold.clean, X_gold, silver, gold),
    ]
    table("MODE A Ensemble (Best)", res_ens)
    with open("best_params.json", "w") as fp:
        json.dump({k: v.get_params() for k, v in BEST.items() if k != "Rule"}, fp, indent=2)

    return vec, silver, gold, results

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
                    task_res = [r for r in res if r["task"] == task and r["model"] != "TxtImg"]
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
                log.warning("Vectorizer failed: %s. Using rule-based fallback.", e)
                prob = RuleModel("keto", RX_KETO, RX_WL_KETO).predict_proba([normalized])[0, 1]
        else:
            prob = RuleModel("keto", RX_KETO, RX_WL_KETO).predict_proba([normalized])[0, 1]

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
                log.warning("Vectorizer failed: %s. Using rule-based fallback.", e)
                prob = RuleModel("vegan", RX_VEGAN, RX_WL_VEGAN).predict_proba([normalized])[0, 1]
        else:
            prob = RuleModel("vegan", RX_VEGAN, RX_WL_VEGAN).predict_proba([normalized])[0, 1]

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
                ingredients = [i.strip() for i in ingredients.split(',') if i.strip()]
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
                ingredients = [i.strip() for i in ingredients.split(',') if i.strip()]
        except Exception:
            ingredients = [ingredients]
    return all(is_ingredient_vegan(ing) for ing in ingredients)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Main function for command line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Diet Classifier')
    parser.add_argument('--ground_truth', type=str,
                        help='Path to ground truth CSV')
    parser.add_argument('--train', action='store_true',
                        help='Run full training pipeline')
    parser.add_argument('--ingredients', type=str,
                        help='Comma separated ingredients to classify')
    # Use both text and image features by default

    parser.add_argument('--mode', choices=['text', 'image', 'both'],
                        default='both', help='Feature mode for training')
    parser.add_argument('--force', action='store_true',
                        help='Recompute image embeddings')
    args = parser.parse_args()

    if args.ingredients:
        if args.ingredients.startswith('['):
            ingredients = json.loads(args.ingredients)
        else:
            ingredients = [i.strip() for i in args.ingredients.split(',') if i.strip()]

        keto = is_keto(ingredients)
        vegan = is_vegan(ingredients)
        print(json.dumps({'keto': keto, 'vegan': vegan}))
        return

    elif args.train:
        # Run full pipeline with selected feature mode
        vec, silver, gold, res = run_full_pipeline(mode=args.mode, force=args.force)

        # Save models
        try:
            import pickle
            CFG.data_dir.mkdir(parents=True, exist_ok=True)

            # Save vectorizer
            with open(CFG.data_dir / "vectorizer.pkl", 'wb') as f:
                pickle.dump(vec, f)

            # Save best models
            best_models = {}
            for task in ['keto', 'vegan']:
                # Get best performing model from results
                task_res = [r for r in res if r['task'] == task]
                best = max(task_res, key=lambda x: x['F1'])
                model_name = best['model']

                if model_name in BEST:
                    best_models[task] = BEST[model_name]
                else:
                    best_models[task] = build_models(task)[model_name]

            with open(CFG.data_dir / "models.pkl", 'wb') as f:
                pickle.dump(best_models, f)

            log.info("Saved trained models to %s", CFG.data_dir)

        except Exception as e:
            log.error("Could not save models: %s", e)

    elif args.ground_truth:
        # Evaluate on ground truth
        try:
            df = pd.read_csv(args.ground_truth)

            # Find columns
            keto_col = next(
                (col for col in df.columns if 'keto' in col.lower()), None)
            vegan_col = next(
                (col for col in df.columns if 'vegan' in col.lower()), None)

            if 'ingredients' not in df.columns:
                print("Error: 'ingredients' column required")
                return

            # Parse ingredients and evaluate
            correct_keto = 0
            correct_vegan = 0

            for idx, row in df.iterrows():
                # Parse ingredients
                if isinstance(row['ingredients'], str) and row['ingredients'].startswith('['):
                    import ast
                    ingredients = ast.literal_eval(row['ingredients'])
                else:
                    ingredients = [i.strip()
                                   for i in str(row['ingredients']).split(',')]

                # Classify (ALL ingredients must pass)
                pred_keto = all(is_ingredient_keto(ing) for ing in ingredients)
                pred_vegan = all(is_ingredient_vegan(ing)
                                 for ing in ingredients)

                # Check accuracy
                if keto_col and pred_keto == bool(row[keto_col]):
                    correct_keto += 1
                if vegan_col and pred_vegan == bool(row[vegan_col]):
                    correct_vegan += 1

            # Report
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
        # Run full pipeline in dev mode
        run_full_pipeline(mode=args.mode, force=args.force)


if __name__ == "__main__":
    main()
