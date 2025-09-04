#!/usr/bin/env python
"""
Flask API:
* /select2, /search, /substitutions, /modify-recipe, /convert-recipes,
  /check-compliance, /export-modified, /save-modified-recipe, /get-modified-recipes, ...
* /api/chat — RAG Chat over recipes (BM25 + KNN + RRF + diet intent)
"""
from __future__ import annotations
import logging
import sys
import os
import re
import json
import sqlite3
import textwrap
import requests
import numpy as np
from time import sleep
from typing import List
from datetime import datetime
from decouple import config
from flask import Flask, jsonify, render_template, request, session
from flask import send_from_directory, abort
from werkzeug.utils import safe_join
from pathlib import Path
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
# ── helpers -------------------------------------------------------
from diet_classifiers import is_keto, is_vegan, diet_score
from utils.query_flags import split_query_flags
from utils.substitutions import (
    format_substitution_text,
    suggest_recipe_substitutions,
    make_recipe_compliant,
    convert_recipe_batch,
    quick_compliance_check,
    export_modified_recipe,
)
import math
import traceback
from typing import Any, Dict, List


ALLOWED_DIETS = {"auto", "", "none", None, "keto", "vegan", "both"}
DATA_ROOT = Path(os.environ.get("DATA_DIR", "/app/data")).resolve()


# ── logging -------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
for noisy in ("opensearchpy", "urllib3", "opensearch"):
    logging.getLogger(noisy).setLevel(logging.ERROR)
log = logging.getLogger(__name__)

# ── Flask & config -----------------------------------------------
app = Flask(__name__)
app.secret_key = config(
    "SECRET_KEY", default="dev-secret-key-change-in-production")

OPENSEARCH_URL = config(
    "OPENSEARCH_URL", default="http://localhost:9200").strip()
OLLAMA_URL = config("OLLAMA_URL",     default="http://localhost:11434").strip()
OLLAMA_MODEL = config("OLLAMA_MODEL",   default="llama3.1:8b").strip()
RECIPES_INDEX = config("RECIPES_INDEX",  default="recipes_v2").strip()
EMBED_MODEL = config(
    "EMBED_MODEL",    default="sentence-transformers/all-MiniLM-L6-v2").strip()


client = OpenSearch(OPENSEARCH_URL, use_ssl=False,
                    verify_certs=False, ssl_show_warn=False)
_embed = SentenceTransformer(EMBED_MODEL)


def _qvec(text: str) -> list[float]:
    return _embed.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()


# ── diet intent detection ----------------------------------------
DIET_PATTERNS = {
    "both":  re.compile(r"\b(keto\s*(and|&)\s*vegan|vegan\s*(and|&)\s*keto|keto-?vegan)\b", re.I),
    "keto":  re.compile(r"\b(keto|low[-\s]?carb|ketogenic)\b", re.I),
    "vegan": re.compile(r"\b(vegan|plant[-\s]?based|no\s*animal\s*products?)\b", re.I),
}


def infer_diet_rule_based(text: str) -> str:
    if DIET_PATTERNS["both"].search(text):
        return "both"
    if DIET_PATTERNS["keto"].search(text):
        if DIET_PATTERNS["vegan"].search(text):
            return "both"
        return "keto"
    if DIET_PATTERNS["vegan"].search(text):
        return "vegan"
    return "none"


def infer_diet_via_llm(text: str, timeout=18) -> str:
    system = ("Classify the user's diet intent for a recipe request. "
              "Respond with exactly one word: none, keto, vegan, or both.")
    user = f"Text: {text}\nAnswer with one of: none|keto|vegan|both"
    payload = {"model": OLLAMA_MODEL, "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ], "stream": False}
    try:
        r = requests.post(f"{OLLAMA_URL.rstrip('/')}/api/chat",
                          json=payload, timeout=timeout)
        r.raise_for_status()
        out = r.json()["message"]["content"].strip().lower()
        if out in {"none", "keto", "vegan", "both"}:
            return out
    except Exception:
        pass
    return "none"


def infer_diet(text: str) -> str:
    rb = infer_diet_rule_based(text)
    return rb if rb != "none" else infer_diet_via_llm(text)

# ── retrieval (BM25 + KNN) & RRF ---------------------------------


def search_bm25(query: str, topn: int = 20):
    body = {"size": topn, "query": {"multi_match": {
        "query": query, "fields": ["title^3", "ingredients^2", "instructions", "description"]
    }}}
    res = client.search(index=RECIPES_INDEX, body=body)
    hits = res["hits"]["hits"]
    for h in hits:
        h["_score"] = float(h.get("_score") or 0.0)
    return hits


def search_knn(query: str, topn: int = 20):
    qv = _qvec(query)
    body = {"size": topn, "query": {
        "knn": {"embedding": {"vector": qv, "k": topn}}}}
    res = client.search(index=RECIPES_INDEX, body=body)
    hits = res["hits"]["hits"]
    for h in hits:
        h["_score"] = float(h.get("_score") or 0.0)
    return hits


def rrf_fuse(bm25_hits, knn_hits, k=60):
    def ranks(hits): return {h["_id"]: i for i, h in enumerate(hits)}
    r_b, r_k = ranks(bm25_hits), ranks(knn_hits)
    ids = set(r_b) | set(r_k)
    fused = []
    for _id in ids:
        rb, rk = r_b.get(_id, 10**6), r_k.get(_id, 10**6)
        score = 1.0/(k+rb) + 1.0/(k+rk)
        src = next((h["_source"] for h in bm25_hits if h["_id"] == _id), None) \
            or next((h["_source"] for h in knn_hits if h["_id"] == _id), None)
        fused.append({"_id": _id, "_source": src, "_score": score})
    fused.sort(key=lambda x: x["_score"], reverse=True)
    return fused


def _diet_filter_pred(doc, diet: str, threshold: float) -> bool:
    thr = int(max(0.0, min(1.0, threshold)) * 100)
    ks, vs = doc.get("keto_score"), doc.get("vegan_score")
    if diet == "keto":
        return ks is None or ks >= thr
    if diet == "vegan":
        return vs is None or vs >= thr
    if diet == "both":
        return (ks is None or ks >= thr) and (vs is None or vs >= thr)
    return True


def hybrid_rrf_search(query: str, topk: int = 5, diet: str | None = None, threshold: float = 1.0):
    bm25 = search_bm25(query, topn=max(20, topk*4))
    knn = search_knn(query,  topn=max(20, topk*4))
    fused = rrf_fuse(bm25, knn, k=60)
    if diet and diet != "none":
        fused = [h for h in fused if _diet_filter_pred(
            h["_source"], diet, threshold)]
    return fused[:topk]

# ── RAG helpers & endpoint ---------------------------------------


def _to_str_block(x):
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return "\n".join(str(i) for i in x)
    return str(x)


def _build_context(hits):
    parts = []
    for i, h in enumerate(hits, 1):
        s = h["_source"]
        title = s.get("title", "")
        ings = _to_str_block(s.get("ingredients", ""))
        instr = _to_str_block(s.get("instructions", ""))
        keto = s.get("keto_score")
        vegan = s.get("vegan_score")
        keto_txt = f"{int(keto)}%" if isinstance(keto, (int, float)) else "n/a"
        vegan_txt = f"{int(vegan)}%" if isinstance(
            vegan, (int, float)) else "n/a"
        meta = f"[{i}] {title} (keto:{keto_txt}, vegan:{vegan_txt})"
        snippet = textwrap.shorten(
            f"Ingredients: {ings}  Instructions: {instr}",
            width=1000,
            placeholder=" …"
        )
        parts.append(meta + "\n" + snippet)
    return "\n\n".join(parts)


def _ollama_answer(question: str, context: str) -> str:
    system = ("You are a concise cooking assistant. Use ONLY the provided recipes. "
              "If information is missing, say 'Not found in the sources.' "
              "Return a short, clear answer in English and end with sources [n].")
    payload = {"model": OLLAMA_MODEL, "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}\n\nRecipes:\n{context}"}
    ], "stream": False}
    r = requests.post(f"{OLLAMA_URL.rstrip('/')}/api/chat",
                      json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


@app.route("/data/<path:subpath>")
def serve_data(subpath: str):
    """
    Serve files from /app/data securely.
    Allows URLs like /data/clean/images/<file>.jpg
    """
    # prevent path traversal
    safe_path = safe_join(str(DATA_ROOT), subpath)
    if not safe_path:
        abort(403)
    full = Path(safe_path).resolve()
    if not str(full).startswith(str(DATA_ROOT)):
        abort(403)
    if not full.exists():
        abort(404)

    # optional: add simple caching
    resp = send_from_directory(
        str(DATA_ROOT), subpath, conditional=True, max_age=60*60*24)
    return resp


def _j(code: int, payload: Dict[str, Any]):
    resp = jsonify(payload)
    resp.status_code = code
    resp.headers.setdefault("Content-Type", "application/json; charset=utf-8")
    resp.headers.setdefault("Cache-Control", "no-store")
    return resp


@app.route("/api/chat", methods=["GET", "POST"])
def api_chat():
    """
    Robust RAG chat endpoint.
    - POST requires application/json; GET reads query params.
    - Validates diet/threshold/topk.
    - Catches and downgrades failures in inference/search/context/LLM so the UI never sees a blank 500.
    """
    try:
        # ------------- parse input -------------
        if request.method == "POST":
            ct = (request.headers.get("Content-Type") or "").lower()
            if not (ct.startswith("application/json") or request.is_json):
                return _j(415, {"error_code": "unsupported_media_type",
                                "message": "POST requires application/json"})
            data = request.get_json(silent=True)
            if not isinstance(data, dict):
                return _j(400, {"error_code": "invalid_json", "message": "Malformed JSON body"})
            query = (data.get("query") or "").strip()
            client_diet = (data.get("diet") or "auto").strip().lower()
            try:
                threshold = float(data.get("threshold", 1.0))
            except Exception:
                return _j(422, {"error_code": "bad_threshold", "message": "threshold must be a number in [0,1]"})
            try:
                topk = int(data.get("topk", 5))
            except Exception:
                return _j(422, {"error_code": "bad_topk", "message": "topk must be an integer in [1,10]"})
        else:
            query = (request.args.get("query") or "").strip()
            client_diet = (request.args.get("diet") or "auto").strip().lower()
            try:
                threshold = float(request.args.get("threshold", 1.0))
            except Exception:
                return _j(422, {"error_code": "bad_threshold", "message": "threshold must be a number in [0,1]"})
            try:
                topk = int(request.args.get("topk", 5))
            except Exception:
                return _j(422, {"error_code": "bad_topk", "message": "topk must be an integer in [1,10]"})

        if not query:
            return _j(400, {"error_code": "missing_query", "message": "query is required"})

        if client_diet not in ALLOWED_DIETS:
            return _j(422, {"error_code": "bad_diet",
                            "message": "diet must be one of: auto, '', none, keto, vegan, both"})

        if not (0.0 <= threshold <= 1.0 or math.isclose(threshold, 0.0) or math.isclose(threshold, 1.0)):
            return _j(422, {"error_code": "bad_threshold", "message": "threshold must be in [0,1]"})

        if not (1 <= topk <= 10):
            return _j(422, {"error_code": "bad_topk", "message": "topk must be in [1,10]"})

        # ------------- diet inference -------------
        try:
            inferred = (
                infer_diet(query)
                if client_diet in {"auto", "", "none", None}
                else client_diet
            )
            if inferred not in {"keto", "vegan", "both", "none"}:
                # sanitize unexpected model outputs
                inferred = "none"
        except Exception as e:
            app.logger.exception("infer_diet failed: %s", e)
            inferred = "none"  # degrade — do search without diet gating

        # ------------- search -------------
        try:
            hits = hybrid_rrf_search(
                query,
                topk=topk,
                diet=(inferred if inferred != "none" else None),
                threshold=threshold,
            )
        except Exception as e:
            app.logger.exception("hybrid_rrf_search failed: %s", e)
            return _j(502, {"error_code": "search_failed",
                            "message": "Search backend failed"})

        if not hits:
            return _j(200, {"answer": "Not found in the sources.",
                            "sources": [], "diet_inferred": inferred})

        # ------------- context -------------
        try:
            context = _build_context(hits)
        except Exception as e:
            app.logger.exception("_build_context failed: %s", e)
            # fallback: build a tiny context from titles/ingredients
            parts: List[str] = []
            for h in hits:
                s = (h or {}).get("_source", {}) or {}
                parts.append(
                    f"{s.get('title', '')}\nIngredients:\n{(s.get('ingredients') or '')}")
            context = "\n\n---\n\n".join(parts[:topk])

        # ------------- model answer -------------
        try:
            answer = _ollama_answer(query, context)
        except Exception as e:
            app.logger.exception("_ollama_answer failed: %s", e)
            # graceful fallback: stitch an answer stub + top sources
            titles = []
            for h in hits:
                s = (h or {}).get("_source", {}) or {}
                titles.append(s.get("title") or s.get("id") or "recipe")
            answer = (
                "I couldn't generate a full answer right now, "
                "but here are relevant recipes you can open:\n- "
                + "\n- ".join(titles[:topk])
            )

        # ------------- format sources -------------
        sources = []
        for i, h in enumerate(hits, 1):
            s = (h or {}).get("_source", {}) or {}
            sources.append({
                "ref": i,
                "id": h.get("_id"),
                "title": s.get("title"),
                "keto_score": s.get("keto_score"),
                "vegan_score": s.get("vegan_score"),
            })

        return _j(200, {"answer": answer, "sources": sources, "diet_inferred": inferred})

    except Exception as e:
        # Final safety net — should rarely trigger now
        app.logger.error("api_chat unexpected failure: %s\n%s",
                         e, traceback.format_exc())
        return _j(500, {"error_code": "internal_error", "message": "internal server error"})


def _wait(os_client: OpenSearch, tries: int = 30, delay: int = 2) -> bool:
    for _ in range(tries):
        if os_client.ping():
            return True
        sleep(delay)
    return False


def _os_ready(os_client: OpenSearch, tries: int = 30, delay: int = 2) -> bool:
    for _ in range(tries):
        try:
            if os_client.ping():
                return True
        except Exception:
            pass
        sleep(delay)
    return False


def _bootstrap():
    os_client = OpenSearch(
        hosts=[OPENSEARCH_URL],
        use_ssl=False, verify_certs=False, ssl_show_warn=False
    )

    # Wait for OS
    if not _os_ready(os_client):
        log.error("OpenSearch not reachable at %s", OPENSEARCH_URL)
        sys.exit(1)

    # Ensure 'ingredients' index exists (empty is fine)
    try:
        if not os_client.indices.exists(index="ingredients"):
            os_client.indices.create(
                index="ingredients",
                body={"mappings": {"properties": {"ingredients": {"type": "text"}}}}
            )
            log.info(
                "Created empty 'ingredients' index (will be filled by indexer).")
    except Exception as e:
        log.warning("Could not ensure 'ingredients' index exists: %s", e)

    # Load tokens (gracefully handle empty)
    try:
        resp = os_client.search(
            index="ingredients",
            body={"query": {"match_all": {}}},
            size=10_000
        )
        vocab: List[str] = [h["_source"]["ingredients"]
                            for h in resp["hits"]["hits"]]
    except Exception as e:
        log.warning(
            "Ingredients search failed (continuing with empty vocab): %s", e)
        vocab = []

    log.info("Loaded %s ingredient tokens", len(vocab))
    return os_client, vocab


# Initialize ingredients vocabulary (used by /select2)
client, INGREDIENTS = _bootstrap()


# ── SQLite database for modified recipes -------------------------


def init_db():
    """Initialize SQLite database for storing modified recipes."""
    conn = sqlite3.connect('modified_recipes.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS modified_recipes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recipe_id TEXT NOT NULL,
            title TEXT NOT NULL,
            original_ingredients TEXT NOT NULL,
            modified_ingredients TEXT NOT NULL,
            diet_modification TEXT NOT NULL,
            substitutions_applied TEXT NOT NULL,
            changes_made INTEGER NOT NULL,
            original_keto_score REAL,
            original_vegan_score REAL,
            new_keto_score REAL,
            new_vegan_score REAL,
            modification_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description TEXT,
            instructions TEXT,
            photo_url TEXT,
            unique_id TEXT UNIQUE
        )
    ''')

    conn.commit()
    conn.close()
    log.info("SQLite database initialized for modified recipes")


# Initialize database
init_db()

# ── routes --------------------------------------------------------


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/select2")
def select2():
    """
    Autocomplete endpoint for Select2.
    Handles both ingredient searches and diet tags (#keto, #vegan, #both).
    """
    query = request.args.get("q", "").strip()

    if not query:
        return jsonify({"results": []})

    results = []

    # Check if this is a tag query
    if query.startswith("#"):
        # Handle diet tags
        tag_lower = query[1:].lower()  # Remove # and lowercase

        # Check if user is typing a custom threshold
        import re
        threshold_match = re.match(
            r'^(keto|vegan|both)(?::(\d*\.?\d*)?)?', tag_lower)

        if threshold_match:
            diet_type = threshold_match.group(1)
            threshold_part = threshold_match.group(
                2) if threshold_match.group(2) else ""

            # If user hasn't typed a colon yet, suggest both options
            if ":" not in tag_lower:
                if diet_type == "both":
                    results.append({
                        "id": f"#both",
                        "text": f"#both - Keto & Vegan diet filter (100% compliant)"
                    })
                    results.append({
                        "id": f"#both:",
                        "text": f"#both: - Type a threshold (e.g., 0.8 for 80%)"
                    })
                else:
                    results.append({
                        "id": f"#{diet_type}",
                        "text": f"#{diet_type} - {diet_type.title()} diet filter (100% compliant)"
                    })
                    # Suggest adding threshold
                    results.append({
                        "id": f"#{diet_type}:",
                        "text": f"#{diet_type}: - Type a threshold (e.g., 0.8 for 80%)"
                    })
            else:
                # User is typing a threshold
                if threshold_part:
                    # Validate the threshold
                    try:
                        threshold_value = float(threshold_part)
                        if 0 <= threshold_value <= 1:
                            # Valid threshold - create the suggestion
                            percentage = int(threshold_value * 100)
                            diet_display = "keto & vegan" if diet_type == "both" else diet_type
                            results.append({
                                "id": f"#{diet_type}:{threshold_part}",
                                "text": f"#{diet_type}:{threshold_part} - {percentage}% {diet_display} compliant"
                            })
                        else:
                            # Out of range - suggest correction
                            results.append({
                                "id": f"#{diet_type}:0.8",
                                "text": "Threshold must be between 0 and 1 (e.g., 0.8 for 80%)"
                            })
                    except ValueError:
                        # Invalid number - suggest examples
                        pass

                # Always show some example thresholds when typing after colon
                example_thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
                for thresh in example_thresholds:
                    if not threshold_part or str(thresh).startswith(threshold_part):
                        diet_display = "keto & vegan" if diet_type == "both" else diet_type
                        results.append({
                            "id": f"#{diet_type}:{thresh}",
                            "text": f"#{diet_type}:{thresh} - {int(thresh*100)}% {diet_display} compliant"
                        })
        else:
            # No match - suggest available diet types
            for diet in ["keto", "vegan", "both"]:
                if diet.startswith(tag_lower) or not tag_lower:
                    diet_display = "Keto & Vegan" if diet == "both" else diet.title()
                    results.append({
                        "id": f"#{diet}",
                        "text": f"#{diet} - {diet_display} diet filter"
                    })

        # Limit results
        return jsonify({"results": results[:10]})

    # Regular ingredient search
    # Use split_query_flags to separate ingredients from any tags
    free_text, flag_list = split_query_flags(query)

    if not free_text:
        # If only tags were entered, return empty results
        return jsonify({"results": []})

    # Search ingredients based on free_text only
    query_lower = free_text.lower()

    for idx, ingredient in enumerate(INGREDIENTS):
        if query_lower in ingredient.lower():
            results.append({
                "id": str(idx),
                "text": ingredient
            })

    # Sort by relevance (exact matches first, then by length)
    results.sort(key=lambda x: (
        not x["text"].lower().startswith(
            query_lower),  # Prioritize prefix matches
        len(x["text"])  # Then shorter ingredients
    ))

    return jsonify({"results": results[:50]})  # Limit to 50 results

# -------------------------------------------------------------------
# /search  — AND-combines any number of #keto[#:] / #vegan[#:] / #both[#:] flags
# Enhanced with substitution info
# -------------------------------------------------------------------


@app.route("/search")
def search():
    raw_q = request.args.get("q", "").strip()
    if not raw_q:
        return jsonify({"error": "Missing query"}), 400

    # 1) split out diet flags   (#keto, #vegan:0.7, #both …)
    free_text, flag_list = split_query_flags(raw_q)

    # 1-b) URL overrides → replace flag_list completely
    diet_url = request.args.get("diet", "").lower()
    if diet_url in ("keto", "vegan", "both"):
        thr_url = float(request.args.get("threshold", "1"))
        flag_list = [(diet_url, thr_url)]

    # 2) numeric Select2 IDs → tokens  (IDs stay the same)
    ids = [int(tok) for tok in free_text.split() if tok.isdigit()]
    tokens = [INGREDIENTS[i] for i in ids if i < len(INGREDIENTS)]
    keywords = " ".join(tokens) if tokens else free_text

    # 3) build OpenSearch bool query
    must_clause = [{"match": {"ingredients": {
        "query": keywords, "fuzziness": "AUTO"}}}] if keywords else [{"match_all": {}}]

    filt_clause = []
    for diet, thr in flag_list:
        if diet == "both":
            # For "both", we need recipes that meet thresholds for BOTH diets
            if thr >= 1.0:
                # Strict: must be both keto AND vegan
                filt_clause.append({"term": {"keto": True}})
                filt_clause.append({"term": {"vegan": True}})
            else:
                # Score threshold for both
                filt_clause.append({
                    "range": {"keto_score": {"gte": int(thr * 100)}}
                })
                filt_clause.append({
                    "range": {"vegan_score": {"gte": int(thr * 100)}}
                })
        else:
            # Single diet filtering (unchanged)
            if thr >= 1.0:                           # strict boolean
                filt_clause.append({"term": {diet: True}})
            else:                                    # score threshold
                filt_clause.append({
                    "range": {f"{diet}_score": {"gte": int(thr * 100)}}
                })

    body = {"query": {"bool": {"must": must_clause, "filter": filt_clause}}}

    # 4) run search
    try:
        # Prefer alias 'recipes' (backwards compatible). If it 404s, fall back.
        try:
            resp = client.search(index="recipes", body=body, size=12)
        except Exception:
            resp = client.search(index=RECIPES_INDEX, body=body, size=12)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # 5) enrich & return with substitution info
    hits = resp["hits"]["hits"]
    results = []

    # Get diet from flags for substitution analysis
    diet_for_analysis = None
    if flag_list:
        diet_for_analysis = flag_list[0][0]  # Use first diet flag

    for h in hits:
        src = h["_source"]
        ings = src.get("ingredients", [])

        result = {
            "title": src["title"],
            "description": src.get("description", ""),
            "ingredients": ings,
            "instructions": src.get("instructions", ""),
            "photo_url": src.get("photo_url", ""),
            "keto": src.get("keto",  is_keto(ings)),
            "keto_score": src.get("keto_score",  diet_score(ings, "keto")),
            "vegan": src.get("vegan", is_vegan(ings)),
            "vegan_score": src.get("vegan_score", diet_score(ings, "vegan")),
            "score": h["_score"],
        }

        # Add substitution info if searching for specific diet
        if diet_for_analysis:
            # For "both", check if recipe needs modifications for either diet
            if diet_for_analysis == "both":
                needs_modification = result["keto_score"] < 100 or result["vegan_score"] < 100
            else:
                needs_modification = not result[diet_for_analysis]

            if needs_modification:
                analysis = suggest_recipe_substitutions(
                    ings, diet_for_analysis)
                result["substitution_count"] = analysis["non_compliant_count"]
                result["can_be_modified"] = analysis["non_compliant_count"] > 0
                result["easily_adaptable"] = analysis["easily_adaptable"]

        results.append(result)

    return jsonify({"total": resp["hits"]["total"]["value"], "results": results})


@app.route("/substitutions")
def substitutions():
    ings = request.args.getlist("ingredient")
    diet = request.args.get("diet", "").lower()
    if not ings:
        return jsonify({"error": "Provide at least one ingredient"}), 400
    if diet not in ("keto", "vegan", "both"):
        return jsonify({"error": 'Diet must be "keto", "vegan", or "both"'}), 400

    data = suggest_recipe_substitutions(ings, diet)
    data["formatted_text"] = format_substitution_text(data)
    return jsonify(data)


@app.route("/modify-recipe", methods=["POST"])
def modify_recipe():
    """
    Endpoint to modify a recipe to be diet-compliant.

    Expected JSON body:
    {
        "ingredients": ["2 cups flour", "3 eggs", ...],
        "diet": "keto", "vegan", or "both",
        "auto_substitute": true/false (optional, default true),
        "preserve_quantities": true/false (optional, default true)
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    ingredients = data.get("ingredients", [])
    diet = data.get("diet", "").lower()
    auto_substitute = data.get("auto_substitute", True)
    preserve_quantities = data.get("preserve_quantities", True)

    if not ingredients:
        return jsonify({"error": "Provide at least one ingredient"}), 400
    if diet not in ("keto", "vegan", "both"):
        return jsonify({"error": 'Diet must be "keto", "vegan", or "both"'}), 400

    modified, substitutions = make_recipe_compliant(
        ingredients, diet, auto_substitute, preserve_quantities
    )

    # Calculate new compliance
    new_analysis = suggest_recipe_substitutions(modified, diet)

    response = {
        "original_ingredients": ingredients,
        "modified_ingredients": modified,
        "substitutions": substitutions,
        "is_now_compliant": new_analysis["is_compliant"],
        "new_compliance_percentage": new_analysis["compliance_percentage"],
        "changes_made": len(substitutions)
    }

    # Add individual diet percentages for "both"
    if diet == "both":
        response["keto_percentage"] = new_analysis.get("keto_percentage", 0)
        response["vegan_percentage"] = new_analysis.get("vegan_percentage", 0)

    return jsonify(response)


@app.route("/convert-recipes", methods=["POST"])
def convert_recipes():
    """
    Convert multiple recipes to be diet-compliant.

    Expected JSON body:
    {
        "recipes": [
            {"title": "...", "ingredients": [...]},
            ...
        ],
        "diet": "keto", "vegan", or "both"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    recipes = data.get("recipes", [])
    diet = data.get("diet", "").lower()

    if not recipes:
        return jsonify({"error": "Provide at least one recipe"}), 400
    if diet not in ("keto", "vegan", "both"):
        return jsonify({"error": 'Diet must be "keto", "vegan", or "both"'}), 400

    converted = convert_recipe_batch(recipes, diet)

    # Summary statistics
    total_recipes = len(converted)
    fully_compliant = sum(1 for r in converted if r["is_now_compliant"])
    total_changes = sum(r["changes_made"] for r in converted)

    return jsonify({
        "converted_recipes": converted,
        "summary": {
            "total_recipes": total_recipes,
            "fully_compliant": fully_compliant,
            "partially_compliant": total_recipes - fully_compliant,
            "total_substitutions": total_changes,
            "average_substitutions": round(total_changes / total_recipes, 1) if total_recipes > 0 else 0
        }
    })


@app.route("/check-compliance")
def check_compliance():
    """
    Quick compliance check for a recipe.
    Query params: diet, ingredient (multiple)
    Returns: is_compliant (bool), percentage
    """
    ings = request.args.getlist("ingredient")
    diet = request.args.get("diet", "").lower()

    if not ings:
        return jsonify({"error": "Provide at least one ingredient"}), 400
    if diet not in ("keto", "vegan", "both"):
        return jsonify({"error": 'Diet must be "keto", "vegan", or "both"'}), 400

    result = quick_compliance_check(ings, diet)
    return jsonify(result)


@app.route("/export-modified")
def export_modified():
    """
    Export recipe with substitutions applied.
    Query params: format (json/text/markdown), diet, title, 
                 ingredient (multiple), instructions (optional)
    """
    format_type = request.args.get("format", "json").lower()
    diet = request.args.get("diet", "").lower()
    title = request.args.get("title", "Recipe")
    ingredients = request.args.getlist("ingredient")
    instructions = request.args.get("instructions", "")

    if not ingredients:
        return jsonify({"error": "Provide at least one ingredient"}), 400
    if diet not in ("keto", "vegan", "both"):
        return jsonify({"error": 'Diet must be "keto", "vegan", or "both"'}), 400
    if format_type not in ("json", "text", "markdown"):
        return jsonify({"error": 'Format must be "json", "text", or "markdown"'}), 400

    recipe = {
        "title": title,
        "ingredients": ingredients,
        "instructions": instructions
    }

    exported = export_modified_recipe(recipe, diet, format_type)

    # Set appropriate content type
    content_types = {
        "json": "application/json",
        "text": "text/plain",
        "markdown": "text/markdown"
    }

    response = app.response_class(
        response=exported,
        status=200,
        mimetype=content_types[format_type]
    )

    # Add download headers
    diet_suffix = "keto_vegan" if diet == "both" else diet
    filename = f"{title.lower().replace(' ', '_')}_{diet_suffix}.{format_type}"
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"

    return response


@app.route("/recipe-metrics")
def recipe_metrics():
    """Get substitution metrics for recipes in the database."""
    # This would be used to add substitution data to OpenSearch index
    # Implementation depends on your indexing strategy

    sample_recipes = [
        {"title": "Chocolate Cake", "ingredients": [
            "flour", "sugar", "eggs", "butter", "cocoa"]},
        {"title": "Veggie Stir Fry", "ingredients": [
            "tofu", "vegetables", "soy sauce", "oil"]},
    ]

    metrics = []
    for recipe in sample_recipes:
        keto_analysis = suggest_recipe_substitutions(
            recipe["ingredients"], "keto")
        vegan_analysis = suggest_recipe_substitutions(
            recipe["ingredients"], "vegan")
        both_analysis = suggest_recipe_substitutions(
            recipe["ingredients"], "both")

        metrics.append({
            "title": recipe["title"],
            "keto_substitutions_needed": keto_analysis["non_compliant_count"],
            "vegan_substitutions_needed": vegan_analysis["non_compliant_count"],
            "both_substitutions_needed": both_analysis["non_compliant_count"],
            "easily_adaptable_keto": keto_analysis["easily_adaptable"],
            "easily_adaptable_vegan": vegan_analysis["easily_adaptable"],
            "easily_adaptable_both": both_analysis["easily_adaptable"]
        })

    return jsonify({"metrics": metrics})


@app.route("/save-modified-recipe", methods=["POST"])
def save_modified_recipe():
    """
    Save a modified recipe to the database.

    Expected JSON body:
    {
        "recipe_id": "original_recipe_id",
        "title": "Recipe Title",
        "original_ingredients": ["ingredient1", "ingredient2"],
        "modified_ingredients": ["modified1", "modified2"],
        "diet_modification": "keto|vegan|both",
        "substitutions_applied": {"original": "substitution"},
        "changes_made": 3,
        "original_keto_score": 50.0,
        "original_vegan_score": 80.0,
        "new_keto_score": 100.0,
        "new_vegan_score": 100.0,
        "description": "Recipe description",
        "instructions": ["step1", "step2"],
        "photo_url": "https://example.com/photo.jpg"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        conn = sqlite3.connect('modified_recipes.db')
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO modified_recipes (
                recipe_id, title, original_ingredients, modified_ingredients,
                diet_modification, substitutions_applied, changes_made,
                original_keto_score, original_vegan_score, new_keto_score, new_vegan_score,
                description, instructions, photo_url, unique_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('recipe_id', ''),
            data.get('title', ''),
            json.dumps(data.get('original_ingredients', [])),
            json.dumps(data.get('modified_ingredients', [])),
            data.get('diet_modification', ''),
            json.dumps(data.get('substitutions_applied', {})),
            data.get('changes_made', 0),
            data.get('original_keto_score', 0.0),
            data.get('original_vegan_score', 0.0),
            data.get('new_keto_score', 0.0),
            data.get('new_vegan_score', 0.0),
            data.get('description', ''),
            json.dumps(data.get('instructions', [])),
            data.get('photo_url', ''),
            data.get('unique_id', '')
        ))

        recipe_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "id": recipe_id,
            "message": "Recipe variation saved successfully"
        })

    except Exception as e:
        log.error(f"Error saving modified recipe: {e}")
        return jsonify({"error": "Failed to save recipe variation"}), 500


@app.route("/get-modified-recipes")
def get_modified_recipes():
    """
    Retrieve modified recipes from the database.
    Optional query parameters:
    - id: get specific recipe by ID
    - diet: filter by diet modification (keto, vegan, both)
    - title: filter by recipe title
    - unique_id: filter by unique identifier
    - limit: limit number of results (default 50)
    """
    try:
        recipe_id = request.args.get('id')
        diet_filter = request.args.get('diet')
        title_filter = request.args.get('title')
        unique_id_filter = request.args.get('unique_id')
        limit = int(request.args.get('limit', 50))

        conn = sqlite3.connect('modified_recipes.db')
        cursor = conn.cursor()

        query = '''
            SELECT * FROM modified_recipes
        '''
        params = []

        # Build WHERE clause
        conditions = []
        if recipe_id:
            conditions.append('id = ?')
            params.append(int(recipe_id))

        if diet_filter:
            conditions.append('diet_modification = ?')
            params.append(diet_filter)

        if title_filter:
            conditions.append('title = ?')
            params.append(title_filter)

        if unique_id_filter:
            conditions.append('unique_id = ?')
            params.append(unique_id_filter)

        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)

        # Add ORDER BY and LIMIT (only if not fetching by ID)
        if not recipe_id:
            query += ' ORDER BY modification_date DESC LIMIT ?'
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Get column names
        columns = [description[0] for description in cursor.description]

        recipes = []
        for row in rows:
            recipe = dict(zip(columns, row))

            # Parse JSON fields
            recipe['original_ingredients'] = json.loads(
                recipe['original_ingredients'])
            recipe['modified_ingredients'] = json.loads(
                recipe['modified_ingredients'])
            recipe['substitutions_applied'] = json.loads(
                recipe['substitutions_applied'])
            recipe['instructions'] = json.loads(recipe['instructions'])

            recipes.append(recipe)

        conn.close()

        return jsonify({
            "success": True,
            "recipes": recipes,
            "count": len(recipes)
        })

    except Exception as e:
        log.error(f"Error retrieving modified recipes: {e}")
        return jsonify({"error": "Failed to retrieve recipe variations"}), 500


@app.route("/delete-modified-recipe/<int:recipe_id>", methods=["DELETE"])
def delete_modified_recipe(recipe_id):
    """
    Delete a modified recipe from the database.
    """
    try:
        conn = sqlite3.connect('modified_recipes.db')
        cursor = conn.cursor()

        cursor.execute(
            'DELETE FROM modified_recipes WHERE id = ?', (recipe_id,))

        if cursor.rowcount == 0:
            conn.close()
            return jsonify({"error": "Recipe not found"}), 404

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "Recipe variation deleted successfully"
        })

    except Exception as e:
        log.error(f"Error deleting modified recipe: {e}")
        return jsonify({"error": "Failed to delete recipe variation"}), 500


@app.route("/clear-all-modified-recipes", methods=["DELETE"])
def clear_all_modified_recipes():
    """
    Delete all modified recipes from the database.
    """
    try:
        conn = sqlite3.connect('modified_recipes.db')
        cursor = conn.cursor()

        cursor.execute('DELETE FROM modified_recipes')
        deleted_count = cursor.rowcount

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": f"All {deleted_count} recipe variations deleted successfully"
        })

    except Exception as e:
        log.error(f"Error clearing all modified recipes: {e}")
        return jsonify({"error": "Failed to clear all recipe variations"}), 500


# ── dev runner ----------------------------------------------------
if __name__ == "__main__":
    log.info("CFG: %s %s %s %s %s", OPENSEARCH_URL, OLLAMA_URL,
             OLLAMA_MODEL, RECIPES_INDEX, EMBED_MODEL)
    app.run(host="0.0.0.0", port=8080, debug=True)
