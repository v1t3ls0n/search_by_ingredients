#!/usr/bin/env python
"""
Flask API:
* /select2        – ingredient autocomplete, flag-aware
* /search         – ingredient search with #keto / #vegan / #both filters
* /substitutions  – diet-compliant ingredient swaps
* /modify-recipe  – modify recipe to be diet-compliant
* /convert-recipes – batch recipe conversion
* /check-compliance – quick compliance check
* /export-modified – export modified recipe
"""
from __future__ import annotations

import logging
import sys
from time import sleep
from typing import List
import json

from decouple import config
from flask import Flask, jsonify, render_template, request, session
from opensearchpy import OpenSearch

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

# ── logging -------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
for noisy in ("opensearchpy", "urllib3", "opensearch"):
    logging.getLogger(noisy).setLevel(logging.ERROR)
log = logging.getLogger(__name__)

# ── Flask & OpenSearch bootstrap ---------------------------------
app = Flask(__name__)
app.secret_key = config("SECRET_KEY", "dev-secret-key-change-in-production")


def _wait(client: OpenSearch, tries: int = 30, delay: int = 2) -> bool:
    for _ in range(tries):
        if client.ping():
            return True
        sleep(delay)
    return False


def _bootstrap():
    client = OpenSearch(
        hosts=[config("OPENSEARCH_URL", "http://localhost:9200")],
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False,
    )
    if not _wait(client):
        log.error("OpenSearch not reachable")
        sys.exit(1)

    resp = client.search(index="ingredients",
                         body={"query": {"match_all": {}}}, size=10_000)
    vocab: List[str] = [h["_source"]["ingredients"]
                        for h in resp["hits"]["hits"]]
    log.info("Loaded %s ingredient tokens", len(vocab))
    return client, vocab


client, INGREDIENTS = _bootstrap()

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
        resp = client.search(index="recipes", body=body, size=12)
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


# ── dev runner ----------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
