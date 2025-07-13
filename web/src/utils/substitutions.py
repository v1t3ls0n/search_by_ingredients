# utils/substitutions.py
from typing import List, Dict, Tuple
import re
# Import the existing logic from other modules
from utils.text_processing import normalise
from utils.classifiers import is_ingredient_keto, is_ingredient_vegan
from utils.constants import KETO_SUBSTITUTIONS, VEGAN_SUBSTITUTIONS


def find_non_compliant_ingredients(
    ingredients: List[str], 
    diet: str
) -> List[Tuple[str, List[str]]]:
    """
    Find ingredients that don't comply with the diet and suggest substitutions.
    
    Args:
        ingredients: List of ingredient strings
        diet: Either "keto", "vegan", or "both"
        
    Returns:
        List of tuples (non_compliant_ingredient, [substitution_suggestions])
    """
    if diet == "both":
        # For "both", we need ingredients that fail either keto OR vegan
        keto_non_compliant = find_non_compliant_ingredients(ingredients, "keto")
        vegan_non_compliant = find_non_compliant_ingredients(ingredients, "vegan")
        
        # Merge the results
        merged = {}
        
        # Add all keto non-compliant
        for ing, subs in keto_non_compliant:
            merged[ing] = {"keto": subs, "vegan": []}
        
        # Add/update with vegan non-compliant
        for ing, subs in vegan_non_compliant:
            if ing in merged:
                merged[ing]["vegan"] = subs
            else:
                merged[ing] = {"keto": [], "vegan": subs}
        
        # Find substitutions that work for both diets
        result = []
        for ing, diet_subs in merged.items():
            combined_subs = _find_combined_substitutions(
                ing, diet_subs["keto"], diet_subs["vegan"]
            )
            result.append((ing, combined_subs))
        
        return result
    
    # Single diet logic (unchanged)
    substitutions = []
    check_function = is_ingredient_keto if diet == "keto" else is_ingredient_vegan
    substitution_dict = KETO_SUBSTITUTIONS if diet == "keto" else VEGAN_SUBSTITUTIONS
    
    for ingredient in ingredients:
        if not check_function(ingredient):
            # Extract the core ingredient (remove quantities, etc.)
            core_ingredient = _extract_core_ingredient(ingredient)
            
            # Find substitutions
            suggestions = _find_substitutions(core_ingredient, substitution_dict)
            
            if suggestions:
                substitutions.append((ingredient, suggestions))
            else:
                # No direct substitution found
                substitutions.append((ingredient, []))
    
    return substitutions


def _find_combined_substitutions(
    ingredient: str,
    keto_subs: List[str],
    vegan_subs: List[str]
) -> List[str]:
    """
    Find substitutions that work for both keto AND vegan diets.
    
    Args:
        ingredient: The original ingredient
        keto_subs: List of keto-friendly substitutions
        vegan_subs: List of vegan-friendly substitutions
        
    Returns:
        List of substitutions that satisfy both diets
    """
    combined = []
    seen = set()
    
    # First, check if the ingredient needs substitution for both diets
    needs_keto_sub = not is_ingredient_keto(ingredient)
    needs_vegan_sub = not is_ingredient_vegan(ingredient)
    
    if not needs_keto_sub and not needs_vegan_sub:
        # Already compliant with both diets
        return []
    elif not needs_keto_sub:
        # Only needs vegan substitution, but must maintain keto compliance
        for vegan_sub in vegan_subs:
            if is_ingredient_keto(vegan_sub) and vegan_sub not in seen:
                combined.append(vegan_sub)
                seen.add(vegan_sub)
    elif not needs_vegan_sub:
        # Only needs keto substitution, but must maintain vegan compliance
        for keto_sub in keto_subs:
            if is_ingredient_vegan(keto_sub) and keto_sub not in seen:
                combined.append(keto_sub)
                seen.add(keto_sub)
    else:
        # Needs substitution for both diets
        # Check each keto sub to see if it's also vegan
        for keto_sub in keto_subs:
            if is_ingredient_vegan(keto_sub) and keto_sub not in seen:
                combined.append(keto_sub)
                seen.add(keto_sub)
        
        # Check each vegan sub to see if it's also keto
        for vegan_sub in vegan_subs:
            if is_ingredient_keto(vegan_sub) and vegan_sub not in seen:
                combined.append(vegan_sub)
                seen.add(vegan_sub)
    
    # If no perfect matches, use intelligent alternatives
    if not combined:
        combined = _get_both_diet_alternatives(ingredient)
    
    return combined[:3]  # Return top 3


def _get_both_diet_alternatives(ingredient: str) -> List[str]:
    """Get alternatives that work for both keto and vegan diets by finding intersection."""
    core = _extract_core_ingredient(ingredient)
    
    # Get substitutions from both diets
    keto_subs = _find_substitutions(core, KETO_SUBSTITUTIONS)
    vegan_subs = _find_substitutions(core, VEGAN_SUBSTITUTIONS)
    
    # Find alternatives that satisfy both diets
    both_alternatives = []
    
    # Check keto substitutions to see if they're also vegan
    for keto_sub in keto_subs:
        if is_ingredient_vegan(keto_sub):
            both_alternatives.append(keto_sub)
    
    # Check vegan substitutions to see if they're also keto
    for vegan_sub in vegan_subs:
        if is_ingredient_keto(vegan_sub) and vegan_sub not in both_alternatives:
            both_alternatives.append(vegan_sub)
    
    # If no direct matches found, try to find intelligent combinations
    if not both_alternatives:
        both_alternatives = _find_intelligent_combinations(core, keto_subs, vegan_subs)
    
    # If still no alternatives, provide helpful message
    if not both_alternatives:
        both_alternatives = [f"Remove {core} or find keto-vegan alternative"]
    
    return both_alternatives[:3]  # Return top 3


def _find_intelligent_combinations(ingredient: str, keto_subs: List[str], vegan_subs: List[str]) -> List[str]:
    """Find intelligent combinations when no direct matches exist."""
    alternatives = []
    
    # Special cases that need intelligent handling
    special_combinations = {
        # For dairy ingredients, prefer plant-based keto options
        ("milk", "dairy", "cream", "butter"): {
            "preferred": ["coconut cream", "coconut oil", "unsweetened almond milk", "unsweetened coconut milk"],
            "modifier": "unsweetened"
        },
        # For sweeteners, check for sugar-free versions of vegan options
        ("sugar", "honey", "syrup", "sweetener"): {
            "preferred": ["stevia", "erythritol", "monk fruit sweetener"],
            "modifier": "sugar-free"
        },
        # For eggs, suggest limited portions of vegan alternatives
        ("egg", "eggs"): {
            "preferred": ["flax egg", "chia egg", "aquafaba"],
            "modifier": "use sparingly for keto"
        },
        # For meat, prefer low-carb plant options
        ("meat", "chicken", "beef", "pork", "bacon"): {
            "preferred": ["mushrooms", "tofu", "tempeh", "nuts"],
            "modifier": "in moderation"
        }
    }
    
    # Check if ingredient falls into special categories
    for keywords, handling in special_combinations.items():
        if any(keyword in ingredient.lower() for keyword in keywords):
            # Check if any preferred alternatives are in either substitution list
            for preferred in handling["preferred"]:
                if any(preferred in sub for sub in keto_subs + vegan_subs):
                    # Check if this alternative works for both diets
                    if is_ingredient_keto(preferred) and is_ingredient_vegan(preferred):
                        modifier = handling.get("modifier", "")
                        if modifier and modifier not in preferred:
                            alternatives.append(f"{preferred} ({modifier})")
                        else:
                            alternatives.append(preferred)
    
    # Try to find modified versions of existing substitutions
    if not alternatives:
        # Look for vegan substitutions that could be made keto
        for vegan_sub in vegan_subs:
            if "unsweetened" not in vegan_sub and any(word in vegan_sub for word in ["milk", "yogurt"]):
                unsweetened_version = f"unsweetened {vegan_sub}"
                if is_ingredient_keto(unsweetened_version):
                    alternatives.append(unsweetened_version)
        
        # Look for keto substitutions that could be made vegan
        for keto_sub in keto_subs:
            # Check if there's a plant-based version
            if any(word in keto_sub for word in ["butter", "cream"]):
                if "coconut" in keto_sub or "almond" in keto_sub:
                    alternatives.append(keto_sub)
    
    return alternatives


def _extract_core_ingredient(ingredient: str) -> str:
    """Extract the core ingredient name from a full ingredient string."""
    # Use the normalise function from text_processing.py
    normalized = normalise(ingredient)
    
    # Additional cleanup specific to ingredient extraction
    # Remove common descriptors
    descriptors = ['fresh', 'dried', 'frozen', 'canned', 'chopped', 'diced', 
                   'sliced', 'minced', 'grated', 'shredded', 'whole', 'ground',
                   'organic', 'raw', 'cooked', 'peeled', 'crushed', 'melted']
    
    words = normalized.split()
    cleaned_words = [w for w in words if w not in descriptors]
    
    # Handle compound ingredients
    result = ' '.join(cleaned_words) if cleaned_words else normalized
    
    # Handle plural forms - simple singularization
    if result.endswith('ies'):
        result = result[:-3] + 'y'  # berries -> berry
    elif result.endswith('es'):
        result = result[:-2]  # tomatoes -> tomato
    elif result.endswith('s') and not result.endswith('ss'):
        result = result[:-1]  # eggs -> egg
    
    return result


def _find_substitutions(
    ingredient: str, 
    substitution_dict: Dict[str, List[str]]
) -> List[str]:
    """Find substitutions for an ingredient with enhanced matching."""
    # Direct match
    if ingredient in substitution_dict:
        return substitution_dict[ingredient]
    
    # Try variations for compound ingredients
    variations = [
        ingredient,
        ingredient.replace('-', ' '),  # all-purpose -> all purpose
        ingredient.replace(' ', '-'),  # all purpose -> all-purpose
    ]
    
    # Check each variation
    for variant in variations:
        if variant in substitution_dict:
            return substitution_dict[variant]
    
    # Partial match (ingredient contains key or key contains ingredient)
    suggestions = []
    scored_suggestions = []  # (score, suggestion) tuples
    
    for key, values in substitution_dict.items():
        # Exact substring match gets higher score
        if key in ingredient:
            for v in values:
                scored_suggestions.append((2, v))
        elif ingredient in key:
            for v in values:
                scored_suggestions.append((1, v))
    
    # Sort by score and remove duplicates
    scored_suggestions.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    unique_suggestions = []
    
    for score, suggestion in scored_suggestions:
        if suggestion not in seen:
            seen.add(suggestion)
            unique_suggestions.append(suggestion)
    
    return unique_suggestions[:3]  # Return top 3 suggestions


def suggest_recipe_substitutions(
    ingredients: List[str], 
    diet: str
) -> Dict[str, any]:
    """
    Analyze a recipe and suggest substitutions to make it diet-compliant.
    
    Args:
        ingredients: List of ingredient strings
        diet: Either "keto", "vegan", or "both"
        
    Returns:
        Dictionary with analysis results
    """
    if diet == "both":
        # Check compliance for both diets
        keto_compliant = all(is_ingredient_keto(ing) for ing in ingredients)
        vegan_compliant = all(is_ingredient_vegan(ing) for ing in ingredients)
        is_compliant = keto_compliant and vegan_compliant
        
        # Calculate combined compliance
        keto_count = sum(1 for ing in ingredients if is_ingredient_keto(ing))
        vegan_count = sum(1 for ing in ingredients if is_ingredient_vegan(ing))
        both_count = sum(1 for ing in ingredients if is_ingredient_keto(ing) and is_ingredient_vegan(ing))
        
        total = len(ingredients) if ingredients else 1
        compliance_percentage = (both_count / total * 100) if total > 0 else 0
        
        non_compliant = find_non_compliant_ingredients(ingredients, diet)
        
        return {
            "diet": "keto & vegan",
            "is_compliant": is_compliant,
            "compliance_percentage": round(compliance_percentage, 1),
            "keto_percentage": round((keto_count / total * 100), 1),
            "vegan_percentage": round((vegan_count / total * 100), 1),
            "total_ingredients": total,
            "non_compliant_count": len(non_compliant),
            "easily_adaptable": len(non_compliant) > 0 and len(non_compliant) <= 3,
            "substitutions": [
                {
                    "ingredient": ing,
                    "suggestions": subs if subs else ["Remove or find keto-vegan alternative"]
                }
                for ing, subs in non_compliant
            ]
        }
    
    # Single diet logic (unchanged)
    non_compliant = find_non_compliant_ingredients(ingredients, diet)
    
    # Check if recipe is already compliant
    is_compliant = len(non_compliant) == 0
    
    # Calculate compliance percentage
    total_ingredients = len(ingredients)
    compliant_count = total_ingredients - len(non_compliant)
    compliance_percentage = (compliant_count / total_ingredients * 100) if total_ingredients > 0 else 0
    
    # Check if easily adaptable (< 3 substitutions needed)
    easily_adaptable = len(non_compliant) > 0 and len(non_compliant) <= 3
    
    return {
        "diet": diet,
        "is_compliant": is_compliant,
        "compliance_percentage": round(compliance_percentage, 1),
        "total_ingredients": total_ingredients,
        "non_compliant_count": len(non_compliant),
        "easily_adaptable": easily_adaptable,
        "substitutions": [
            {
                "ingredient": ing,
                "suggestions": subs if subs else [f"Remove {ing} or find {diet} alternative"]
            }
            for ing, subs in non_compliant
        ]
    }


def format_substitution_text(substitution_analysis: Dict[str, any]) -> str:
    """Format substitution analysis as readable text."""
    diet_name = substitution_analysis["diet"]
    
    if substitution_analysis["is_compliant"]:
        return f"✅ This recipe is already {diet_name}-compliant!"
    
    lines = [
        f"📊 {diet_name.title()} Compliance: {substitution_analysis['compliance_percentage']}%",
        f"   ({substitution_analysis['total_ingredients'] - substitution_analysis['non_compliant_count']}/{substitution_analysis['total_ingredients']} compliant ingredients)",
    ]
    
    # Add individual diet percentages for "both"
    if diet_name == "keto & vegan":
        lines.extend([
            f"   Keto: {substitution_analysis['keto_percentage']}% | Vegan: {substitution_analysis['vegan_percentage']}%"
        ])
    
    lines.extend(["", "🔄 Suggested Substitutions:"])
    
    for sub in substitution_analysis["substitutions"]:
        lines.append(f"\n❌ {sub['ingredient']}")
        if sub['suggestions'] and not sub['suggestions'][0].startswith("Remove"):
            for i, suggestion in enumerate(sub['suggestions'], 1):
                lines.append(f"   {i}. {suggestion}")
        else:
            lines.append(f"   • {sub['suggestions'][0]}")
    
    return "\n".join(lines)


def make_recipe_compliant(
    ingredients: List[str], 
    diet: str,
    auto_substitute: bool = True,
    preserve_quantities: bool = True
) -> Tuple[List[str], Dict[str, str]]:
    """
    Modify a recipe to make it diet-compliant by substituting ingredients.
    
    Args:
        ingredients: List of ingredient strings
        diet: Either "keto", "vegan", or "both"
        auto_substitute: If True, automatically apply first suggestion; 
                        If False, remove non-compliant ingredients
        preserve_quantities: If True, preserve original quantities in substitutions
    
    Returns:
        Tuple of (modified_ingredients, substitution_map)
    """
    # Get the analysis
    analysis = suggest_recipe_substitutions(ingredients, diet)
    
    # If already compliant, return as-is
    if analysis["is_compliant"]:
        return ingredients.copy(), {}
    
    modified_ingredients = ingredients.copy()
    substitution_map = {}
    
    # Process each non-compliant ingredient
    for sub in analysis["substitutions"]:
        original = sub["ingredient"]
        suggestions = sub["suggestions"]
        
        # Find the ingredient in the list
        try:
            idx = modified_ingredients.index(original)
            
            if auto_substitute and suggestions and \
               not suggestions[0].startswith("Remove"):
                # Use the first suggestion
                replacement = suggestions[0]
                
                # Preserve quantities if requested
                if preserve_quantities:
                    # Extract quantity from original
                    quantity_match = re.match(r'^([\d\s/\-]+(?:cups?|tbsp|tsp|oz|lb|g|ml|l)?)\s*(.+)', original)
                    if quantity_match:
                        quantity = quantity_match.group(1)
                        replacement = f"{quantity} {replacement}"
                
                modified_ingredients[idx] = replacement
                substitution_map[original] = replacement
            else:
                # Remove the ingredient
                modified_ingredients[idx] = None
                substitution_map[original] = "REMOVED"
        except ValueError:
            # Ingredient not found in list (shouldn't happen)
            pass
    
    # Filter out None values (removed ingredients)
    modified_ingredients = [ing for ing in modified_ingredients if ing is not None]
    
    return modified_ingredients, substitution_map


def convert_recipe_batch(
    recipes: List[Dict[str, any]], 
    diet: str
) -> List[Dict[str, any]]:
    """
    Convert multiple recipes to be diet-compliant.
    
    Args:
        recipes: List of recipe dictionaries with 'title' and 'ingredients'
        diet: Either "keto", "vegan", or "both"
    
    Returns:
        List of modified recipes with conversion info
    """
    converted_recipes = []
    
    for recipe in recipes:
        original_ingredients = recipe.get("ingredients", [])
        modified_ingredients, substitution_map = make_recipe_compliant(
            original_ingredients, diet
        )
        
        # Calculate new compliance
        new_analysis = suggest_recipe_substitutions(modified_ingredients, diet)
        
        converted_recipes.append({
            "title": recipe.get("title", "Untitled"),
            "original_ingredients": original_ingredients,
            "modified_ingredients": modified_ingredients,
            "substitutions": substitution_map,
            "is_now_compliant": new_analysis["is_compliant"],
            "original_compliance": suggest_recipe_substitutions(original_ingredients, diet)["compliance_percentage"],
            "new_compliance": new_analysis["compliance_percentage"],
            "changes_made": len(substitution_map)
        })
    
    return converted_recipes


def quick_compliance_check(ingredients: List[str], diet: str) -> Dict[str, any]:
    """
    Quick compliance check without full analysis.
    
    Args:
        ingredients: List of ingredient strings
        diet: Either "keto", "vegan", or "both"
    
    Returns:
        Simplified compliance info
    """
    if diet == "both":
        keto_count = sum(1 for ing in ingredients if is_ingredient_keto(ing))
        vegan_count = sum(1 for ing in ingredients if is_ingredient_vegan(ing))
        both_count = sum(1 for ing in ingredients if is_ingredient_keto(ing) and is_ingredient_vegan(ing))
        total = len(ingredients)
        
        return {
            "diet": "keto & vegan",
            "is_compliant": both_count == total,
            "compliance_percentage": round((both_count / total * 100) if total > 0 else 0, 1),
            "compliant_count": both_count,
            "total_ingredients": total,
            "keto_percentage": round((keto_count / total * 100) if total > 0 else 0, 1),
            "vegan_percentage": round((vegan_count / total * 100) if total > 0 else 0, 1)
        }
    
    # Single diet logic
    check_function = is_ingredient_keto if diet == "keto" else is_ingredient_vegan
    
    compliant_count = sum(1 for ing in ingredients if check_function(ing))
    total = len(ingredients)
    
    return {
        "diet": diet,
        "is_compliant": compliant_count == total,
        "compliance_percentage": round((compliant_count / total * 100) if total > 0 else 0, 1),
        "compliant_count": compliant_count,
        "total_ingredients": total
    }


def export_modified_recipe(
    recipe: Dict[str, any],
    diet: str,
    format: str = "json"
) -> str:
    """
    Export recipe with substitutions in various formats.
    
    Args:
        recipe: Recipe dictionary with title and ingredients
        diet: Diet type ("keto", "vegan", or "both")
        format: Export format (json, text, markdown)
    
    Returns:
        Formatted recipe string
    """
    # Modify the recipe
    modified_ingredients, substitution_map = make_recipe_compliant(
        recipe.get("ingredients", []), diet
    )
    
    diet_display = "Keto & Vegan" if diet == "both" else diet.title()
    
    if format == "json":
        import json
        return json.dumps({
            "title": recipe.get("title", ""),
            "diet": diet,
            "ingredients": modified_ingredients,
            "substitutions": substitution_map,
            "instructions": recipe.get("instructions", "")
        }, indent=2)
    
    elif format == "text":
        lines = [
            f"{recipe.get('title', 'Recipe')} ({diet_display} Version)",
            "=" * 50,
            "",
            "Ingredients:",
        ]
        for ing in modified_ingredients:
            lines.append(f"- {ing}")
        
        if substitution_map:
            lines.extend(["", "Substitutions Made:"])
            for orig, sub in substitution_map.items():
                if sub != "REMOVED":
                    lines.append(f"- {orig} → {sub}")
                else:
                    lines.append(f"- {orig} (removed)")
        
        if recipe.get("instructions"):
            lines.extend(["", "Instructions:", recipe["instructions"]])
        
        return "\n".join(lines)
    
    elif format == "markdown":
        lines = [
            f"# {recipe.get('title', 'Recipe')} ({diet_display} Version)",
            "",
            "## Ingredients",
        ]
        for ing in modified_ingredients:
            lines.append(f"- {ing}")
        
        if substitution_map:
            lines.extend(["", "## Substitutions"])
            for orig, sub in substitution_map.items():
                if sub != "REMOVED":
                    lines.append(f"- **{orig}** → {sub}")
                else:
                    lines.append(f"- ~~{orig}~~ *(removed)*")
        
        if recipe.get("instructions"):
            lines.extend(["", "## Instructions", recipe["instructions"]])
        
        return "\n".join(lines)
    
    return ""