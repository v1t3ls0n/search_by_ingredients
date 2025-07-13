import re
def parse_ingredients(ingredients_str):
    """Parse string representation of ingredient list into actual list"""
    if isinstance(ingredients_str, str):
        try:
            # Remove the outer brackets and quotes
            content = ingredients_str.strip()[2:-2]  # Remove ["... and ..."]
            
            # Split on quote-space-quote pattern: ' '
            ingredients = re.split(r"'\s+'", content)
            
            # Clean up any remaining quotes
            ingredients = [ing.strip("'\"") for ing in ingredients if ing.strip()]
            
            return ingredients
        except:
            return [ingredients_str]
    return ingredients_str
