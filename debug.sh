#!/bin/bash

echo "🔍 Debugging Keto Classification Issues..."
echo "========================================"

# Debug keto false negatives
python3 -c "
import pandas as pd
from diet_classifiers import is_keto, is_ingredient_keto, parse_ingredients

print('Loading ground truth data...')
df = pd.read_csv('/usr/src/data/ground_truth_sample.csv')

# Apply predictions
print('Running predictions...')
df['keto_pred'] = df['ingredients'].apply(lambda x: is_keto(parse_ingredients(x)))

# Find false negatives (missed keto recipes)
false_negatives = df[(df['keto'] == True) & (df['keto_pred'] == False)]
total_keto = sum(df['keto'])

print(f'\\n📊 KETO CLASSIFICATION ANALYSIS')
print(f'Total keto recipes in test set: {total_keto}')
print(f'Missed keto recipes (false negatives): {len(false_negatives)}')
print(f'Keto recall: {(total_keto - len(false_negatives))/total_keto:.2%}')

print(f'\\n🚫 ANALYZING MISSED KETO RECIPES:')
print('=' * 50)

for i, (idx, row) in enumerate(false_negatives.head(5).iterrows()):
    ingredients = parse_ingredients(row['ingredients'])
    print(f'\\n🍽️  Missed Recipe #{i+1} (Index {idx}):')
    print(f'Ingredients ({len(ingredients)} total):')
    
    # Check each ingredient
    problem_ingredients = []
    for j, ing in enumerate(ingredients[:8]):  # Check first 8 ingredients
        result = is_ingredient_keto(ing)
        status = '✅' if result else '❌'
        print(f'  {status} {ing[:45]:45} -> {result}')
        if not result:
            problem_ingredients.append(ing)
    
    if len(ingredients) > 8:
        print(f'  ... and {len(ingredients)-8} more ingredients')
    
    print(f'\\n  🎯 Non-keto ingredients found: {len(problem_ingredients)}')
    if problem_ingredients:
        print(f'  🔍 First problematic: {problem_ingredients[0][:50]}')

print(f'\\n🔧 RECOMMENDATIONS:')
print('=' * 30)
print('1. Check if blacklist is too aggressive')
print('2. Consider expanding whitelist patterns') 
print('3. Review USDA carb threshold (currently 10g/100g)')
print('4. Improve ingredient parsing for compound terms')
"

echo ""
echo "🎯 Quick ingredient tests:"
echo "========================"

# Test some common keto ingredients
python3 -c "
from diet_classifiers import is_ingredient_keto

test_ingredients = [
    'chicken thigh',
    'heavy cream', 
    'cheddar cheese',
    'avocado oil',
    'ground beef',
    'salmon fillet',
    'cauliflower',
    'spinach',
    'butter',
    'olive oil',
    'eggs',
    'bacon'
]

print('Testing common keto ingredients:')
for ing in test_ingredients:
    result = is_ingredient_keto(ing)
    status = '✅' if result else '❌'
    print(f'{status} {ing:15} -> {result}')
"

echo ""
echo "✅ Debug complete! Check the analysis above."