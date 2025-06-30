#!/bin/bash

echo "🧪 COMPREHENSIVE DIET CLASSIFIER TESTING SUITE"
echo "=============================================="
echo "Current time: $(date)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to run tests with error handling
run_test() {
    echo -e "${BLUE}$1${NC}"
    echo "----------------------------------------"
    if MSYS_NO_PATHCONV=1 docker exec -it search_by_ingredients_v1t3ls0n-nb-1 python3 -c "$2"; then
        echo -e "${GREEN}✅ Test completed successfully${NC}"
    else
        echo -e "${RED}❌ Test failed${NC}"
    fi
    echo ""
}

# Test 1: Individual Ingredient Testing
run_test "🧪 INDIVIDUAL INGREDIENT TESTS" "
from diet_classifiers import is_ingredient_keto, is_ingredient_vegan

print('=== KETO INGREDIENT TESTS ===')
keto_tests = [
    ('heavy cream', True),
    ('chicken breast', True), 
    ('eggs', True),
    ('almond flour', True),
    ('broccoli', True),
    ('white rice', False),
    ('strawberries', False),
    ('sugar', False)
]

for ingredient, expected in keto_tests:
    result = is_ingredient_keto(ingredient)
    status = '✅' if result == expected else '❌'
    print(f'{status} {ingredient:15} -> {result} (expected {expected})')

print('\\n=== VEGAN INGREDIENT TESTS ===')
vegan_tests = [
    ('broccoli', True),
    ('almond milk', True),
    ('kidney beans', True),
    ('tofu', True),
    ('eggs', False),
    ('chicken', False),
    ('milk', False),
    ('cheese', False)
]

for ingredient, expected in vegan_tests:
    result = is_ingredient_vegan(ingredient)
    status = '✅' if result == expected else '❌'
    print(f'{status} {ingredient:15} -> {result} (expected {expected})')
"

# Test 2: Recipe Parsing Test
run_test "📋 RECIPE PARSING TEST" "
import pandas as pd
from diet_classifiers import parse_ingredients

df = pd.read_csv('/usr/src/data/ground_truth_sample.csv')

print('Testing ingredient parsing:')
for i in range(3):
    print(f'\\nRecipe {i+1}:')
    raw = df.iloc[i]['ingredients']
    parsed = parse_ingredients(raw)
    print(f'Raw length: {len(raw)}')
    print(f'Parsed count: {len(parsed)}')
    print(f'First 3 ingredients:')
    for j, ing in enumerate(parsed[:3]):
        print(f'  {j+1}. \"{ing}\"')
"

# Test 3: Full Evaluation
run_test "📊 FULL PERFORMANCE EVALUATION" "
import pandas as pd
from diet_classifiers import is_keto, is_vegan, parse_ingredients
from sklearn.metrics import classification_report, accuracy_score, f1_score
from time import time

print('Loading ground truth data...')
df = pd.read_csv('/usr/src/data/ground_truth_sample.csv')

print('Running predictions...')
start_time = time()
df['keto_pred'] = df['ingredients'].apply(lambda x: is_keto(parse_ingredients(x)))
df['vegan_pred'] = df['ingredients'].apply(lambda x: is_vegan(parse_ingredients(x)))
end_time = time()

print('\\n' + '='*50)
print('🥑 KETO CLASSIFICATION RESULTS')
print('='*50)
keto_acc = accuracy_score(df['keto'], df['keto_pred'])
keto_f1 = f1_score(df['keto'], df['keto_pred'])
print(f'Accuracy: {keto_acc:.3f}')
print(f'F1-Score: {keto_f1:.3f}')
print('\\nDetailed Report:')
print(classification_report(df['keto'], df['keto_pred']))

print('\\n' + '='*50)
print('🌱 VEGAN CLASSIFICATION RESULTS')
print('='*50)
vegan_acc = accuracy_score(df['vegan'], df['vegan_pred'])
vegan_f1 = f1_score(df['vegan'], df['vegan_pred'])
print(f'Accuracy: {vegan_acc:.3f}')
print(f'F1-Score: {vegan_f1:.3f}')
print('\\nDetailed Report:')
print(classification_report(df['vegan'], df['vegan_pred']))

print(f'\\n⚡ PERFORMANCE METRICS')
print('='*30)
print(f'Processing time: {end_time - start_time:.2f} seconds')
print(f'Recipes per second: {len(df)/(end_time - start_time):.1f}')
print(f'Total recipes processed: {len(df)}')
"

# Test 4: Error Analysis
run_test "🔍 ERROR ANALYSIS" "
import pandas as pd
from diet_classifiers import is_keto, is_vegan, parse_ingredients

df = pd.read_csv('/usr/src/data/ground_truth_sample.csv')
df['keto_pred'] = df['ingredients'].apply(lambda x: is_keto(parse_ingredients(x)))
df['vegan_pred'] = df['ingredients'].apply(lambda x: is_vegan(parse_ingredients(x)))

# Keto errors
keto_fn = df[(df['keto'] == True) & (df['keto_pred'] == False)]
keto_fp = df[(df['keto'] == False) & (df['keto_pred'] == True)]

print('🥑 KETO ERROR ANALYSIS')
print('='*30)
print(f'False Negatives (missed keto): {len(keto_fn)}')
print(f'False Positives (wrong keto): {len(keto_fp)}')

if len(keto_fn) > 0:
    print('\\n❌ Missed Keto Recipes:')
    for i, (idx, row) in enumerate(keto_fn.head(3).iterrows()):
        ingredients = parse_ingredients(row['ingredients'])
        print(f'  {i+1}. {ingredients[:2]}...')

if len(keto_fp) > 0:
    print('\\n🚨 Wrong Keto Predictions:')
    for i, (idx, row) in enumerate(keto_fp.head(3).iterrows()):
        ingredients = parse_ingredients(row['ingredients'])
        print(f'  {i+1}. {ingredients[:2]}...')

# Vegan errors
vegan_fn = df[(df['vegan'] == True) & (df['vegan_pred'] == False)]
vegan_fp = df[(df['vegan'] == False) & (df['vegan_pred'] == True)]

print('\\n🌱 VEGAN ERROR ANALYSIS')
print('='*30)
print(f'False Negatives (missed vegan): {len(vegan_fn)}')
print(f'False Positives (wrong vegan): {len(vegan_fp)}')

if len(vegan_fn) > 0:
    print('\\n❌ Missed Vegan Recipes:')
    for i, (idx, row) in enumerate(vegan_fn.head(3).iterrows()):
        ingredients = parse_ingredients(row['ingredients'])
        print(f'  {i+1}. {ingredients[:2]}...')

if len(vegan_fp) > 0:
    print('\\n🚨 Wrong Vegan Predictions:')
    for i, (idx, row) in enumerate(vegan_fp.head(3).iterrows()):
        ingredients = parse_ingredients(row['ingredients'])
        print(f'  {i+1}. {ingredients[:2]}...')
"

# Test 5: Pattern Matching Debug
run_test "🔍 PATTERN MATCHING DEBUG" "
from diet_classifiers import RX_KETO, RX_WL_KETO, RX_VEGAN, RX_WL_VEGAN, normalise

test_ingredients = [
    '6 eggs',
    'heavy cream', 
    'almond flour',
    'white rice',
    'chicken breast',
    'strawberries'
]

print('PATTERN MATCHING ANALYSIS')
print('='*40)
for ing in test_ingredients:
    norm = normalise(ing)
    keto_wl = RX_WL_KETO.search(ing) is not None
    keto_bl = RX_KETO.search(norm) is not None
    vegan_wl = RX_WL_VEGAN.search(ing) is not None  
    vegan_bl = RX_VEGAN.search(norm) is not None
    
    print(f'\\n\"{ing}\" -> \"{norm}\"')
    print(f'  Keto whitelist: {keto_wl}')
    print(f'  Keto blacklist: {keto_bl}')
    print(f'  Vegan whitelist: {vegan_wl}')
    print(f'  Vegan blacklist: {vegan_bl}')
"

# Test 6: USDA Database Test
run_test "🏛️ USDA DATABASE TEST" "
from diet_classifiers import carbs_per_100g, _load_usda_carb_table

carb_map = _load_usda_carb_table()
print(f'USDA database loaded: {len(carb_map)} items')

test_lookups = ['chicken', 'rice', 'broccoli', 'heavy cream', 'strawberries']
print('\\nUSDA Lookup Tests:')
for item in test_lookups:
    carbs = carbs_per_100g(item)
    print(f'{item:15} -> {carbs}g carbs/100g' if carbs else f'{item:15} -> Not found')
"

echo ""
echo "🎯 TESTING COMPLETE!"
echo "====================="
echo -e "${GREEN}All tests finished. Review results above for any issues.${NC}"
echo -e "${YELLOW}For live debugging, run individual test sections as needed.${NC}"
echo ""
echo "📝 Quick Commands:"
echo "  Full test: ./test_diet_classifiers.sh"
echo "  Individual test: Copy specific python code from script"
echo "  Performance only: Just run the evaluation section"
echo ""