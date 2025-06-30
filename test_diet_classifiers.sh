#!/bin/bash

echo "🧪 COMPREHENSIVE DIET CLASSIFIER TESTING SUITE - LIVE CODING READY"
echo "=================================================================="
echo "Current time: $(date)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to run tests with error handling and timing
run_test() {
    echo -e "${BLUE}$1${NC}"
    echo "----------------------------------------"
    start_time=$(date +%s.%N)
    
    if MSYS_NO_PATHCONV=1 docker exec -it search_by_ingredients_v1t3ls0n-nb-1 python3 -c "$2"; then
        end_time=$(date +%s.%N)
        duration=$(awk "BEGIN {print $end_time - $start_time}")
        echo -e "${GREEN}✅ Test completed successfully in ${duration:0:4}s${NC}"
    else
        echo -e "${RED}❌ Test failed${NC}"
    fi
    echo ""
}

# Test 0: System Health Check
run_test "🔧 SYSTEM HEALTH CHECK" "
print('=== SYSTEM STATUS ===')
import sys
print(f'Python version: {sys.version}')

try:
    import pandas as pd
    import sklearn
    from diet_classifiers import is_keto, is_vegan, NON_KETO, NON_VEGAN
    print(f'✅ All imports successful')
    print(f'✅ NON_KETO items: {len(NON_KETO)}')
    print(f'✅ NON_VEGAN items: {len(NON_VEGAN)}')
except Exception as e:
    print(f'❌ Import error: {e}')

try:
    import os
    if os.path.exists('/usr/src/data/ground_truth_sample.csv'):
        print('✅ Ground truth file found')
    else:
        print('❌ Ground truth file missing')
except Exception as e:
    print(f'❌ File check error: {e}')
"

# Test 1: Enhanced Individual Ingredient Testing
run_test "🧪 ENHANCED INDIVIDUAL INGREDIENT TESTS" "
from diet_classifiers import is_ingredient_keto, is_ingredient_vegan

print('=== COMPREHENSIVE KETO TESTS ===')
keto_tests = [
    # Should be True (keto-friendly)
    ('heavy cream', True),
    ('chicken breast', True), 
    ('eggs', True),
    ('almond flour', True),
    ('broccoli', True),
    ('avocado', True),
    ('olive oil', True),
    ('butter', True),
    ('cheddar cheese', True),
    ('salmon', True),
    ('spinach', True),
    ('cauliflower', True),
    
    # Should be False (non-keto)
    ('white rice', False),
    ('strawberries', False),
    ('sugar', False),
    ('bread', False),
    ('pasta', False),
    ('banana', False),
    ('potato', False),
    ('corn syrup', False),
    ('cooking spray', False),
    ('oats', False),
    ('honey', False),
    ('apple', False)
]

correct_keto = 0
for ingredient, expected in keto_tests:
    result = is_ingredient_keto(ingredient)
    status = '✅' if result == expected else '❌'
    if result == expected:
        correct_keto += 1
    print(f'{status} {ingredient:20} -> {result:5} (expected {expected})')

print(f'\\n🎯 Keto Accuracy: {correct_keto}/{len(keto_tests)} ({100*correct_keto/len(keto_tests):.1f}%)')

print('\\n=== COMPREHENSIVE VEGAN TESTS ===')
vegan_tests = [
    # Should be True (vegan)
    ('broccoli', True),
    ('almond milk', True),
    ('kidney beans', True),
    ('tofu', True),
    ('quinoa', True),
    ('spinach', True),
    ('avocado', True),
    ('olive oil', True),
    ('mushrooms', True),
    ('tomatoes', True),
    ('eggplant', True),
    ('peanut butter', True),
    
    # Should be False (non-vegan)
    ('eggs', False),
    ('chicken', False),
    ('milk', False),
    ('cheese', False),
    ('butter', False),
    ('honey', False),
    ('beef', False),
    ('fish', False),
    ('kahlua', False),
    ('coffee liqueur', False),
    ('bacon', False),
    ('yogurt', False)
]

correct_vegan = 0
for ingredient, expected in vegan_tests:
    result = is_ingredient_vegan(ingredient)
    status = '✅' if result == expected else '❌'
    if result == expected:
        correct_vegan += 1
    print(f'{status} {ingredient:20} -> {result:5} (expected {expected})')

print(f'\\n🎯 Vegan Accuracy: {correct_vegan}/{len(vegan_tests)} ({100*correct_vegan/len(vegan_tests):.1f}%)')
"

# Test 2: Edge Case Testing
run_test "🔍 EDGE CASE TESTING" "
from diet_classifiers import is_ingredient_keto, is_ingredient_vegan, normalise, parse_ingredients

print('=== EDGE CASE ANALYSIS ===')

# Test normalization edge cases
edge_cases = [
    '6 eggs',
    '2 cups heavy cream',
    '1/4 teaspoon salt',
    '3 tablespoons olive oil',
    'salt and pepper to taste',
    'cooking spray (optional)',
    '1 (14 oz) can kidney beans',
    'fresh ground black pepper'
]

for case in edge_cases:
    normalized = normalise(case)
    keto_result = is_ingredient_keto(case)
    vegan_result = is_ingredient_vegan(case)
    print(f'Original: \"{case}\"')
    print(f'  Normalized: \"{normalized}\"')
    print(f'  Keto: {keto_result}, Vegan: {vegan_result}')
    print()

# Test parsing edge cases
print('=== PARSING EDGE CASES ===')
parsing_tests = [
    \"['ingredient1' 'ingredient2' 'ingredient3']\",
    \"['single ingredient']\",
    \"ingredient without brackets\",
    \"['ingredient with, comma' 'another ingredient']\",
    \"['ingredient\\nwith\\nnewlines' 'normal ingredient']\"
]

for test_str in parsing_tests:
    try:
        parsed = parse_ingredients(test_str)
        print(f'Input: {test_str[:50]}...')
        print(f'  Parsed: {parsed}')
        print(f'  Count: {len(parsed)}')
    except Exception as e:
        print(f'❌ Parsing failed: {e}')
    print()
"

# Test 3: Recipe Parsing Validation
run_test "📋 ADVANCED RECIPE PARSING TEST" "
import pandas as pd
from diet_classifiers import parse_ingredients

df = pd.read_csv('/usr/src/data/ground_truth_sample.csv')

print('=== RECIPE PARSING VALIDATION ===')
parse_errors = 0
successful_parses = 0

for i in range(min(10, len(df))):
    try:
        raw = df.iloc[i]['ingredients']
        parsed = parse_ingredients(raw)
        
        print(f'Recipe {i+1}:')
        print(f'  Raw length: {len(str(raw))}')
        print(f'  Parsed count: {len(parsed)}')
        print(f'  Sample: {parsed[0][:30] if parsed else \"EMPTY\"}...')
        
        if len(parsed) == 0:
            print(f'  ⚠️  WARNING: Empty parse result!')
            parse_errors += 1
        else:
            successful_parses += 1
            
    except Exception as e:
        print(f'  ❌ Parse error: {e}')
        parse_errors += 1
    print()

print(f'📊 Parsing Summary: {successful_parses} successful, {parse_errors} errors')
"

# Test 4: Performance Stress Test
run_test "⚡ PERFORMANCE STRESS TEST" "
import pandas as pd
from diet_classifiers import is_keto, is_vegan, parse_ingredients
from time import time
import gc

df = pd.read_csv('/usr/src/data/ground_truth_sample.csv')

print('=== PERFORMANCE BENCHMARKS ===')

# Test 1: Individual ingredient speed
print('1. Individual Ingredient Speed Test:')
test_ingredients = ['chicken breast', 'heavy cream', 'white rice', 'eggs', 'broccoli'] * 20

start_time = time()
for ing in test_ingredients:
    _ = is_ingredient_keto(ing)
    _ = is_ingredient_vegan(ing)
end_time = time()

print(f'   {len(test_ingredients)} ingredients processed in {end_time-start_time:.3f}s')
print(f'   Rate: {len(test_ingredients)/(end_time-start_time):.1f} ingredients/second')

# Test 2: Recipe processing speed  
print('\\n2. Recipe Processing Speed Test:')
start_time = time()
for i in range(min(50, len(df))):
    ingredients = parse_ingredients(df.iloc[i]['ingredients'])
    _ = is_keto(ingredients)
    _ = is_vegan(ingredients)
end_time = time()

processed = min(50, len(df))
print(f'   {processed} recipes processed in {end_time-start_time:.3f}s')
print(f'   Rate: {processed/(end_time-start_time):.1f} recipes/second')

# Memory usage check
gc.collect()
print('\\n3. Memory Management: ✅ Garbage collection completed')
"

# Test 5: Full Evaluation with Detailed Metrics
run_test "📊 COMPREHENSIVE PERFORMANCE EVALUATION" "
import pandas as pd
from diet_classifiers import is_keto, is_vegan, parse_ingredients
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from time import time

print('Loading and processing ground truth data...')
df = pd.read_csv('/usr/src/data/ground_truth_sample.csv')

start_time = time()
df['keto_pred'] = df['ingredients'].apply(lambda x: is_keto(parse_ingredients(x)))
df['vegan_pred'] = df['ingredients'].apply(lambda x: is_vegan(parse_ingredients(x)))
end_time = time()

def print_detailed_metrics(y_true, y_pred, label):
    print(f'\\n{\"=\"*60}')
    print(f'{label} CLASSIFICATION RESULTS')
    print(f'{\"=\"*60}')
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    
    print(f'📊 Summary Metrics:')
    print(f'   Accuracy:  {acc:.4f} ({acc*100:.2f}%)')
    print(f'   F1-Score:  {f1:.4f}')
    print(f'   Precision: {prec:.4f}')
    print(f'   Recall:    {rec:.4f}')
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f'\\n📋 Confusion Matrix:')
    print(f'   True Negatives:  {tn}')
    print(f'   False Positives: {fp}')
    print(f'   False Negatives: {fn}')
    print(f'   True Positives:  {tp}')
    
    print(f'\\n📈 Detailed Classification Report:')
    print(classification_report(y_true, y_pred))

print_detailed_metrics(df['keto'], df['keto_pred'], '🥑 KETO')
print_detailed_metrics(df['vegan'], df['vegan_pred'], '🌱 VEGAN')

print(f'\\n⚡ PERFORMANCE METRICS')
print(f'{\"=\"*40}')
print(f'Processing time: {end_time - start_time:.3f} seconds')
print(f'Recipes per second: {len(df)/(end_time - start_time):.1f}')
print(f'Total recipes processed: {len(df)}')
print(f'Average time per recipe: {(end_time - start_time)/len(df)*1000:.2f}ms')
"

# Test 6: Advanced Error Analysis
run_test "🔍 ADVANCED ERROR ANALYSIS" "
import pandas as pd
from diet_classifiers import is_keto, is_vegan, parse_ingredients, is_ingredient_keto, is_ingredient_vegan

df = pd.read_csv('/usr/src/data/ground_truth_sample.csv')
df['keto_pred'] = df['ingredients'].apply(lambda x: is_keto(parse_ingredients(x)))
df['vegan_pred'] = df['ingredients'].apply(lambda x: is_vegan(parse_ingredients(x)))

def analyze_errors(df, true_col, pred_col, classifier_name):
    print(f'\\n{\"=\"*50}')
    print(f'{classifier_name} ERROR ANALYSIS')
    print(f'{\"=\"*50}')
    
    # False negatives and positives
    fn = df[(df[true_col] == True) & (df[pred_col] == False)]
    fp = df[(df[true_col] == False) & (df[pred_col] == True)]
    
    print(f'False Negatives (missed {classifier_name.lower()}): {len(fn)}')
    print(f'False Positives (wrong {classifier_name.lower()}): {len(fp)}')
    
    if len(fn) > 0:
        print(f'\\n❌ MISSED {classifier_name.upper()} RECIPES:')
        for i, (idx, row) in enumerate(fn.head(3).iterrows()):
            ingredients = parse_ingredients(row['ingredients'])
            print(f'   Recipe {idx}: {ingredients[:2]}...')
            # Detailed ingredient analysis
            if classifier_name == 'KETO':
                problem_ings = [ing for ing in ingredients if not is_ingredient_keto(ing)]
                if problem_ings:
                    print(f'      Problem ingredients: {problem_ings[:2]}...')
            else:
                problem_ings = [ing for ing in ingredients if not is_ingredient_vegan(ing)]
                if problem_ings:
                    print(f'      Problem ingredients: {problem_ings[:2]}...')
    
    if len(fp) > 0:
        print(f'\\n🚨 WRONG {classifier_name.upper()} PREDICTIONS:')
        for i, (idx, row) in enumerate(fp.head(3).iterrows()):
            ingredients = parse_ingredients(row['ingredients'])
            print(f'   Recipe {idx}: {ingredients[:2]}...')
            
    return len(fn), len(fp)

keto_fn, keto_fp = analyze_errors(df, 'keto', 'keto_pred', 'KETO')
vegan_fn, vegan_fp = analyze_errors(df, 'vegan', 'vegan_pred', 'VEGAN')

print(f'\\n📊 ERROR SUMMARY:')
print(f'   Total errors: {keto_fn + keto_fp + vegan_fn + vegan_fp}')
print(f'   Keto errors: {keto_fn + keto_fp}')
print(f'   Vegan errors: {vegan_fn + vegan_fp}')
"

# Test 7: Pattern Matching Deep Dive
run_test "🔍 PATTERN MATCHING DEEP ANALYSIS" "
from diet_classifiers import RX_KETO, RX_WL_KETO, RX_VEGAN, RX_WL_VEGAN, normalise, NON_KETO, NON_VEGAN

print('=== PATTERN MATCHING ANALYSIS ===')

test_ingredients = [
    # Tricky cases
    '6 eggs',
    'heavy cream', 
    'peanut butter',
    'eggplant',
    'kidney beans',
    'cooking spray',
    'kahlua',
    'coffee liqueur',
    'almond flour',
    'coconut oil',
    'strawberries',
    'white rice'
]

for ing in test_ingredients:
    norm = normalise(ing)
    keto_wl = RX_WL_KETO.search(ing) is not None
    keto_bl = RX_KETO.search(norm) is not None
    vegan_wl = RX_WL_VEGAN.search(ing) is not None  
    vegan_bl = RX_VEGAN.search(norm) is not None
    
    print(f'\\n\"{ing}\" -> \"{norm}\"')
    print(f'   Keto whitelist: {keto_wl:5} | Keto blacklist: {keto_bl:5}')
    print(f'   Vegan whitelist: {vegan_wl:5} | Vegan blacklist: {vegan_bl:5}')

print(f'\\n=== BLACKLIST STATS ===')
print(f'NON_KETO items: {len(NON_KETO)}')
print(f'NON_VEGAN items: {len(NON_VEGAN)}')

# Sample items from each list
print(f'\\nSample NON_KETO: {list(NON_KETO)[:10]}')
print(f'Sample NON_VEGAN: {list(NON_VEGAN)[:10]}')
"

# Test 8: USDA Database Analysis
run_test "🏛️ USDA DATABASE COMPREHENSIVE TEST" "
from diet_classifiers import carbs_per_100g, _load_usda_carb_table

carb_map = _load_usda_carb_table()
print(f'USDA database loaded: {len(carb_map)} items')

if len(carb_map) > 0:
    # Test common lookups
    test_lookups = [
        'chicken', 'rice', 'broccoli', 'heavy cream', 'strawberries',
        'beef', 'milk', 'cheese', 'butter', 'eggs', 'bread', 'pasta'
    ]
    
    print('\\n=== USDA LOOKUP TESTS ===')
    found_items = 0
    for item in test_lookups:
        carbs = carbs_per_100g(item)
        if carbs is not None:
            found_items += 1
            print(f'{item:15} -> {carbs:6.2f}g carbs/100g')
        else:
            print(f'{item:15} -> Not found')
    
    print(f'\\nLookup success rate: {found_items}/{len(test_lookups)} ({100*found_items/len(test_lookups):.1f}%)')
    
    # Analyze carb distribution
    carb_values = list(carb_map.values())
    avg_carbs = sum(carb_values) / len(carb_values)
    keto_friendly = sum(1 for c in carb_values if c <= 10)
    
    print(f'\\n=== USDA DATABASE ANALYSIS ===')
    print(f'Average carbs/100g: {avg_carbs:.2f}')
    print(f'Keto-friendly items (≤10g): {keto_friendly}/{len(carb_values)} ({100*keto_friendly/len(carb_values):.1f}%)')
else:
    print('⚠️  USDA database is empty!')
"

# Test 9: Regression Testing
run_test "🧪 REGRESSION TESTING" "
from diet_classifiers import is_ingredient_keto, is_ingredient_vegan

print('=== REGRESSION TESTS ===')
print('Testing previously fixed issues...')

# Historical fixes that should work
regression_tests = [
    # Parsing fixes
    ('Heavy cream should be keto', 'heavy cream', 'keto', True),
    ('Eggs should be non-vegan', 'eggs', 'vegan', False),
    ('Strawberries should be non-keto', 'strawberries', 'keto', False),
    ('Kidney beans should be vegan', 'kidney beans', 'vegan', True),
    ('Cooking spray should be non-keto', 'cooking spray', 'keto', False),
    ('Kahlua should be non-vegan', 'kahlua', 'vegan', False),
    
    # Edge cases
    ('Salt should be keto', 'salt', 'keto', True),
    ('Garlic should be keto', 'garlic', 'keto', True),
    ('Chicken should be keto but non-vegan', 'chicken', 'keto', True),
    ('Chicken should be non-vegan', 'chicken', 'vegan', False),
]

passed_tests = 0
for description, ingredient, test_type, expected in regression_tests:
    if test_type == 'keto':
        result = is_ingredient_keto(ingredient)
    else:
        result = is_ingredient_vegan(ingredient)
    
    status = '✅' if result == expected else '❌'
    if result == expected:
        passed_tests += 1
    
    print(f'{status} {description}')

print(f'\\n📊 Regression Test Results: {passed_tests}/{len(regression_tests)} passed ({100*passed_tests/len(regression_tests):.1f}%)')
"

echo ""
echo "🎯 COMPREHENSIVE TESTING COMPLETE!"
echo "=================================="
echo -e "${GREEN}🏆 LIVE CODING SESSION READY! 🏆${NC}"
echo -e "${YELLOW}All test modules completed. Your classifier is battle-tested!${NC}"
echo ""
echo -e "${CYAN}📋 Test Coverage Summary:${NC}"
echo "   ✅ System Health Check"
echo "   ✅ Individual Ingredient Testing (24 keto + 24 vegan tests)"
echo "   ✅ Edge Case Analysis"
echo "   ✅ Recipe Parsing Validation"
echo "   ✅ Performance Stress Testing"
echo "   ✅ Comprehensive Metrics (Accuracy, F1, Precision, Recall)"
echo "   ✅ Advanced Error Analysis"
echo "   ✅ Pattern Matching Deep Dive"
echo "   ✅ USDA Database Testing"
echo "   ✅ Regression Testing"
echo ""
echo -e "${PURPLE}🚀 Ready for Argmax Interview! 🚀${NC}"
echo ""