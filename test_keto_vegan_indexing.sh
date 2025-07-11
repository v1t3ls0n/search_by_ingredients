#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}🧪 KETO/VEGAN INDEXING VERIFICATION TEST${NC}"
echo "========================================"
echo "Testing that keto/vegan fields are properly indexed in OpenSearch"
echo ""

# Function to run Python code in web container
run_in_container() {
    docker exec -it search_by_ingredients-web-1 python3 -c "$1"
}

# Function to check OpenSearch via curl
check_opensearch() {
    curl -s -X GET "localhost:9200/$1" -H 'Content-Type: application/json' -d "$2" 2>/dev/null
}

echo -e "${BLUE}1. Checking if OpenSearch is ready...${NC}"
if check_opensearch "_cluster/health" | grep -q "green\|yellow"; then
    echo -e "${GREEN}✅ OpenSearch is healthy${NC}"
else
    echo -e "${RED}❌ OpenSearch is not ready${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}2. Checking index mappings...${NC}"
MAPPING_CHECK=$(run_in_container "
from opensearchpy import OpenSearch
import json

client = OpenSearch(
    hosts=['http://os:9200'],
    http_auth=None,
    use_ssl=False,
    verify_certs=False,
    ssl_show_warn=False,
)

try:
    mapping = client.indices.get_mapping(index='recipes')
    properties = mapping['recipes']['mappings']['properties']
    
    has_keto = 'keto' in properties
    has_vegan = 'vegan' in properties
    
    print(f'MAPPING_CHECK:')
    print(f'keto_field:{has_keto}')
    print(f'vegan_field:{has_vegan}')
    
    if has_keto:
        print(f'keto_type:{properties[\"keto\"][\"type\"]}')
    if has_vegan:
        print(f'vegan_type:{properties[\"vegan\"][\"type\"]}')
except Exception as e:
    print(f'ERROR:{e}')
")

if echo "$MAPPING_CHECK" | grep -q "keto_field:True"; then
    echo -e "${GREEN}✅ Keto field exists in mapping${NC}"
else
    echo -e "${RED}❌ Keto field missing from mapping${NC}"
fi

if echo "$MAPPING_CHECK" | grep -q "vegan_field:True"; then
    echo -e "${GREEN}✅ Vegan field exists in mapping${NC}"
else
    echo -e "${RED}❌ Vegan field missing from mapping${NC}"
fi

echo ""
echo -e "${BLUE}3. Sampling recipes to check field population...${NC}"
SAMPLE_CHECK=$(run_in_container "
from opensearchpy import OpenSearch
import json

client = OpenSearch(
    hosts=['http://os:9200'],
    http_auth=None,
    use_ssl=False,
    verify_certs=False,
    ssl_show_warn=False,
)

try:
    # Get total count
    total = client.count(index='recipes')['count']
    print(f'TOTAL_RECIPES:{total}')
    
    # Sample 50 recipes
    response = client.search(
        index='recipes',
        body={
            'size': 50,
            'query': {'match_all': {}},
            '_source': ['title', 'keto', 'vegan', 'ingredients']
        }
    )
    
    hits = response['hits']['hits']
    
    # Analyze
    keto_true = 0
    keto_false = 0
    keto_missing = 0
    vegan_true = 0
    vegan_false = 0
    vegan_missing = 0
    
    for hit in hits:
        source = hit['_source']
        
        if 'keto' not in source:
            keto_missing += 1
        elif source['keto']:
            keto_true += 1
        else:
            keto_false += 1
            
        if 'vegan' not in source:
            vegan_missing += 1
        elif source['vegan']:
            vegan_true += 1
        else:
            vegan_false += 1
    
    print(f'SAMPLE_SIZE:{len(hits)}')
    print(f'KETO_TRUE:{keto_true}')
    print(f'KETO_FALSE:{keto_false}')
    print(f'KETO_MISSING:{keto_missing}')
    print(f'VEGAN_TRUE:{vegan_true}')
    print(f'VEGAN_FALSE:{vegan_false}')
    print(f'VEGAN_MISSING:{vegan_missing}')
    
    # Show an example
    if hits:
        example = hits[0]['_source']
        print(f'EXAMPLE_TITLE:{example.get(\"title\", \"No title\")}')
        print(f'EXAMPLE_KETO:{example.get(\"keto\", \"MISSING\")}')
        print(f'EXAMPLE_VEGAN:{example.get(\"vegan\", \"MISSING\")}')
        
except Exception as e:
    print(f'ERROR:{e}')
")

# Parse results
TOTAL=$(echo "$SAMPLE_CHECK" | grep "TOTAL_RECIPES:" | cut -d: -f2)
SAMPLE_SIZE=$(echo "$SAMPLE_CHECK" | grep "SAMPLE_SIZE:" | cut -d: -f2)
KETO_TRUE=$(echo "$SAMPLE_CHECK" | grep "KETO_TRUE:" | cut -d: -f2)
KETO_FALSE=$(echo "$SAMPLE_CHECK" | grep "KETO_FALSE:" | cut -d: -f2)
KETO_MISSING=$(echo "$SAMPLE_CHECK" | grep "KETO_MISSING:" | cut -d: -f2)
VEGAN_TRUE=$(echo "$SAMPLE_CHECK" | grep "VEGAN_TRUE:" | cut -d: -f2)
VEGAN_FALSE=$(echo "$SAMPLE_CHECK" | grep "VEGAN_FALSE:" | cut -d: -f2)
VEGAN_MISSING=$(echo "$SAMPLE_CHECK" | grep "VEGAN_MISSING:" | cut -d: -f2)

echo -e "${CYAN}Total recipes in index: $TOTAL${NC}"
echo -e "${CYAN}Sample size: $SAMPLE_SIZE${NC}"
echo ""
echo "Keto field distribution:"
echo -e "  ${GREEN}True (keto-friendly): $KETO_TRUE${NC}"
echo -e "  ${YELLOW}False (not keto): $KETO_FALSE${NC}"
echo -e "  ${RED}Missing: $KETO_MISSING${NC}"
echo ""
echo "Vegan field distribution:"
echo -e "  ${GREEN}True (vegan): $VEGAN_TRUE${NC}"
echo -e "  ${YELLOW}False (not vegan): $VEGAN_FALSE${NC}"
echo -e "  ${RED}Missing: $VEGAN_MISSING${NC}"

# Show example
echo ""
echo -e "${BLUE}4. Example recipe:${NC}"
EXAMPLE_TITLE=$(echo "$SAMPLE_CHECK" | grep "EXAMPLE_TITLE:" | cut -d: -f2-)
EXAMPLE_KETO=$(echo "$SAMPLE_CHECK" | grep "EXAMPLE_KETO:" | cut -d: -f2)
EXAMPLE_VEGAN=$(echo "$SAMPLE_CHECK" | grep "EXAMPLE_VEGAN:" | cut -d: -f2)
echo "  Title: $EXAMPLE_TITLE"
echo "  Keto: $EXAMPLE_KETO"
echo "  Vegan: $EXAMPLE_VEGAN"

echo ""
echo -e "${BLUE}5. Testing filter queries...${NC}"

# Test keto filter
KETO_COUNT=$(curl -s -X GET "localhost:9200/recipes/_count" -H 'Content-Type: application/json' -d '{
  "query": {"term": {"keto": true}}
}' | grep -o '"count":[0-9]*' | grep -o '[0-9]*')

echo -e "Keto recipes (keto=true): ${GREEN}$KETO_COUNT${NC}"

# Test vegan filter
VEGAN_COUNT=$(curl -s -X GET "localhost:9200/recipes/_count" -H 'Content-Type: application/json' -d '{
  "query": {"term": {"vegan": true}}
}' | grep -o '"count":[0-9]*' | grep -o '[0-9]*')

echo -e "Vegan recipes (vegan=true): ${GREEN}$VEGAN_COUNT${NC}"

# Test combined filter
BOTH_COUNT=$(curl -s -X GET "localhost:9200/recipes/_count" -H 'Content-Type: application/json' -d '{
  "query": {
    "bool": {
      "must": [
        {"term": {"keto": true}},
        {"term": {"vegan": true}}
      ]
    }
  }
}' | grep -o '"count":[0-9]*' | grep -o '[0-9]*')

echo -e "Keto AND Vegan recipes: ${GREEN}$BOTH_COUNT${NC}"

echo ""
echo -e "${BLUE}6. Checking specific test cases...${NC}"

# Test specific recipes
TEST_CASES=$(run_in_container "
from opensearchpy import OpenSearch

client = OpenSearch(
    hosts=['http://os:9200'],
    http_auth=None,
    use_ssl=False,
    verify_certs=False,
    ssl_show_warn=False,
)

# Search for recipes with specific ingredients
test_queries = [
    ('heavy cream', 'should be keto, not vegan'),
    ('tofu', 'should be vegan, might be keto'),
    ('chicken', 'should be keto, not vegan'),
    ('broccoli', 'should be both keto and vegan')
]

for ingredient, expectation in test_queries:
    try:
        response = client.search(
            index='recipes',
            body={
                'size': 1,
                'query': {
                    'match': {
                        'ingredients': ingredient
                    }
                },
                '_source': ['title', 'keto', 'vegan', 'ingredients']
            }
        )
        
        if response['hits']['hits']:
            hit = response['hits']['hits'][0]['_source']
            print(f'TEST:{ingredient}')
            print(f'FOUND:{hit.get(\"title\", \"No title\")[:50]}')
            print(f'KETO:{hit.get(\"keto\", \"MISSING\")}')
            print(f'VEGAN:{hit.get(\"vegan\", \"MISSING\")}')
            print(f'EXPECTATION:{expectation}')
            print('---')
    except:
        pass
")

echo "$TEST_CASES" | while IFS= read -r line; do
    if [[ $line == TEST:* ]]; then
        ingredient=$(echo "$line" | cut -d: -f2-)
        echo -e "\n${CYAN}Testing recipes with '$ingredient':${NC}"
    elif [[ $line == FOUND:* ]]; then
        title=$(echo "$line" | cut -d: -f2-)
        echo "  Found: $title..."
    elif [[ $line == KETO:* ]]; then
        value=$(echo "$line" | cut -d: -f2)
        echo -n "  Keto: "
        if [[ $value == "True" ]]; then
            echo -e "${GREEN}$value${NC}"
        elif [[ $value == "False" ]]; then
            echo -e "${YELLOW}$value${NC}"
        else
            echo -e "${RED}$value${NC}"
        fi
    elif [[ $line == VEGAN:* ]]; then
        value=$(echo "$line" | cut -d: -f2)
        echo -n "  Vegan: "
        if [[ $value == "True" ]]; then
            echo -e "${GREEN}$value${NC}"
        elif [[ $value == "False" ]]; then
            echo -e "${YELLOW}$value${NC}"
        else
            echo -e "${RED}$value${NC}"
        fi
    elif [[ $line == EXPECTATION:* ]]; then
        expectation=$(echo "$line" | cut -d: -f2-)
        echo -e "  ${BLUE}Expected: $expectation${NC}"
    fi
done

echo ""
echo -e "${BLUE}7. Summary:${NC}"
echo "============================="

if [[ $KETO_MISSING -eq 0 ]] && [[ $VEGAN_MISSING -eq 0 ]]; then
    echo -e "${GREEN}✅ All sampled recipes have both keto and vegan fields!${NC}"
    echo -e "${GREEN}✅ Indexing appears to be working correctly.${NC}"
    
    # Calculate percentages
    if [[ $TOTAL -gt 0 ]]; then
        KETO_PERCENT=$(awk "BEGIN {printf \"%.1f\", 100*$KETO_COUNT/$TOTAL}")
        VEGAN_PERCENT=$(awk "BEGIN {printf \"%.1f\", 100*$VEGAN_COUNT/$TOTAL}")
        BOTH_PERCENT=$(awk "BEGIN {printf \"%.1f\", 100*$BOTH_COUNT/$TOTAL}")
        
        echo ""
        echo "Dataset statistics:"
        echo "  - $KETO_PERCENT% of recipes are keto-friendly"
        echo "  - $VEGAN_PERCENT% of recipes are vegan"
        echo "  - $BOTH_PERCENT% of recipes are both keto and vegan"
    fi
else
    echo -e "${RED}❌ Some recipes are missing diet classification fields!${NC}"
    echo -e "${RED}❌ Check your indexing logic in batch_index_recipes()${NC}"
    echo ""
    echo "Make sure you're using recipe_with_diet instead of recipe when appending to actions!"
fi

echo ""
echo -e "${CYAN}Test complete!${NC}"