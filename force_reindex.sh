#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔄 FORCE RE-INDEXING WITH KETO/VEGAN FIELDS${NC}"
echo "============================================"
echo ""

echo -e "${BLUE}1. Stopping containers...${NC}"
docker-compose stop

echo -e "${BLUE}2. Removing old index data...${NC}"
# Remove the indexed marker
docker exec -it search_by_ingredients-web-1 rm -f /app/.indexed 2>/dev/null || true

# Delete the OpenSearch data volume to start fresh
docker-compose down -v

echo -e "${BLUE}3. Starting OpenSearch fresh...${NC}"
docker-compose up -d os

# Wait for OpenSearch to be ready
echo -e "${BLUE}4. Waiting for OpenSearch to be ready...${NC}"
for i in {1..30}; do
    if curl -s "http://localhost:9200/_cluster/health" | grep -q "green\|yellow"; then
        echo -e "${GREEN}✅ OpenSearch is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

echo -e "${BLUE}5. Starting web container (will trigger indexing)...${NC}"
docker-compose up -d web

echo -e "${BLUE}6. Monitoring indexing progress...${NC}"
echo "Following logs for 30 seconds to see indexing happen..."
echo "Look for lines showing keto/vegan classification..."
echo ""

# Follow logs for a bit to see indexing
timeout 30 docker logs -f search_by_ingredients-web-1 2>&1 | grep -E "(Indexing|keto|vegan|Recipe #|NLTK|Starting)" || true

echo ""
echo -e "${BLUE}7. Waiting for indexing to complete...${NC}"
# Give it some more time to finish
sleep 10

echo ""
echo -e "${BLUE}8. Running verification test...${NC}"
echo ""

# Run a quick check
docker exec -it search_by_ingredients-web-1 python3 -c "
from opensearchpy import OpenSearch
client = OpenSearch(hosts=['http://os:9200'])

# Check total count
total = client.count(index='recipes')['count']
print(f'Total recipes indexed: {total}')

# Check a few recipes
response = client.search(
    index='recipes', 
    body={
        'size': 5, 
        '_source': ['title', 'keto', 'vegan']
    }
)

print('\\nSample recipes with diet fields:')
print('-' * 60)
for hit in response['hits']['hits']:
    r = hit['_source']
    keto = r.get('keto', 'MISSING')
    vegan = r.get('vegan', 'MISSING')
    title = r.get('title', 'No title')[:40]
    print(f'{title:40} | Keto: {str(keto):7} | Vegan: {vegan}')

# Count keto and vegan recipes
keto_count = client.count(index='recipes', body={'query': {'term': {'keto': True}}})['count']
vegan_count = client.count(index='recipes', body={'query': {'term': {'vegan': True}}})['count']

print(f'\\nKeto recipes: {keto_count}')
print(f'Vegan recipes: {vegan_count}')
"

echo ""
echo -e "${GREEN}✅ Re-indexing complete!${NC}"
echo ""
echo "You can now run the full test again:"
echo "  ./test_keto_vegan_indexing.sh"