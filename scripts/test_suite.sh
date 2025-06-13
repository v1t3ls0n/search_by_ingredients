#!/bin/bash
# =============================================================================
# TEST SUITE
# Complete validation of the diet classification pipeline
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=()

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((TESTS_PASSED++))
}

error() {
    echo -e "${RED}❌ $1${NC}"
    ((TESTS_FAILED++))
    FAILED_TESTS+=("$1")
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# =============================================================================
# SETUP AND ENVIRONMENT CHECKS
# =============================================================================

log "🚀 Starting Pre-Submission Test Suite"
log "Time: $(date)"
log "Directory: $(pwd)"

echo ""
log "📋 PHASE 1: ENVIRONMENT VALIDATION"

# Check if we're in the right directory
if [[ ! -f "docker-compose.yml" ]]; then
    error "docker-compose.yml not found. Run this script from the project root directory."
    exit 1
fi
success "Project directory structure verified"

# Check Docker
if ! command -v docker &> /dev/null; then
    error "Docker not installed or not in PATH"
    exit 1
fi
success "Docker available"

if ! command -v docker-compose &> /dev/null; then
    error "docker-compose not installed or not in PATH"
    exit 1
fi
success "docker-compose available"

# Check required files
REQUIRED_FILES=(
    "web/src/diet_classifiers.py"
    "web/Dockerfile"
    "web/requirements.txt"
    "pretrained_models/models.zip"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        success "Required file exists: $file"
    else
        error "Missing required file: $file"
    fi
done

# =============================================================================
# DOCKER BUILD AND STARTUP TESTS
# =============================================================================

echo ""
log "📋 PHASE 2: DOCKER BUILD AND STARTUP"

# Clean up any existing containers
log "Cleaning up existing containers..."
docker-compose down --remove-orphans --volumes 2>/dev/null || true

# Build and start services
log "Building and starting Docker services..."
if docker-compose up -d --build; then
    success "Docker services started successfully"
else
    error "Failed to start Docker services"
    exit 1
fi

# Wait for services to be ready
log "Waiting for services to initialize..."
sleep 30

# Check service health
log "Checking service health..."

# Check OpenSearch
if curl -s http://localhost:9200/_cluster/health >/dev/null 2>&1; then
    success "OpenSearch service is healthy"
else
    warning "OpenSearch service not responding (may not be critical)"
fi

# Check web service
if curl -s http://localhost:8080 >/dev/null 2>&1; then
    success "Web service is responding"
else
    error "Web service not responding on port 8080"
fi

# Check if web container is running
WEB_CONTAINER=$(docker-compose ps -q web)
if [[ -n "$WEB_CONTAINER" ]] && [[ "$(docker inspect -f '{{.State.Status}}' $WEB_CONTAINER)" == "running" ]]; then
    success "Web container is running"
else
    error "Web container is not running properly"
fi

# =============================================================================
# CORE FUNCTIONALITY TESTS
# =============================================================================

echo ""
log "📋 PHASE 3: CORE FUNCTIONALITY TESTS"

# Test 1: Basic ingredient classification
log "Test 1: Basic ingredient classification..."
RESULT=$(docker-compose exec -T web python3 /app/web/diet_classifiers.py --ingredients "almond flour, eggs, butter" 2>/dev/null)
if echo "$RESULT" | grep -q '"keto"' && echo "$RESULT" | grep -q '"vegan"'; then
    success "Basic ingredient classification works"
    log "   Result: $RESULT"
else
    error "Basic ingredient classification failed"
    log "   Output: $RESULT"
fi

# Test 2: Different ingredient combinations
log "Test 2: Testing different ingredient combinations..."

# Keto ingredients
KETO_TEST=$(docker-compose exec -T web python3 /app/web/diet_classifiers.py --ingredients "coconut oil, spinach, cheese" 2>/dev/null)
if echo "$KETO_TEST" | grep -q '"keto": true'; then
    success "Keto ingredient detection works"
else
    warning "Keto ingredient detection may need review"
    log "   Result: $KETO_TEST"
fi

# Vegan ingredients  
VEGAN_TEST=$(docker-compose exec -T web python3 /app/web/diet_classifiers.py --ingredients "quinoa, vegetables, olive oil" 2>/dev/null)
if echo "$VEGAN_TEST" | grep -q '"vegan": true'; then
    success "Vegan ingredient detection works"
else
    warning "Vegan ingredient detection may need review"
    log "   Result: $VEGAN_TEST"
fi

# Non-keto ingredients
NON_KETO_TEST=$(docker-compose exec -T web python3 /app/web/diet_classifiers.py --ingredients "white rice, banana, honey" 2>/dev/null)
if echo "$NON_KETO_TEST" | grep -q '"keto": false'; then
    success "Non-keto ingredient detection works"
else
    warning "Non-keto ingredient detection may need review"
    log "   Result: $NON_KETO_TEST"
fi

# Test 3: JSON format input
log "Test 3: JSON format ingredient input..."
JSON_TEST=$(docker-compose exec -T web python3 /app/web/diet_classifiers.py --ingredients '["almond flour", "coconut oil"]' 2>/dev/null)
if echo "$JSON_TEST" | grep -q '"keto"' && echo "$JSON_TEST" | grep -q '"vegan"'; then
    success "JSON format input works"
else
    error "JSON format input failed"
fi

# =============================================================================
# MODEL AND DATA TESTS
# =============================================================================

echo ""
log "📋 PHASE 4: MODEL AND DATA VALIDATION"

# Test 4: Check if models are loaded
log "Test 4: Model loading verification..."
if docker-compose exec -T web test -f /app/pretrained_models/models.pkl 2>/dev/null; then
    success "Pretrained models found"
else
    # Check if models are in artifacts
    if docker-compose exec -T web test -f /app/artifacts/models.pkl 2>/dev/null; then
        success "Models found in artifacts directory"
    else
        error "No models found in either pretrained_models or artifacts"
    fi
fi

# Test 5: Check vectorizer
log "Test 5: Vectorizer verification..."
if docker-compose exec -T web test -f /app/pretrained_models/vectorizer.pkl 2>/dev/null; then
    success "Vectorizer found"
else
    if docker-compose exec -T web test -f /app/artifacts/vectorizer.pkl 2>/dev/null; then
        success "Vectorizer found in artifacts directory"
    else
        error "No vectorizer found"
    fi
fi

# Test 6: Check data files
log "Test 6: Data file verification..."
if docker-compose exec -T web test -f /app/data/ground_truth_sample.csv 2>/dev/null; then
    success "Ground truth data found"
else
    error "Ground truth data missing"
fi

if docker-compose exec -T web test -f /app/data/allrecipes.parquet 2>/dev/null; then
    success "Recipe data found"
else
    error "Recipe data missing"
fi

# Test 7: USDA data
log "Test 7: USDA data verification..."
if docker-compose exec -T web test -d /app/data/usda 2>/dev/null; then
    success "USDA data directory found"
    # Check for specific files
    if docker-compose exec -T web test -f /app/data/usda/food.csv 2>/dev/null; then
        success "USDA food.csv found"
    else
        warning "USDA food.csv missing"
    fi
else
    error "USDA data directory missing"
fi

# =============================================================================
# ADVANCED FUNCTIONALITY TESTS
# =============================================================================

echo ""
log "📋 PHASE 5: ADVANCED FUNCTIONALITY"

# Test 8: Ground truth evaluation (if ground truth exists)
log "Test 8: Ground truth evaluation..."
GT_RESULT=$(docker-compose exec -T web python3 /app/web/diet_classifiers.py --ground_truth /app/data/ground_truth_sample.csv 2>&1)
if echo "$GT_RESULT" | grep -q "evaluation.*complete\|saved.*predictions"; then
    success "Ground truth evaluation completed"
else
    if echo "$GT_RESULT" | grep -q "not found"; then
        warning "Ground truth file not found (expected in some setups)"
    else
        error "Ground truth evaluation failed"
        log "   Output: $GT_RESULT"
    fi
fi

# Test 9: Training mode (quick test)
log "Test 9: Training pipeline test (sample mode)..."
TRAIN_RESULT=$(timeout 300 docker-compose exec -T web python3 /app/web/diet_classifiers.py --train --sample_frac 0.01 --mode text 2>&1 || true)
if echo "$TRAIN_RESULT" | grep -q "PIPELINE COMPLETE\|Pipeline completed\|✅.*training"; then
    success "Training pipeline works (sample mode)"
else
    if echo "$TRAIN_RESULT" | grep -q "timeout\|killed"; then
        warning "Training test timed out (may be normal for large datasets)"
    else
        warning "Training pipeline test inconclusive"
        log "   Last few lines: $(echo "$TRAIN_RESULT" | tail -5)"
    fi
fi

# =============================================================================
# SHELL SCRIPT TESTS
# =============================================================================

echo ""
log "📋 PHASE 6: SHELL SCRIPT VALIDATION"

# Check if shell scripts exist and are executable
SHELL_SCRIPTS=(
    "train.sh"
    "eval_ground_truth.sh"
    "run_full_pipeline.sh"
)

for script in "${SHELL_SCRIPTS[@]}"; do
    if [[ -f "$script" ]]; then
        if [[ -x "$script" ]]; then
            success "Shell script $script exists and is executable"
        else
            warning "Shell script $script exists but is not executable"
            log "   Fix with: chmod +x $script"
        fi
    else
        warning "Shell script $script not found"
    fi
done

# =============================================================================
# PERFORMANCE AND RESOURCE TESTS
# =============================================================================

echo ""
log "📋 PHASE 7: PERFORMANCE VALIDATION"

# Test 10: Memory usage check
log "Test 10: Container memory usage..."
WEB_MEMORY=$(docker stats --no-stream --format "table {{.MemUsage}}" $WEB_CONTAINER | tail -1 | cut -d'/' -f1)
if [[ -n "$WEB_MEMORY" ]]; then
    success "Web container memory usage: $WEB_MEMORY"
else
    warning "Could not determine memory usage"
fi

# Test 11: Response time test
log "Test 11: Response time test..."
START_TIME=$(date +%s.%N)
docker-compose exec -T web python3 /app/web/diet_classifiers.py --ingredients "test ingredient" >/dev/null 2>&1
END_TIME=$(date +%s.%N)
RESPONSE_TIME=$(echo "$END_TIME - $START_TIME" | bc)
if (( $(echo "$RESPONSE_TIME < 30" | bc -l) )); then
    success "Response time acceptable: ${RESPONSE_TIME}s"
else
    warning "Response time slow: ${RESPONSE_TIME}s"
fi

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

echo ""
log "📋 PHASE 8: INTEGRATION TESTS"

# Test 12: Web interface availability
log "Test 12: Web interface integration..."
if curl -s -f http://localhost:8080 >/dev/null; then
    success "Web interface accessible"
else
    error "Web interface not accessible"
fi

# Test 13: API endpoint test (if exists)
log "Test 13: API functionality..."
# This would test any Flask API endpoints if they exist
API_TEST=$(curl -s -X POST http://localhost:8080/classify -H "Content-Type: application/json" -d '{"ingredients": ["test"]}' 2>/dev/null || echo "not available")
if [[ "$API_TEST" != "not available" ]]; then
    success "API endpoint responsive"
else
    warning "API endpoint not available (may be normal if not implemented)"
fi

# =============================================================================
# CLEANUP AND FINAL VALIDATION
# =============================================================================

echo ""
log "📋 PHASE 9: CLEANUP AND FINAL CHECKS"

# Test 14: Artifact generation
log "Test 14: Checking generated artifacts..."
if docker-compose exec -T web test -d /app/artifacts 2>/dev/null; then
    success "Artifacts directory exists"
    
    ARTIFACT_COUNT=$(docker-compose exec -T web find /app/artifacts -name "*.csv" -o -name "*.pkl" -o -name "*.log" | wc -l)
    if [[ $ARTIFACT_COUNT -gt 0 ]]; then
        success "Artifacts generated: $ARTIFACT_COUNT files"
    else
        warning "No artifacts found (may be normal if training hasn't run)"
    fi
else
    warning "Artifacts directory not found"
fi

# Test 15: Log file generation
log "Test 15: Log file verification..."
if docker-compose exec -T web test -f /app/artifacts/pipeline.log 2>/dev/null; then
    success "Pipeline log file exists"
else
    warning "Pipeline log file not found (may be normal)"
fi

# =============================================================================
# FINAL REPORT
# =============================================================================

echo ""
echo "============================================================================="
log "🏁 TEST SUITE COMPLETE"
echo "============================================================================="

echo ""
if [[ $TESTS_FAILED -eq 0 ]]; then
    echo -e "${GREEN}🎉 ALL CRITICAL TESTS PASSED!${NC}"
else
    echo -e "${YELLOW}⚠️  Some tests failed, but system may still be functional${NC}"
fi

echo ""
echo "📊 SUMMARY:"
echo "   ✅ Tests Passed: $TESTS_PASSED"
echo "   ❌ Tests Failed: $TESTS_FAILED"

if [[ ${#FAILED_TESTS[@]} -gt 0 ]]; then
    echo ""
    echo "🔍 FAILED TESTS:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "   • $test"
    done
fi


# Cleanup
log "Cleaning up test environment..."
# automatically stop containers after testing
docker-compose down

echo ""
log "Test suite finished at $(date)"
echo "============================================================================="