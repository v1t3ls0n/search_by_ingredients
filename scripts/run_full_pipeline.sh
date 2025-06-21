#!/usr/bin/env bash
# ============================================
# run_full_pipeline.sh - Enhanced with logging
# ============================================

set -euo pipefail

# Default values
SAMPLE_FRAC="0.1"
FORCE_FLAG=""

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --sample_frac <value>   Set sample fraction for training (default: 0.1)"
    echo "  --force                 Force recomputation of embeddings"
    echo "  --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                      # Use defaults (sample_frac=0.1)"
    echo "  $0 --sample_frac 0.5    # Use 50% of data"
    echo "  $0 --sample_frac 1.0    # Use full dataset"
    echo "  $0 --force              # Force recompute with default sample_frac"
    echo "  $0 --sample_frac 0.3 --force  # Use 30% of data and force recompute"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sample_frac)
            if [ -z "${2:-}" ]; then
                echo "❌ Error: --sample_frac requires a value"
                usage
            fi
            SAMPLE_FRAC="$2"
            # Validate sample_frac is a number between 0 and 1
            if ! [[ "$SAMPLE_FRAC" =~ ^[0-9]*\.?[0-9]+$ ]] || (( $(echo "$SAMPLE_FRAC > 1" | bc -l) )) || (( $(echo "$SAMPLE_FRAC <= 0" | bc -l) )); then
                echo "❌ Error: sample_frac must be a number between 0 and 1"
                exit 1
            fi
            shift 2
            ;;
        --force)
            FORCE_FLAG="--force"
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "❌ Unknown option: $1"
            usage
            ;;
    esac
done

# Display configuration
echo "🔧 Configuration:"
echo "   - Sample fraction: $SAMPLE_FRAC"
echo "   - Force recompute: $([ -n "$FORCE_FLAG" ] && echo "Yes" || echo "No")"
echo ""

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Define log file in shared directory
LOG_FILE="artifacts/logs/pipeline_run_${TIMESTAMP}.log"
echo "📝 Logging to: $LOG_FILE"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Start logging
{
    echo "============================================"
    echo "Pipeline Run Started: $(date)"
    echo "Configuration:"
    echo "  - Sample Fraction: $SAMPLE_FRAC"
    echo "  - Force Recompute: $([ -n "$FORCE_FLAG" ] && echo "Yes" || echo "No")"
    echo "============================================"
} > "$LOG_FILE"

log "🛠️  Shutting down existing containers..."
docker-compose down 2>&1 | tee -a "$LOG_FILE"

echo "📁 Delete old artifacts from earlier run" 
# Delete old artifacts directory if it exists
if [ -d "artifacts" ]; then
    echo "🗑️  Removing old artifacts directory..."
    rm -rf artifacts
fi

# Since artifacts is shared, create logs subdirectory there
echo "📁 Setting up logging in shared artifacts directory..."
mkdir -p artifacts/logs

log "🛠️  Building Docker images..."
docker-compose build 2>&1 | tee -a "$LOG_FILE"

log "🚀 Starting services..."
docker-compose up -d 2>&1 | tee -a "$LOG_FILE"

# Start background log collection from all containers
log "📋 Starting container log collection..."
docker-compose logs -f --timestamps >> "$LOG_FILE" 2>&1 &
LOGS_PID=$!

# Function to cleanup on exit
cleanup() {
    log "🧹 Cleaning up..."
    if [ ! -z "${LOGS_PID:-}" ]; then
        kill $LOGS_PID 2>/dev/null || true
    fi
    log "Pipeline finished at: $(date)"
}
trap cleanup EXIT

# Wait for services to be ready
log "⏳ Waiting for services to initialize..."
MAX_WAIT=60
WAIT_COUNT=0
while ! docker-compose ps | grep -q "healthy"; do
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
        log "❌ Services failed to become healthy after ${MAX_WAIT}s"
        exit 1
    fi
    echo -n "."
done
echo ""
log "✅ Services are healthy"

log "🐳 Running full pipeline..."
log "   Using sample_frac=$SAMPLE_FRAC"

# Run pipeline and capture output
docker-compose exec web bash -c "
    set -euo pipefail
    
    # Create a pipeline-specific log inside container
    CONTAINER_LOG=/app/artifacts/logs/container_pipeline_${TIMESTAMP}.log
    
    echo '📦 Verifying Python dependencies...' | tee -a \$CONTAINER_LOG
    pip install -r requirements.txt --quiet 2>&1 | tee -a \$CONTAINER_LOG || true

    echo '🧠 Training with sample_frac=$SAMPLE_FRAC...' | tee -a \$CONTAINER_LOG
    echo '📝 Note: Will use cached embeddings if available (add --force to recompute)' | tee -a \$CONTAINER_LOG
    python3 web/diet_classifiers.py --train --mode both --sample_frac $SAMPLE_FRAC $FORCE_FLAG 2>&1 | tee -a \$CONTAINER_LOG

    echo '🧪 Evaluating on provided gold set...' | tee -a \$CONTAINER_LOG
    python3 web/diet_classifiers.py --ground_truth /app/data/ground_truth_sample.csv 2>&1 | tee -a \$CONTAINER_LOG

    echo '🥘 Classifying custom ingredient list...' | tee -a \$CONTAINER_LOG
    python3 web/diet_classifiers.py --ingredients 'almond flour, erythritol, egg whites' 2>&1 | tee -a \$CONTAINER_LOG
    
    echo '✅ Container pipeline complete!' | tee -a \$CONTAINER_LOG
" 2>&1 | tee -a "$LOG_FILE"

# Stop background log collection
if [ ! -z "${LOGS_PID:-}" ]; then
    kill $LOGS_PID 2>/dev/null || true
fi

# Capture final container status
log "📊 Capturing final container status..."
docker-compose ps >> "$LOG_FILE" 2>&1

# List artifacts generated
log "📦 Artifacts generated:"
ls -la artifacts/ | tee -a "$LOG_FILE"

# Create summary file
SUMMARY_FILE="artifacts/logs/summary_${TIMESTAMP}.txt"
{
    echo "Pipeline Run Summary"
    echo "==================="
    echo "Start Time: $(head -n 3 "$LOG_FILE" | tail -n 1)"
    echo "End Time: $(date)"
    echo "Configuration:"
    echo "  - Sample Fraction: $SAMPLE_FRAC"
    echo "  - Force Recompute: $([ -n "$FORCE_FLAG" ] && echo "Yes" || echo "No")"
    echo ""
    echo "Log Files:"
    echo "  - Main log: $LOG_FILE"
    echo "  - Container log: artifacts/logs/container_pipeline_${TIMESTAMP}.log"
    echo "  - Python Source Code Logger:  artifacts/logs/diet_classifiers.py.log (if exists)"
    echo ""
    echo "Artifacts Generated:"
    ls -la artifacts/*.pkl artifacts/*.csv artifacts/*.json 2>/dev/null || echo "No model artifacts yet"
    echo ""
    echo "Container Final Status:"
    docker-compose ps
} > "$SUMMARY_FILE"

log "✅ Pipeline complete!"
echo ""
echo "📊 Results saved to artifacts/"
echo "📋 Logs available at:"
echo "   - Main log: $LOG_FILE"
echo "   - Summary: $SUMMARY_FILE"
echo "   - Python Source Code Logger:  artifacts/logs/diet_classifiers.py.log"
echo ""
echo "💡 View logs:"
echo "   tail -f $LOG_FILE                    # Follow main log"
echo "   cat $SUMMARY_FILE                    # View summary"
echo "   docker-compose logs -f web           # Live container logs"

# Also create a symlink to latest log for convenience
ln -sf "logs/pipeline_run_${TIMESTAMP}.log" "artifacts/latest_run.log" 2>/dev/null || true