#!/usr/bin/env bash
# ============================================
# run_full_pipeline.sh - Enhanced with state preservation
# ============================================

set -euo pipefail

# Default values
SAMPLE_FRAC="1.0"
FORCE_FLAG=""
RESUME_FLAG=""  # New flag for resuming

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --sample_frac <value>   Set sample fraction for training (default: 1.0)"
    echo "  --force                 Force recomputation of embeddings"
    echo "  --resume                Attempt to resume from saved state"
    echo "  --clean                 Clean all state and start fresh"
    echo "  --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                      # Use defaults"
    echo "  $0 --resume             # Resume from checkpoint"
    echo "  $0 --clean              # Start fresh"
    exit 1
}

# Parse command line arguments
CLEAN_STATE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --sample_frac)
            if [ -z "${2:-}" ]; then
                echo "❌ Error: --sample_frac requires a value"
                usage
            fi
            SAMPLE_FRAC="$2"
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
        --resume)
            RESUME_FLAG="--resume"
            shift
            ;;
        --clean)
            CLEAN_STATE=true
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
echo "   - Resume from state: $([ -n "$RESUME_FLAG" ] && echo "Yes" || echo "No")"
echo "   - Clean state: $CLEAN_STATE"
echo ""

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Directory structure
ARTIFACTS_DIR="artifacts"
PIPELINE_STATE_DIR="pipeline_state"
BACKUPS_DIR="backups"
PRETRAINED_DIR="pretrained_models"

# Ensure critical directories exist
mkdir -p "$ARTIFACTS_DIR/logs"
mkdir -p "$PIPELINE_STATE_DIR/checkpoints"
mkdir -p "$PIPELINE_STATE_DIR/cache"

LOG_FILE="$ARTIFACTS_DIR/logs/pipeline_run_${TIMESTAMP}.log"
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
    echo "  - Resume: $([ -n "$RESUME_FLAG" ] && echo "Yes" || echo "No")"
    echo "============================================"
} > "$LOG_FILE"

log "🛠️  Shutting down existing containers..."
docker-compose down 2>&1 | tee -a "$LOG_FILE"

# Handle state cleaning if requested
if [ "$CLEAN_STATE" = true ]; then
    log "🧹 Cleaning all state as requested..."
    rm -rf "$PIPELINE_STATE_DIR"
    mkdir -p "$PIPELINE_STATE_DIR/checkpoints"
    mkdir -p "$PIPELINE_STATE_DIR/cache"
    log "✅ State cleaned"
fi

# Check pipeline state
if [ -n "$RESUME_FLAG" ] && [ -f "$PIPELINE_STATE_DIR/pipeline_state.pkl" ]; then
    log "📂 Found pipeline state - will attempt to resume"
else
    log "📂 No pipeline state found or resume not requested"
fi

# Preserve and clean artifacts directory
if [ -d "$ARTIFACTS_DIR" ]; then
    echo "📁 Found existing artifacts directory"
    
    # Preserve logs
    if [ -d "$ARTIFACTS_DIR/logs" ]; then
        echo "📋 Preserving previous log files..."
        mv "$ARTIFACTS_DIR/logs" "$ARTIFACTS_DIR/logs_backup_${TIMESTAMP}"
    fi
    
    # Preserve important results
    if [ -f "$ARTIFACTS_DIR/pipeline_results_summary.csv" ]; then
        echo "📊 Preserving previous results..."
        mkdir -p "$ARTIFACTS_DIR/previous_runs"
        cp "$ARTIFACTS_DIR"/*.csv "$ARTIFACTS_DIR/previous_runs/" 2>/dev/null || true
    fi
    
    echo "🗑️  Cleaning artifacts directory..."
    # Remove everything except logs_backup and previous_runs
    find "$ARTIFACTS_DIR" -mindepth 1 -not -path "$ARTIFACTS_DIR/logs_backup_*" -not -path "$ARTIFACTS_DIR/previous_runs*" -exec rm -rf {} + 2>/dev/null || true
    
    # Restore logs
    if [ -d "$ARTIFACTS_DIR/logs_backup_${TIMESTAMP}" ]; then
        mv "$ARTIFACTS_DIR/logs_backup_${TIMESTAMP}" "$ARTIFACTS_DIR/logs"
        echo "✅ Previous logs restored"
    fi
fi

# Ensure directory structure
mkdir -p "$ARTIFACTS_DIR/logs"
mkdir -p "$ARTIFACTS_DIR/metrics"

# ============================================
# Restore from backups (with state awareness)
# ============================================

# Function to restore files from a directory
restore_from_directory() {
    local source_dir=$1
    local dir_name=$2
    
    if [ -d "$source_dir" ]; then
        log "📂 Found $dir_name directory - copying files to artifacts..."
        
        local file_count=$(find "$source_dir" -type f 2>/dev/null | wc -l)
        
        if [ "$file_count" -gt 0 ]; then
            log "   Found $file_count files to restore"
            
            # Copy embeddings
            if ls "$source_dir"/embeddings_*.npy 2>/dev/null; then
                log "   📊 Copying embedding files..."
                cp -v "$source_dir"/embeddings_*.npy "$ARTIFACTS_DIR/" 2>&1 | tee -a "$LOG_FILE" || true
            fi
            
            # Copy models and vectorizers
            if ls "$source_dir"/*.pkl 2>/dev/null; then
                log "   🤖 Copying model files..."
                # Don't copy pipeline_state files
                find "$source_dir" -name "*.pkl" -not -name "pipeline_state*.pkl" -exec cp -v {} "$ARTIFACTS_DIR/" \; 2>&1 | tee -a "$LOG_FILE" || true
            fi
            
            # Copy CSV files
            if ls "$source_dir"/*.csv 2>/dev/null; then
                log "   📄 Copying CSV files..."
                cp -v "$source_dir"/*.csv "$ARTIFACTS_DIR/" 2>&1 | tee -a "$LOG_FILE" || true
            fi
            
            # Copy JSON files
            if ls "$source_dir"/*.json 2>/dev/null; then
                log "   📋 Copying JSON files..."
                cp -v "$source_dir"/*.json "$ARTIFACTS_DIR/" 2>&1 | tee -a "$LOG_FILE" || true
            fi
            
            log "   ✅ Restoration from $dir_name complete"
        else
            log "   ⚠️  $dir_name directory is empty - nothing to copy"
        fi
    else
        log "📂 No $dir_name directory found - skipping"
    fi
}

# Restore from various sources
restore_from_directory "$BACKUPS_DIR" "backups"
restore_from_directory "$PRETRAINED_DIR" "pretrained_models"

# ============================================
# Docker operations
# ============================================

log "🛠️  Building Docker images..."
docker-compose build 2>&1 | tee -a "$LOG_FILE"

log "🚀 Starting services..."
docker-compose up -d 2>&1 | tee -a "$LOG_FILE"

# Start background log collection
log "📋 Starting container log collection..."
docker-compose logs -f --timestamps >> "$LOG_FILE" 2>&1 &
LOGS_PID=$!

# Cleanup function
cleanup() {
    log "🧹 Cleaning up..."
    if [ ! -z "${LOGS_PID:-}" ]; then
        kill $LOGS_PID 2>/dev/null || true
    fi
    
    # Save important files to backups
    if [ -f "$ARTIFACTS_DIR/models.pkl" ] && [ -f "$ARTIFACTS_DIR/vectorizer.pkl" ]; then
        log "💾 Backing up trained models..."
        mkdir -p "$BACKUPS_DIR"
        cp "$ARTIFACTS_DIR/models.pkl" "$BACKUPS_DIR/models_${TIMESTAMP}.pkl"
        cp "$ARTIFACTS_DIR/vectorizer.pkl" "$BACKUPS_DIR/vectorizer_${TIMESTAMP}.pkl"
    fi
    
    log "Pipeline finished at: $(date)"
}
trap cleanup EXIT

# Wait for services
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

# Run pipeline with resume support
log "🐳 Running full pipeline..."
log "   Using sample_frac=$SAMPLE_FRAC"

docker-compose exec web bash -c "
    set -euo pipefail
    
    # Ensure directories exist in container
    mkdir -p /app/artifacts/logs
    mkdir -p /app/pipeline_state/checkpoints
    mkdir -p /app/pipeline_state/cache
    
    CONTAINER_LOG=/app/artifacts/logs/container_pipeline_${TIMESTAMP}.log
    
    echo '📦 Verifying Python dependencies...' | tee -a \$CONTAINER_LOG
    pip install -r requirements.txt --quiet 2>&1 | tee -a \$CONTAINER_LOG || true

    # Check if we should resume
    if [ -n '$RESUME_FLAG' ] && [ -f '/app/pipeline_state/pipeline_state.pkl' ]; then
        echo '🔄 Resume flag set - pipeline will attempt to resume from saved state' | tee -a \$CONTAINER_LOG
    fi

    echo '🧠 Training with sample_frac=$SAMPLE_FRAC...' | tee -a \$CONTAINER_LOG
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

# Status and summary
log "📊 Final status:"
docker-compose ps >> "$LOG_FILE" 2>&1

log "📦 Artifacts generated:"
ls -la "$ARTIFACTS_DIR/" 2>/dev/null | tee -a "$LOG_FILE" || echo "No artifacts found"

log "💾 Pipeline state:"
ls -la "$PIPELINE_STATE_DIR/" 2>/dev/null | tee -a "$LOG_FILE" || echo "No state found"

# Create summary
SUMMARY_FILE="$ARTIFACTS_DIR/logs/summary_${TIMESTAMP}.txt"
{
    echo "Pipeline Run Summary"
    echo "==================="
    echo "Start Time: $(head -n 3 "$LOG_FILE" | tail -n 1)"
    echo "End Time: $(date)"
    echo "Configuration:"
    echo "  - Sample Fraction: $SAMPLE_FRAC"
    echo "  - Force Recompute: $([ -n "$FORCE_FLAG" ] && echo "Yes" || echo "No")"
    echo "  - Resume: $([ -n "$RESUME_FLAG" ] && echo "Yes" || echo "No")"
    echo ""
    echo "Directories:"
    echo "  - Artifacts: $ARTIFACTS_DIR/"
    echo "  - Pipeline State: $PIPELINE_STATE_DIR/"
    echo "  - Backups: $BACKUPS_DIR/"
    echo ""
    echo "Log Files:"
    echo "  - Main log: $LOG_FILE"
    echo "  - Container log: $ARTIFACTS_DIR/logs/container_pipeline_${TIMESTAMP}.log"
    echo ""
    echo "Pipeline State Available:"
    ls -la "$PIPELINE_STATE_DIR"/*.pkl 2>/dev/null || echo "No state files"
    echo ""
    echo "Checkpoints Available:"
    ls -la "$PIPELINE_STATE_DIR/checkpoints"/*.pkl 2>/dev/null || echo "No checkpoints"
} > "$SUMMARY_FILE"

log "✅ Pipeline complete!"
echo ""
echo "📊 Results: $ARTIFACTS_DIR/"
echo "💾 State: $PIPELINE_STATE_DIR/"
echo "📋 Summary: $SUMMARY_FILE"
echo ""
echo "💡 Next run options:"
echo "   ./run_full_pipeline.sh --resume     # Resume from checkpoint"
echo "   ./run_full_pipeline.sh --clean      # Start fresh"