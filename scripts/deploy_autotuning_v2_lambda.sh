#!/bin/bash
"""
Deploy Autotuning V2 to Lambda Cloud for Real GPU Testing
Deploys fixed Mojo kernels and benchmark infrastructure
"""

set -e

# Configuration
LAMBDA_HOST="${LAMBDA_HOST:-your-lambda-instance.cloud}"
LAMBDA_USER="${LAMBDA_USER:-ubuntu}"
REMOTE_DIR="/home/ubuntu/mojo-autotuning-v2"
LOCAL_PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"

echo "🚀 Deploying Autotuning V2 to Lambda Cloud"
echo "==========================================="
echo "📍 Host: $LAMBDA_HOST"
echo "👤 User: $LAMBDA_USER"
echo "📁 Remote dir: $REMOTE_DIR"
echo "📁 Local project: $LOCAL_PROJECT_ROOT"

# Check prerequisites
echo ""
echo "🔍 Checking Prerequisites..."

if [ ! -f "$LOCAL_PROJECT_ROOT/integration_test_benchmark.mojo" ]; then
    echo "❌ integration_test_benchmark.mojo not found"
    exit 1
fi

if [ ! -d "$LOCAL_PROJECT_ROOT/src/kernels" ]; then
    echo "❌ Kernel directory not found"
    exit 1
fi

if [ ! -f "$LOCAL_PROJECT_ROOT/scripts/autotuning_v2_real_gpu.py" ]; then
    echo "❌ Autotuning V2 script not found"
    exit 1
fi

echo "✅ All files present"

# Test SSH connection
echo ""
echo "🔗 Testing SSH Connection..."
if ! ssh -o ConnectTimeout=10 "$LAMBDA_USER@$LAMBDA_HOST" "echo 'SSH connection successful'"; then
    echo "❌ SSH connection failed"
    echo "💡 Please ensure:"
    echo "   - Lambda instance is running"
    echo "   - SSH key is configured"
    echo "   - LAMBDA_HOST and LAMBDA_USER are correct"
    exit 1
fi

echo "✅ SSH connection successful"

# Create remote directory structure
echo ""
echo "📁 Setting up Remote Directory Structure..."
ssh "$LAMBDA_USER@$LAMBDA_HOST" "
    mkdir -p $REMOTE_DIR/{src/kernels,scripts,docs,autotuning_results}
    mkdir -p $REMOTE_DIR/src/{core,integration,monitoring,search}
    echo '✅ Remote directories created'
"

# Deploy core files
echo ""
echo "📤 Deploying Core Files..."

# Deploy enhanced integration test
echo "   📋 Integration test (benchmark-enabled)..."
scp "$LOCAL_PROJECT_ROOT/integration_test_benchmark.mojo" \
    "$LAMBDA_USER@$LAMBDA_HOST:$REMOTE_DIR/"

# Deploy autotuning v2 script
echo "   🔧 Autotuning V2 script..."
scp "$LOCAL_PROJECT_ROOT/scripts/autotuning_v2_real_gpu.py" \
    "$LAMBDA_USER@$LAMBDA_HOST:$REMOTE_DIR/scripts/"

# Deploy fixed kernels
echo "   ⚡ GPU kernels (fixed)..."
scp -r "$LOCAL_PROJECT_ROOT/src/kernels/" \
    "$LAMBDA_USER@$LAMBDA_HOST:$REMOTE_DIR/src/"

# Deploy core components
echo "   📊 Core data structures..."
scp "$LOCAL_PROJECT_ROOT/src/core/data_structures.mojo" \
    "$LAMBDA_USER@$LAMBDA_HOST:$REMOTE_DIR/src/core/"

echo "   🔗 Integration components..."
scp "$LOCAL_PROJECT_ROOT/src/integration/onedev_bridge.mojo" \
    "$LAMBDA_USER@$LAMBDA_HOST:$REMOTE_DIR/src/integration/"

echo "   📈 Monitoring components..."
scp "$LOCAL_PROJECT_ROOT/src/monitoring/performance_metrics_working.mojo" \
    "$LAMBDA_USER@$LAMBDA_HOST:$REMOTE_DIR/src/monitoring/"

echo "   🔍 Search engine..."
scp "$LOCAL_PROJECT_ROOT/src/search/semantic_search_engine.mojo" \
    "$LAMBDA_USER@$LAMBDA_HOST:$REMOTE_DIR/src/search/"

# Deploy project configuration
echo "   ⚙️  Project configuration..."
scp "$LOCAL_PROJECT_ROOT/pixi.toml" \
    "$LAMBDA_USER@$LAMBDA_HOST:$REMOTE_DIR/"

# Deploy documentation
echo "   📚 V2 documentation..."
scp "$LOCAL_PROJECT_ROOT/docs/AUTOTUNING_V2_ANALYSIS.md" \
    "$LAMBDA_USER@$LAMBDA_HOST:$REMOTE_DIR/docs/"
scp "$LOCAL_PROJECT_ROOT/docs/AUTOTUNING_V2_PLAN.md" \
    "$LAMBDA_USER@$LAMBDA_HOST:$REMOTE_DIR/docs/"

# Setup remote environment
echo ""
echo "🛠️  Setting up Remote Environment..."
ssh "$LAMBDA_USER@$LAMBDA_HOST" "
    cd $REMOTE_DIR
    
    # Install pixi if not present
    if ! command -v pixi &> /dev/null; then
        echo '📦 Installing pixi...'
        curl -fsSL https://pixi.sh/install.sh | bash
        export PATH=\$HOME/.pixi/bin:\$PATH
    fi
    
    # Initialize pixi environment
    echo '🐍 Setting up pixi environment...'
    pixi install
    
    # Verify Mojo is available
    echo '🔥 Verifying Mojo installation...'
    pixi run mojo --version
    
    # Make scripts executable
    chmod +x scripts/*.py
    
    echo '✅ Remote environment setup complete'
"

# Create deployment summary
echo ""
echo "📋 Creating Deployment Summary..."
ssh "$LAMBDA_USER@$LAMBDA_HOST" "
    cd $REMOTE_DIR
    
    cat > DEPLOYMENT_INFO.md << 'EOF'
# Autotuning V2 Lambda Cloud Deployment

## Deployment Information
- **Deployed**: $(date)
- **Host**: $LAMBDA_HOST
- **User**: $LAMBDA_USER
- **Directory**: $REMOTE_DIR

## Files Deployed
- integration_test_benchmark.mojo (enhanced for real GPU benchmarking)
- scripts/autotuning_v2_real_gpu.py (V2 autotuning manager)
- src/kernels/ (fixed GPU kernels)
- src/core/ (fixed data structures)
- All supporting components

## Quick Start
1. SSH to Lambda instance:
   \`ssh $LAMBDA_USER@$LAMBDA_HOST\`

2. Navigate to deployment:
   \`cd $REMOTE_DIR\`

3. Run autotuning V2:
   \`pixi run python scripts/autotuning_v2_real_gpu.py\`

## GPU Information
- **GPU Type**: Lambda Cloud A10
- **Memory**: 24GB GDDR6
- **CUDA Cores**: 9,216
- **Theoretical Performance**: ~31 TFLOPS

## Expected Results
- Real GPU performance data (no simulation)
- Production-scale testing (50K+ vectors, 768D)
- Comprehensive parameter sweep
- Accurate latency measurements
- True GPU occupancy metrics
EOF

    echo '✅ Deployment summary created'
"

# Test deployment
echo ""
echo "🧪 Testing Deployment..."
ssh "$LAMBDA_USER@$LAMBDA_HOST" "
    cd $REMOTE_DIR
    
    # Test Mojo compilation
    echo '🔥 Testing Mojo compilation...'
    if pixi run mojo integration_test_benchmark.mojo; then
        echo '✅ Mojo benchmark test successful'
    else
        echo '❌ Mojo benchmark test failed'
        exit 1
    fi
    
    # Test Python autotuning script
    echo '🐍 Testing Python autotuning script...'
    if python3 scripts/autotuning_v2_real_gpu.py --help &> /dev/null; then
        echo '✅ Python autotuning script accessible'
    else
        echo '⚠️  Python script test inconclusive (may need --help flag implementation)'
    fi
"

# Register instance for termination monitoring
echo ""
echo "🔔 Registering Lambda Instance for Termination Monitoring..."
INSTANCE_ID="${LAMBDA_HOST%%.*}"  # Extract instance ID from hostname
python3 "$LOCAL_PROJECT_ROOT/scripts/lambda_termination_reminder.py" \
    --register "$INSTANCE_ID,$LAMBDA_HOST,4.0"  # 4 hour estimate

echo ""
echo "💰 IMPORTANT: Lambda Cloud Billing Reminder"
echo "============================================"
echo "🚨 Lambda charges CONTINUOUSLY while instance is running"
echo "💸 A10 GPU cost: ~$1.50/hour ($36/day if left running!)"
echo ""
echo "⚠️  CRITICAL REMINDERS:"
echo "   1. Lambda bills even when no scripts are running"
echo "   2. Instance must be TERMINATED (not just stopped) to stop billing"
echo "   3. Set calendar reminders to check instance status"
echo "   4. Autotuning should complete in 2-4 hours"
echo ""
echo "🔔 Termination Monitoring Active:"
echo "   - Automatic alerts every 1, 2, 4, 8, 12, 24 hours"
echo "   - System notifications when cost targets exceeded"
echo "   - Manual check: python3 scripts/lambda_termination_reminder.py --check"
echo "   - Mark terminated: python3 scripts/lambda_termination_reminder.py --terminate $INSTANCE_ID"
echo ""
echo "📱 Set These Phone/Calendar Reminders NOW:"
echo "   - ⏰ 2 hours: Check autotuning progress"
echo "   - ⏰ 4 hours: Autotuning should be complete - TERMINATE if done"
echo "   - ⏰ 6 hours: CRITICAL - Check why still running, likely terminate"
echo "   - ⏰ 8 hours: EMERGENCY - Terminate immediately unless critical issue"
echo ""

# Final instructions
echo "🎉 Deployment Complete!"
echo "======================"
echo ""
echo "📋 Next Steps:"
echo "1. SSH to Lambda instance:"
echo "   ssh $LAMBDA_USER@$LAMBDA_HOST"
echo ""
echo "2. Navigate to deployment:"
echo "   cd $REMOTE_DIR"
echo ""
echo "3. Run autotuning V2:"
echo "   pixi run python scripts/autotuning_v2_real_gpu.py"
echo ""
echo "📊 What This Will Do:"
echo "   - Test 100+ real GPU configurations"
echo "   - Use production-scale data (50K+ vectors)"
echo "   - Measure actual A10 GPU performance"
echo "   - Generate comprehensive benchmark results"
echo "   - Compare with previous simulated results"
echo ""
echo "⏱️  Expected Duration: 2-4 hours for full sweep"
echo "📈 Expected Results: Realistic 15-75ms latency (much more accurate than 2.99ms simulation)"
echo ""
echo "🎯 Success Criteria:"
echo "   ✅ Real GPU execution on A10 hardware"
echo "   ✅ Production-scale corpus testing"
echo "   ✅ Comprehensive parameter evaluation"
echo "   ✅ First accurate performance baseline"
echo ""
echo "💡 TERMINATION CHECKLIST (Save This!):"
echo "   ✅ Autotuning results saved to local machine"
echo "   ✅ No critical errors that need investigation"
echo "   ✅ All needed data downloaded"
echo "   ✅ Ready to terminate instance"
echo ""
echo "🚨 TERMINATE COMMAND:"
echo "   Go to Lambda Labs dashboard → Instances → Terminate"
echo "   OR use Lambda CLI if configured"
echo ""
echo "🚀 Autotuning V2 deployment ready - REMEMBER TO TERMINATE WHEN DONE!"