#!/usr/bin/env bash
#
# Setup script for YOHAS 3.0 benchmark evaluation.
# Clones benchmark repos, installs dependencies, verifies imports.
#
# Usage:
#   chmod +x scripts/benchmarks/setup_benchmarks.sh
#   ./scripts/benchmarks/setup_benchmarks.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENDOR_DIR="$PROJECT_ROOT/vendor"

echo "=== YOHAS 3.0 Benchmark Setup ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# ─── Create directories ──────────────────────────────────────────────────────

mkdir -p "$VENDOR_DIR"
mkdir -p "$PROJECT_ROOT/results/bixbench"
mkdir -p "$PROJECT_ROOT/results/biomni"
mkdir -p "$PROJECT_ROOT/results/labbench"

# ─── Clone and install BixBench ───────────────────────────────────────────────

echo "── BixBench ──"
if [ -d "$VENDOR_DIR/BixBench" ]; then
    echo "BixBench already cloned, pulling latest..."
    cd "$VENDOR_DIR/BixBench" && git pull --ff-only 2>/dev/null || true
else
    echo "Cloning BixBench..."
    git clone --depth 1 https://github.com/Future-House/BixBench.git "$VENDOR_DIR/BixBench"
fi
echo "Installing BixBench..."
pip install -e "$VENDOR_DIR/BixBench" 2>&1 | tail -3
echo ""

# ─── Clone and install Biomni ─────────────────────────────────────────────────

echo "── Biomni ──"
if [ -d "$VENDOR_DIR/Biomni" ]; then
    echo "Biomni already cloned, pulling latest..."
    cd "$VENDOR_DIR/Biomni" && git pull --ff-only 2>/dev/null || true
else
    echo "Cloning Biomni..."
    git clone --depth 1 https://github.com/snap-stanford/Biomni.git "$VENDOR_DIR/Biomni"
fi
echo "Installing Biomni..."
pip install -e "$VENDOR_DIR/Biomni" 2>&1 | tail -3
echo ""

# ─── Clone and install LAB-Bench ──────────────────────────────────────────────

echo "── LAB-Bench ──"
if [ -d "$VENDOR_DIR/LAB-Bench" ]; then
    echo "LAB-Bench already cloned, pulling latest..."
    cd "$VENDOR_DIR/LAB-Bench" && git pull --ff-only 2>/dev/null || true
else
    echo "Cloning LAB-Bench..."
    git clone --depth 1 https://github.com/Future-House/LAB-Bench.git "$VENDOR_DIR/LAB-Bench"
fi
echo "Installing LAB-Bench..."
pip install -e "$VENDOR_DIR/LAB-Bench" 2>&1 | tail -3
echo ""

# ─── Verify imports ──────────────────────────────────────────────────────────

echo "── Verifying imports ──"

PASS=true

echo -n "  bixbench... "
if python -c "import bixbench; print('OK')" 2>/dev/null; then
    :
else
    echo "FAILED (non-critical — check pip install output above)"
    PASS=false
fi

echo -n "  biomni... "
if python -c "from biomni.eval import BiomniEval1; print('OK')" 2>/dev/null; then
    :
else
    echo "FAILED (non-critical — check pip install output above)"
    PASS=false
fi

echo -n "  labbench... "
if python -c "import labbench; print('OK')" 2>/dev/null; then
    :
else
    echo "FAILED (non-critical — check pip install output above)"
    PASS=false
fi

# Verify YOHAS backend imports
echo -n "  yohas backend... "
cd "$PROJECT_ROOT"
if PYTHONPATH="$PROJECT_ROOT/backend:$PYTHONPATH" python -c "
from core.models import ResearchConfig, AgentType
from core.llm import LLMClient
from orchestrator.research_loop import ResearchOrchestrator
from world_model.knowledge_graph import InMemoryKnowledgeGraph
from agents import get_template, AGENT_TEMPLATES
print('OK')
" 2>/dev/null; then
    :
else
    echo "FAILED — ensure backend dependencies are installed"
    PASS=false
fi

echo ""

if [ "$PASS" = true ]; then
    echo "=== All imports verified. Ready to run benchmarks! ==="
else
    echo "=== Some imports failed. Check errors above and install missing deps. ==="
    echo "    Benchmarks can still run — failed imports will be caught at runtime."
fi

echo ""
echo "Usage:"
echo "  python scripts/benchmarks/run_bixbench.py --mode zero-shot --limit 10"
echo "  python scripts/benchmarks/run_biomni_eval1.py --tasks all --limit 50"
echo "  python scripts/benchmarks/run_labbench.py --categories DbQA,SeqQA,LitQA2 --limit 100"
