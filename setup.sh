#!/bin/bash
# ══════════════════════════════════════════════════════════════════════
# setup.sh — Professional DS Project Template
# Usage: bash setup.sh
# Compatible: Python 3.12 | WSL Ubuntu | VS Code
# Reusable for any Data Science project
# ══════════════════════════════════════════════════════════════════════

PROJECT_NAME=$(basename "$PWD")
KERNEL_NAME="PROJECT_ENVIRONMENT"
VENV_DIR=".venv"

echo "======================================================"
echo "   DS Project Setup"
echo "   Project   : $PROJECT_NAME"
echo "   Python    : $(python3 --version)"
echo "   Directory : $(pwd)"
echo "======================================================"

# ── 1. Check Python 3.10+ ─────────────────────────────────────────────
PYTHON_VERSION=$(python3 -c 'import sys; print(sys.version_info.minor)')
if [ "$PYTHON_VERSION" -lt 10 ]; then
    echo "ERROR: Python 3.10 or higher is required"
    exit 1
fi

# ── 2. Remove previous environment if exists ──────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo ""
    echo "[0/6] Removing previous environment..."
    rm -rf $VENV_DIR
    echo "      Previous environment removed"
fi

# ── 3. Create virtual environment ─────────────────────────────────────
echo ""
echo "[1/6] Creating virtual environment..."
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate
pip install --upgrade pip setuptools wheel --quiet
echo "      Environment created: $VENV_DIR"

# ── 4. Install dependencies ───────────────────────────────────────────
echo ""
echo "[2/6] Installing dependencies..."
echo "      (this may take 5-10 minutes the first time)"
pip install -r requirements.txt
echo "      Dependencies installed"

# ── 5. Register kernel in Jupyter ────────────────────────────────────
echo ""
echo "[3/6] Registering Jupyter kernel..."
python -m ipykernel install \
    --user \
    --name="$KERNEL_NAME" \
    --display-name="PROJECT_ENVIRONMENT"
echo "      Kernel registered: PROJECT_ENVIRONMENT"

# ── 6. Create folder structure ───────────────────────────────────────
echo ""
echo "[4/6] Creating folder structure..."

folders=(
    "data/raw"
    "data/processed"
    "models"
    "notebooks"
    "src"
    "tests"
    "app"
    ".github/workflows"
    "reports"
    "logs"
)

for folder in "${folders[@]}"; do
    mkdir -p "$folder"
    touch "$folder/.gitkeep"
done

echo "      Structure created"

# ── 7. Create configuration files ────────────────────────────────────
echo ""
echo "[5/6] Creating configuration files..."

# .gitignore
if [ ! -f .gitignore ]; then
cat > .gitignore << 'EOF'
# Virtual Environment
.venv/
env/
venv/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/

# Raw Data (version with DVC)
data/raw/*.csv
data/raw/*.xlsx
data/raw/*.json
data/raw/*.parquet

# Serialized Models
models/*.pkl
models/*.joblib
models/*.h5

# MLflow
mlruns/
mlartifacts/

# Environment Variables
.env

# VS Code
.vscode/
*.code-workspace

# macOS / Windows
.DS_Store
Thumbs.db

# Generated logs and reports
logs/*.log
reports/*.html
EOF
echo "      .gitignore created"
fi

# .env.example
if [ ! -f .env.example ]; then
cat > .env.example << 'EOF'
# Project Environment Variables
# Copy as .env and fill in the values

DATA_PATH=data/raw/
MODEL_PATH=models/model.pkl
MLFLOW_TRACKING_URI=http://localhost:5000
API_HOST=0.0.0.0
API_PORT=8000
EOF
echo "      .env.example created"
fi

# ── 8. Final verification ─────────────────────────────────────────────
echo ""
echo "[6/6] Verifying installation..."

python3 -c "
libs = {
    'pandas'          : 'pandas',
    'numpy'           : 'numpy',
    'matplotlib'      : 'matplotlib',
    'seaborn'         : 'seaborn',
    'sklearn'         : 'scikit-learn',
    'lightgbm'        : 'lightgbm',
    'xgboost'         : 'xgboost',
    'imblearn'        : 'imbalanced-learn',
    'flaml'           : 'flaml',
    'ydata_profiling' : 'ydata-profiling',
    'shap'            : 'shap',
    'mlflow'          : 'mlflow',
    'joblib'          : 'joblib',
    'fastapi'         : 'fastapi',
    'uvicorn'         : 'uvicorn',
    'pytest'          : 'pytest',
}

failed = []
for lib, name in libs.items():
    try:
        __import__(lib)
        print(f'      OK  {name}')
    except ImportError:
        print(f'      FAILED  {name}')
        failed.append(name)

if failed:
    print(f'')
    print(f'      Libraries with errors: {failed}')
    print(f'      Run: pip install {\" \".join(failed)}')
else:
    print(f'')
    print(f'      All libraries verified correctly')
"

echo ""
echo "======================================================"
echo "  SETUP COMPLETE - $PROJECT_NAME"
echo "======================================================"
echo ""
echo "  NEXT STEPS:"
echo ""
echo "  1. Activate environment (every time you open VS Code):"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Select kernel in the notebook:"
echo "     PROJECT_ENVIRONMENT"
echo ""
echo "  3. Start MLflow (for experiment tracking):"
echo "     mlflow ui"
echo ""
echo "  4. Verify tests:"
echo "     pytest tests/"
echo ""
echo "======================================================"
