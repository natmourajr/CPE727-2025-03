#!/bin/bash
# Diagnostic script to understand PyTorch environment differences

echo "=========================================="
echo "PYTORCH ENVIRONMENT DIAGNOSTICS"
echo "=========================================="
echo

echo "1. PYTHON VERSIONS"
echo "-------------------"
echo "System python3:"
which python3
python3 --version
echo

echo ".venv python:"
which .venv/bin/python
.venv/bin/python --version
echo

echo "uv run python:"
uv run python --version
echo

echo "2. TORCH INSTALLATION LOCATIONS"
echo "--------------------------------"
echo "In .venv:"
ls -lh .venv/lib/python3.*/site-packages/torch/_C*.so 2>/dev/null || echo "Not found in .venv"
echo

echo "Check for multiple torch installations:"
find .venv -name "torch" -type d 2>/dev/null
echo

echo "3. TORCH BINARY DETAILS"
echo "-----------------------"
if [ -f .venv/lib/python3.12/site-packages/torch/_C.cpython-312-darwin.so ]; then
    echo "Binary exists at: .venv/lib/python3.12/site-packages/torch/_C.cpython-312-darwin.so"
    echo "File type:"
    file .venv/lib/python3.12/site-packages/torch/_C.cpython-312-darwin.so
    echo
    echo "Binary size:"
    ls -lh .venv/lib/python3.12/site-packages/torch/_C.cpython-312-darwin.so
    echo
    echo "Binary timestamp:"
    stat -f "Created: %SB" .venv/lib/python3.12/site-packages/torch/_C.cpython-312-darwin.so
    echo
    echo "Checking for Python 3.13 symbols (should be empty):"
    nm .venv/lib/python3.12/site-packages/torch/_C.cpython-312-darwin.so 2>/dev/null | grep -i "pydict_getitemstringref" || echo "  ✓ No Python 3.13 symbols found (good!)"
    echo
else
    echo "❌ Binary not found!"
fi

echo "4. PYTHON IMPORT PATHS"
echo "----------------------"
echo "Using .venv/bin/python:"
.venv/bin/python -c "import sys; print('\n'.join(sys.path))"
echo

echo "Using uv run python:"
uv run python -c "import sys; print('\n'.join(sys.path))"
echo

echo "5. UV PYTHON DISCOVERY"
echo "----------------------"
echo "UV Python in use:"
uv python find
echo

echo "UV Python list:"
uv python list | head -5
echo

echo "6. ENVIRONMENT VARIABLES"
echo "------------------------"
echo "VIRTUAL_ENV: ${VIRTUAL_ENV:-not set}"
echo "UV_PYTHON: ${UV_PYTHON:-not set}"
echo "PYTHONPATH: ${PYTHONPATH:-not set}"
echo "PATH (first 3 entries):"
echo "$PATH" | tr ':' '\n' | head -3
echo

echo "7. ACTUAL IMPORT TEST"
echo "---------------------"
echo "Test 1: Using .venv/bin/python directly"
.venv/bin/python -c "
import sys
print(f'  Python: {sys.version}')
print(f'  Executable: {sys.executable}')
try:
    import torch
    print(f'  ✅ Torch: {torch.__version__}')
    print(f'  Torch location: {torch.__file__}')
except Exception as e:
    print(f'  ❌ Failed: {e}')
"
echo

echo "Test 2: Using uv run python"
uv run python -c "
import sys
print(f'  Python: {sys.version}')
print(f'  Executable: {sys.executable}')
try:
    import torch
    print(f'  ✅ Torch: {torch.__version__}')
    print(f'  Torch location: {torch.__file__}')
except Exception as e:
    print(f'  ❌ Failed: {e}')
" 2>&1 | grep -v "Uninstalled\|Installed"
echo

echo "Test 3: Using python3 from PATH"
python3 -c "
import sys
print(f'  Python: {sys.version}')
print(f'  Executable: {sys.executable}')
try:
    import torch
    print(f'  ✅ Torch: {torch.__version__}')
    print(f'  Torch location: {torch.__file__}')
except Exception as e:
    print(f'  ❌ Failed: {e}')
"
echo

echo "8. CHECK FOR CACHED/STALE FILES"
echo "--------------------------------"
echo "Torch __pycache__ directories:"
find .venv -path "*torch*__pycache__" -type d 2>/dev/null | wc -l | xargs echo "  Found:"
echo

echo "Torch .pyc files:"
find .venv -path "*torch*.pyc" 2>/dev/null | wc -l | xargs echo "  Found:"
echo

echo "9. UV CACHE INFO"
echo "----------------"
uv cache dir
echo

echo "10. SUBPROCESS SIMULATION (The Real Issue)"
echo "-------------------------------------------"
echo "Simulating how run_all_experiments.py calls uv run python:"
echo
echo "Test A: Direct uv run (shell → uv):"
uv run python -c "import sys; print(f'  Python: {sys.version_info[:2]}'); import torch; print(f'  ✅ Torch: {torch.__version__}')" 2>&1 | grep -v "Uninstalled\|Installed" || echo "  ❌ Failed"
echo

echo "Test B: Via Python 3.13 subprocess (python3.13 → uv):"
python3 -c "
import subprocess
result = subprocess.run(
    ['uv', 'run', 'python', '-c', 'import sys; print(f\"  Python: {sys.version_info[:2]}\"); import torch; print(f\"  ✅ Torch: {torch.__version__}\")'],
    capture_output=True,
    text=True
)
print(result.stdout if result.returncode == 0 else f'  ❌ Failed: {result.stderr[:200]}')
"
echo

echo "Test C: Via Python 3.12 subprocess (.venv/bin/python → uv):"
.venv/bin/python -c "
import subprocess
result = subprocess.run(
    ['uv', 'run', 'python', '-c', 'import sys; print(f\"  Python: {sys.version_info[:2]}\"); import torch; print(f\"  ✅ Torch: {torch.__version__}\")'],
    capture_output=True,
    text=True
)
print(result.stdout if result.returncode == 0 else f'  ❌ Failed: {result.stderr[:200]}')
"
echo

echo "11. DYNAMIC LINKER ENVIRONMENT"
echo "------------------------------"
echo "Environment variables that affect library loading:"
echo "  DYLD_LIBRARY_PATH: ${DYLD_LIBRARY_PATH:-not set}"
echo "  DYLD_FRAMEWORK_PATH: ${DYLD_FRAMEWORK_PATH:-not set}"
echo "  DYLD_FALLBACK_LIBRARY_PATH: ${DYLD_FALLBACK_LIBRARY_PATH:-not set}"
echo

echo "=========================================="
echo "DIAGNOSTICS COMPLETE"
echo "=========================================="
echo
echo "Please share the ENTIRE output with Claude."
echo "Pay special attention to:"
echo "  - Which Python versions are different"
echo "  - Where torch is actually located"
echo "  - Whether .venv/bin/python and 'uv run python' point to same place"
