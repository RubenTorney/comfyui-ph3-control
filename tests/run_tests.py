"""Minimal runner for environments without pytest: python tests/run_tests.py (ComfyUI's venv python)."""
import importlib
import inspect
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import conftest  # noqa: F401,E402

failed = total = 0
for f in sorted(Path(__file__).parent.glob("test_*.py")):
    mod = importlib.import_module(f.stem)
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("test_") and not inspect.signature(fn).parameters:
            total += 1
            try:
                fn()
                pass
            except Exception:
                failed += 1
                print("FAIL", f.stem, name)
                traceback.print_exc()
print(f"{total - failed}/{total} tests passed")
sys.exit(1 if failed else 0)
