"""
Entry point — delegates to run_full_pipeline.py which is the single source of truth.

Usage:
    python run_full_pipeline.py        # recommended
    python main.py                     # identical (redirects here)
"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    pipeline = Path(__file__).parent / "run_full_pipeline.py"
    sys.argv[0] = str(pipeline)
    runpy.run_path(str(pipeline), run_name="__main__")
