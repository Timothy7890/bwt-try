"""One-shot pipeline: manual_labels/*.txt|ply -> labels.npy, objects.json, mask splats.

Usage:  python3 tools/update_labels.py
Run this after adding or renaming files in manual_labels/, then refresh the viewer.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
for step in ("build_labels.py", "make_objects.py", "make_mask_splats.py"):
    print(f"\n=== {step} ===")
    subprocess.run([sys.executable, str(HERE / step)], check=True)
print("\ndone - refresh the viewer")
