"""Shortcut: manual_labels/*.txt|ply -> labels.npy, objects.json, mask splats, then relink the viewer.

Usage:  python3 tools/update_labels.py <scene_dir>
Run this after adding or renaming files in <scene_dir>/manual_labels/, then refresh the viewer.
Same as: python3 tools/pipeline.py <scene_dir> --steps labels,link
"""
import subprocess
import sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit(__doc__)
subprocess.run([sys.executable, str(Path(__file__).parent / "pipeline.py"), sys.argv[1],
                "--steps", "labels,link"], check=True)
