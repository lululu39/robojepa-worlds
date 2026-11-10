"""Small stdlib-only helpers, also usable on a submit node."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def upstream_path():
    import sys
    path = str(ROOT / "vendor/robo_jepa")
    if path not in sys.path:
        sys.path.insert(0, path)
    os.environ["VJEPA2_HUB_REPO"] = str(ROOT / "vendor/vjepa2")
    return path
