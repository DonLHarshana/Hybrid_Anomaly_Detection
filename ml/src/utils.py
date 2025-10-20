# ml/src/utils.py
import json
from pathlib import Path
from datetime import datetime

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text())

def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))

def log(msg: str) -> None:
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    print(f"[{ts}] {msg}")

def align_dataframe_features(df, expected_order, fill_value=0.0):
    """
    Returns a new DataFrame with columns exactly in expected_order.
    - Missing columns are created with fill_value.
    - Extra columns in df are ignored.
    """
    for col in expected_order:
        if col not in df.columns:
            df[col] = fill_value
    return df[expected_order]
