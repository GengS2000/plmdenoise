"""Python 3.8-compatible external-worker persistence; no simulation or fitting imports."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import numpy as np

def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)

def atomic_npz(path, arrays):
    import numpy as np
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temp.replace(path)

def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

def identifiers(shape):
    subjects, visits, taxa = shape
    taxon_ids = [f"taxon_{j + 1:04d}" for j in range(taxa)]
    sample_ids = [f"subject_{i + 1:03d}__visit_{t + 1:02d}"
                  for i in range(subjects) for t in range(visits)]
    return sample_ids, taxon_ids

def export_matrix(path: Path, values):
    values = np.asarray(values)
    if values.ndim != 3 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected finite nonnegative subject x visit x taxon values")
    samples, taxa = identifiers(values.shape)
    write_csv(path, ["sample_id"] + taxa,
              (dict(zip(["sample_id"] + taxa, [sid] + row.tolist()))
               for sid, row in zip(samples, values.reshape(-1, values.shape[-1]))))
