"""Persistence and validation shared by preparation and server jobs."""
from pathlib import Path
import hashlib
import json
import math
import os
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value,float) and not math.isfinite(value):
        return None  # Unexposed event times / absent validation diagnostics.
    return value


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def write_npz(path, **arrays):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(temp, path)


def tensorize(counts, samples):
    """Fill slots INSIDE observed intervals; never extend observed time bounds.

    Missing slots have mask=False and are omitted from all exported sample matrices.
    counts is samples x taxa, samples indexed identically with subject_id/time/depth/batch.
    """
    if not counts.index.equals(samples.index) or not counts.index.is_unique:
        raise ValueError('Sample identity/order mismatch')
    a = counts.to_numpy(dtype=float)
    if not np.isfinite(a).all() or (a < 0).any() or (a != np.floor(a)).any():
        raise ValueError('Counts must be finite nonnegative integers')
    groups = list(samples.groupby('subject_id', sort=True))
    if not groups or any(len(g) < 2 for _, g in groups):
        raise ValueError('Every retained subject needs >=2 visits')
    I, T, J = len(groups), max(len(g) for _, g in groups), counts.shape[1]
    y = np.zeros((I, T, J), dtype=np.int64)
    mask = np.zeros((I, T, J), dtype=bool)
    times = np.zeros((I, T)); depth = np.ones((I, T))
    batch = np.zeros((I, T), dtype=np.int64)
    positions = {}
    for i, (subject, g) in enumerate(groups):
        g = g.sort_values('time')
        observed = g.time.to_numpy(dtype=float)
        if not np.isfinite(observed).all() or not (np.diff(observed) > 0).all():
            raise ValueError(f'Invalid/duplicate times for {subject}')
        slots = list(observed)
        while len(slots) < T:
            slots.sort()
            k = int(np.argmax(np.diff(slots)))
            midpoint = (slots[k] + slots[k+1]) / 2
            if not slots[k] < midpoint < slots[k+1]:
                raise ValueError('Insufficient floating point precision for padding')
            slots.append(midpoint)
        times[i] = sorted(slots)
        batch[i] = int(g.batch.iloc[0])
        for sid, row in g.iterrows():
            t = int(np.searchsorted(times[i], row.time))
            y[i,t] = counts.loc[sid].to_numpy(dtype=np.int64)
            mask[i,t] = True
            depth[i,t] = row.depth
            batch[i,t] = int(row.batch)
            positions[sid] = (i,t)
    if not np.isfinite(depth).all() or (depth <= 0).any():
        raise ValueError('Invalid exposure')
    samples = samples.copy()
    samples['tensor_i'] = [positions[s][0] for s in samples.index]
    samples['tensor_t'] = [positions[s][1] for s in samples.index]
    return dict(Y=y, times=times, depth=depth, batch=batch, mask=mask), samples
