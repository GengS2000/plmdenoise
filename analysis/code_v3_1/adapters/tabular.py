"""Text interchange between the Python driver and the R / Octave adapters.

One row per (subject, visit) of the *whole* grid, so missingness is explicit and
the subject / time / taxon order round-trips by name rather than by assumption:

    subject  visit  time  taxon_00000 ... taxon_<J-1>

Unobserved visits carry `NA` in every taxon column -- never a zero, which a
count table would read as "sequenced and found nothing".  The reader checks that
the returned key columns are identical to the ones written, so a runner that
reorders or drops samples fails instead of producing a silently permuted result.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

KEY = ("subject", "visit", "time")
NA = "NA"


def taxon_columns(J):
    return [f"taxon_{j:05d}" for j in range(J)]


def _open(path, mode):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="\n")
    return path.open(mode, encoding="utf-8", newline="\n")


def write_counts(path, truth, keep=None):
    """Write counts for every (subject, visit); NA where the visit is not kept."""
    Y = np.asarray(truth["Y"])
    times = np.asarray(truth["times"], dtype=float)
    I, T, J = Y.shape
    keep = np.asarray(truth["mask"]).all(axis=2) if keep is None else np.asarray(keep, bool)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _open(path, "w") as handle:
        handle.write("\t".join(KEY + tuple(taxon_columns(J))) + "\n")
        for i in range(I):
            for t in range(T):
                head = [str(i), str(t), repr(float(times[i, t]))]
                body = ([str(int(v)) for v in Y[i, t]] if keep[i, t] else [NA] * J)
                handle.write("\t".join(head + body) + "\n")
    return keep


def read_values(path, shape):
    """Read a runner's table back into (I, T, J); NA becomes NaN.

    Raises ValueError if the key columns are not exactly the grid that was
    written, in the order it was written.
    """
    I, T, J = shape
    values = np.full(shape, np.nan)
    seen = []
    with _open(path, "r") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header[:3] != list(KEY):
            raise ValueError(f"{Path(path).name}: expected key columns {KEY}, got {header[:3]}")
        if len(header) - 3 != J:
            raise ValueError(f"{Path(path).name}: {len(header) - 3} value columns, expected {J}")
        if header[3:] != taxon_columns(J):
            raise ValueError(f"{Path(path).name}: taxon columns were renamed or reordered")
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            i, t = int(fields[0]), int(fields[1])
            seen.append((i, t))
            row = [np.nan if f in (NA, "NaN", "nan", "") else float(f) for f in fields[3:]]
            values[i, t] = row
    expected = [(i, t) for i in range(I) for t in range(T)]
    if seen != expected:
        raise ValueError(f"{Path(path).name}: subject/visit order did not round-trip "
                         f"({len(seen)} rows, first mismatch at "
                         f"{next((k for k, (a, b) in enumerate(zip(seen, expected)) if a != b), 'end')})")
    return values


# --------------------------------------------------- numeric-only CSV variant
# MATLAB/Octave read plain numeric matrices (`dlmread`) but not headers or "NA",
# so the same grid is written with a sentinel instead of a missing marker.
MISSING = -1.0


def write_counts_csv(path, truth, keep=None, missing=MISSING):
    """subject,visit,time,counts...  with `missing` (default -1) for unkept visits."""
    Y = np.asarray(truth["Y"], dtype=float)
    times = np.asarray(truth["times"], dtype=float)
    I, T, J = Y.shape
    keep = np.asarray(truth["mask"]).all(axis=2) if keep is None else np.asarray(keep, bool)
    rows = np.empty((I * T, 3 + J))
    for i in range(I):
        for t in range(T):
            row = rows[i * T + t]
            row[0], row[1], row[2] = i, t, times[i, t]
            row[3:] = Y[i, t] if keep[i, t] else missing
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, delimiter=",", fmt="%.10g")
    return keep


def read_values_csv(path, shape, missing=MISSING):
    """Read the numeric variant back, checking the key columns round-tripped."""
    I, T, J = shape
    rows = np.atleast_2d(np.loadtxt(path, delimiter=","))
    if rows.shape != (I * T, 3 + J):
        raise ValueError(f"{Path(path).name}: shape {rows.shape}, expected {(I * T, 3 + J)}")
    expected = np.array([(i, t) for i in range(I) for t in range(T)], dtype=float)
    if not np.array_equal(rows[:, :2], expected):
        raise ValueError(f"{Path(path).name}: subject/visit order did not round-trip")
    values = rows[:, 3:].reshape(shape)
    return np.where(values == missing, np.nan, values)
