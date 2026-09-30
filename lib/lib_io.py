"""
lib_io.py — I/O, provenance and exclusion accounting for the revision pipeline.

Design rules enforced here:
  * require()  — a referenced file that does not exist is a hard stop naming the
                 exact path. No similar file is ever substituted.
  * Ledger     — every row that leaves the cohort is counted against a named rule.
                 A step whose arithmetic does not balance raises. Nothing is
                 dropped silently anywhere in the pipeline.
  * Receptor status, which defines cohort membership, is never imputed. That rule
                 is not enforceable in code here, but no imputation helper is
                 provided by this module, by design.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
LIB_DIR = Path(__file__).resolve().parent
REVISION = LIB_DIR.parent
DATA = REVISION / "data"
# Both the output directory and the seed may be overridden by environment
# variable. This exists for one caller: scripts/18_montecarlo_error.py, which
# re-runs the RNG-dependent stages under alternate seeds into a scratch copy so
# that the canonical output/ is never touched. Unset, the defaults below are
# what every stage uses, so a plain run is bit-for-bit what it always was.
OUTPUT = Path(os.environ.get("REVISION_OUTPUT") or (REVISION / "output"))
PAPER = REVISION / "paper"

OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = int(os.environ.get("REVISION_SEED") or 20260802)

# GDC / TCGA null sentinels. Kept in one place so every script parses the raw
# files identically — the original pipeline redefined this list per script.
TCGA_NA = [
    "'--", "--", "NA", "N/A", "Not Reported", "not reported",
    "[Not Available]", "[Unknown]", "[Not Applicable]", "nan", "",
]


class MissingInput(FileNotFoundError):
    """Raised when a referenced input file is absent. Never caught internally."""


def require(path: Path, why: str = "") -> Path:
    """Return `path` if it exists, otherwise stop with an explicit message.

    The pipeline must fail loudly rather than fall back to a similar-looking
    file; several defects in the audited code trace to exactly that fallback
    behaviour (e.g. 04_create_treatment_arms.py searching three candidate
    locations for clinical.tsv).
    """
    path = Path(path)
    if not path.exists():
        msg = f"Required input not found: {path}"
        if why:
            msg += f"\n  Needed for: {why}"
        msg += "\n  Nothing was substituted. Supply this exact file and re-run."
        raise MissingInput(msg)
    return path


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _as_series(s) -> pd.Series:
    """Coerce to a Series while preserving the caller's index.

    pd.Series(some_index) silently attaches a fresh RangeIndex, so assigning the
    result back onto a DataFrame aligns to NaN. Every barcode helper goes
    through here to make that impossible.
    """
    if isinstance(s, pd.Series):
        return s
    if isinstance(s, pd.Index):
        return pd.Series(s.to_numpy(), index=s)
    return pd.Series(s)


def barcode12(s) -> pd.Series:
    """Normalise a TCGA barcode column to the 12-character patient identifier."""
    return _as_series(s).astype(str).str.strip().str.upper().str[:12]


def site_code(patient_id) -> pd.Series:
    """Tissue source site = characters 6 and 7 of the barcode (0-indexed [5:7]).

    TCGA-XX-YYYY  ->  XX
    """
    return _as_series(patient_id).astype(str).str.upper().str[5:7]


# ── Exclusion accounting ──────────────────────────────────────────────────────
class Ledger:
    """Records every exclusion so no row can leave the cohort unaccounted for.

    Usage:
        led = Ledger("TNBC cohort")
        df = led.start(df, "all patients with an outcome record")
        df = led.exclude(df, mask_keep, "ER status not directly measured")
    """

    def __init__(self, name: str):
        self.name = name
        self.rows: list[dict] = []
        self._n = None

    def start(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        self._n = len(df)
        self.rows.append(
            dict(step=len(self.rows), rule=rule, n_before=None,
                 n_dropped=None, n_after=self._n)
        )
        return df

    def exclude(self, df: pd.DataFrame, keep_mask, rule: str) -> pd.DataFrame:
        if self._n is None:
            raise RuntimeError("Ledger.start() must be called before exclude()")
        n_before = len(df)
        if n_before != self._n:
            raise RuntimeError(
                f"{self.name}: ledger desync at '{rule}'. Ledger holds "
                f"{self._n} rows, frame holds {n_before}. A row left the "
                f"cohort without being recorded."
            )
        keep_mask = pd.Series(keep_mask).fillna(False).to_numpy(dtype=bool)
        out = df.loc[keep_mask].copy()
        n_after = len(out)
        n_dropped = n_before - n_after
        if n_before - n_dropped != n_after:  # arithmetic guard
            raise RuntimeError(f"{self.name}: exclusion arithmetic failed at '{rule}'")
        self.rows.append(
            dict(step=len(self.rows), rule=rule, n_before=n_before,
                 n_dropped=n_dropped, n_after=n_after)
        )
        self._n = n_after
        return out

    def note(self, rule: str, n: int) -> None:
        """Record a count that is reported but does not remove rows."""
        self.rows.append(
            dict(step=len(self.rows), rule=f"[note] {rule}", n_before=None,
                 n_dropped=None, n_after=n)
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def check(self) -> None:
        """Verify the flow balances: start - sum(dropped) == final."""
        f = self.frame()
        drops = f["n_dropped"].dropna()
        start = f.iloc[0]["n_after"]
        final = f["n_after"].iloc[-1]
        real = f[f["rule"].astype(str).str.startswith("[note]") == False]
        final = real["n_after"].iloc[-1]
        if start - drops.sum() != final:
            raise RuntimeError(
                f"{self.name}: cohort flow does not balance. "
                f"start={start} dropped={drops.sum()} final={final}"
            )

    def to_markdown(self) -> str:
        f = self.frame()
        lines = [f"**{self.name}**", "",
                 "| step | rule | n before | excluded | n after |",
                 "|---:|---|---:|---:|---:|"]
        for _, r in f.iterrows():
            nb = "" if pd.isna(r["n_before"]) else f"{int(r['n_before']):,}"
            nd = "" if pd.isna(r["n_dropped"]) else f"{int(r['n_dropped']):,}"
            lines.append(
                f"| {int(r['step'])} | {r['rule']} | {nb} | {nd} | {int(r['n_after']):,} |"
            )
        return "\n".join(lines)


# ── Result collection ─────────────────────────────────────────────────────────
class Results:
    """Flat key/value store persisted to output/results.json.

    Every number quoted in AUDIT_REPORT.md is written here first, so the report
    is generated from computed values rather than transcribed by hand.
    """

    def __init__(self, path: Path = None):
        self.path = Path(path) if path else OUTPUT / "results.json"
        self.data: dict = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def put(self, key: str, value) -> None:
        self.data[key] = _jsonable(value)
        self.flush()

    def update(self, mapping: dict) -> None:
        for k, v in mapping.items():
            self.data[k] = _jsonable(v)
        self.flush()

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def flush(self) -> None:
        self.path.write_text(
            json.dumps(self.data, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )


def _jsonable(v):
    import numpy as np
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, pd.DataFrame):
        return v.to_dict(orient="records")
    if isinstance(v, pd.Series):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return v


def banner(text: str) -> None:
    print("\n" + "=" * 74)
    print("  " + text)
    print("=" * 74)
