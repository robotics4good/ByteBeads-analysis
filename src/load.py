"""Data loading for the Byte Beads CT study.

Reads data/byte_beads_data.json and returns tidy pandas structures.

Scoring codes in the source:
    "absent"      -> student absent / test not administered
    "no_response" -> item left blank
    null          -> not applicable
Both "absent" and "no_response" are treated as NOT correct, but an
`is_valid_response` flag is preserved so accuracy can be computed over
attempted items only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ITEMS = ["q1", "q2", "q3", "q4", "q5"]
PHASES = ["pre", "post", "delayed"]
_NONRESPONSE = {"absent", "no_response"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: str | Path | None = None) -> dict:
    path = Path(path) if path else _repo_root() / "data" / "byte_beads_data.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _is_correct(v) -> int:
    """1 only if the item was scored 1; everything else (0, absent, blank) is 0."""
    return 1 if v == 1 else 0


def _is_valid(v) -> bool:
    """True if the student actually attempted the item."""
    if v is None:
        return False
    if isinstance(v, str) and v in _NONRESPONSE:
        return False
    return True


def load_distal_long(data: dict | None = None) -> pd.DataFrame:
    """Long-format item table: one row per student x phase x item."""
    data = data or load_json()
    rows = []
    for s in data["students"]:
        d = s["distal"]
        for phase in PHASES:
            for item in ITEMS:
                raw = d[phase][item]
                rows.append(
                    {
                        "student_id": s["student_id"],
                        "grade": s["grade"],
                        "phase": phase,
                        "item": item,
                        "correct": _is_correct(raw),
                        "is_valid_response": _is_valid(raw),
                        "raw": raw,
                    }
                )
    df = pd.DataFrame(rows)
    df["phase"] = pd.Categorical(df["phase"], categories=PHASES, ordered=True)
    return df


def load_distal_totals(data: dict | None = None) -> pd.DataFrame:
    """Wide table: one row per student with totals, gains, and novel probes."""
    data = data or load_json()
    rows = []
    for s in data["students"]:
        d = s["distal"]
        rows.append(
            {
                "student_id": s["student_id"],
                "grade": s["grade"],
                "pre_total": d["pre"]["total"],
                "post_total": d["post"]["total"],
                "del_total": d["delayed"]["total"],
                "post_q6_novel": _novel(d["post"].get("q6_novel")),
                "del_q6_novel": _novel(d["delayed"].get("q6_novel")),
                "gain_post_pre": d["gains"]["post_pre"],
                "retention_del_post": d["gains"]["del_post_retention"],
                "gain_del_pre": d["gains"]["del_pre"],
            }
        )
    return pd.DataFrame(rows)


def _novel(v):
    """Novel probe -> 1/0, or NaN if not attempted."""
    if v == 1:
        return 1.0
    if v in (0,):
        return 0.0
    if _is_valid(v):
        return float(v == 1)
    return np.nan


def load_proximal(data: dict | None = None) -> pd.DataFrame:
    """Wide table of Day 2 proximal measures + coded fields."""
    data = data or load_json()
    rows = []
    for s in data["students"]:
        p = s["proximal_day2"]
        a, b, c = p["part_a"], p["part_b"], p["part_c"]
        rows.append(
            {
                "student_id": s["student_id"],
                "grade": s["grade"],
                "partA_total": a.get("partA_total"),
                "partB_cs_quality": b.get("cs_quality"),
                "partB_cards_logged": b.get("cards_logged"),
                "partC_valid_total": c.get("valid_total"),
                "b1_cs_idea_code": b["card1"].get("cs_idea_code"),
                "b1_cs_link": b["card1"].get("cs_link"),
                "b2_cs_idea_code": b["card2"].get("cs_idea_code"),
                "b2_cs_link": b["card2"].get("cs_link"),
                "c1_section": c["spinner1"].get("section"),
                "c1_length_change": c["spinner1"].get("length_change"),
            }
        )
    return pd.DataFrame(rows)


def merged(data: dict | None = None) -> pd.DataFrame:
    """Distal totals joined to proximal measures on student_id."""
    data = data or load_json()
    return load_distal_totals(data).merge(
        load_proximal(data).drop(columns=["grade"]), on="student_id", how="left"
    )


if __name__ == "__main__":
    d = load_json()
    print("students:", d["n_students"])
    print(load_distal_totals(d).describe())
