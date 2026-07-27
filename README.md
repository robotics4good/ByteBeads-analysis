# ByteBeads-analysis

Analysis of a computational-thinking (CT) intervention at Southwest Middle School.
Students (n=30, grades 7-8, one intact class, **no control group**) played the Byte Beads
game between a Bebras pretest and posttest.

## Data
- **Distal** — Bebras CT assessment (5 scored items) at three times: pre (Apr 30),
  post (May 7, Q1-Q5 identical to pre + novel Q6), delayed (May 14, parallel form + novel Q6).
- **Proximal** — in-game Day 2 task (May 5): Part A objective, Parts B/C open + coded.

All data live in `data/byte_beads_data.json` (30 students, nested per student). Score codes:
`"absent"` = not administered, `"no_response"` = blank, `null` = N/A; neither counts as correct.

## Layout
```
data/     byte_beads_data.json
src/      load.py (JSON -> tidy DataFrames), stats.py (test helpers)
notebooks/01_eda_and_stats.ipynb   step-by-step EDA then statistics
outputs/  figures/ and tables/ written by the notebook
```

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/01_eda_and_stats.ipynb   # or: run all cells
```

## Key measures
- `gain_post_pre = post_total - pre_total` (Q1-Q5) — primary learning measure.
- Three **separate** proximal measures: `partA_total`, `partB_cs_quality`, `partC_valid_total`.

## Limitations (carry into any writeup)
1. Q1-Q5 identical pre->post: gains partly reflect retest/practice. The novel Q6 probe is the
   cleaner transfer signal.
2. The delayed parallel form is **not** validated as equivalent; delayed scores collapse on
   Q1/Q3/Q4 (floor) — a form artifact, not forgetting. Do not read delayed as retention.
3. n=30, single class, no control group: descriptive/exploratory, not causal.
