import nbformat as nbf
nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
code = lambda s: c.append(nbf.v4.new_code_cell(s))

md("""# Byte Beads — Bebras (distal) and Day 2 proximal analysis

Step-by-step exploratory data analysis followed by statistics for a computational-thinking
intervention at Southwest Middle School (n=30, grades 7-8, one intact class, **no control group**).

**Design.** Students played the Byte Beads game between a Bebras pretest and posttest.
- **Distal** Bebras CT test at three times: pre (Apr 30), post (May 7), delayed (May 14).
- **Proximal** in-game Day 2 task (May 5): Part A objective, Parts B/C open + coded.

**Read the limitations cell before interpreting anything.**""")

code("""import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import load
import stats as st

sns.set_theme(style="whitegrid")
FIG = Path.cwd().parent / "outputs" / "figures"
TAB = Path.cwd().parent / "outputs" / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

data = load.load_json()
totals = load.load_distal_totals(data)
long = load.load_distal_long(data)
prox = load.load_proximal(data)
mrg = load.merged(data)
print("students:", data["n_students"])""")

md("""## Limitations (fixed — do not interpret results without these)

1. **Identical pre/post items.** Q1-Q5 are the *same questions* pre and post, so the pre-post
   gain blends real learning with retest/practice effects. The novel Q6 probe is the cleaner
   transfer signal.
2. **Delayed parallel form is not validated as equivalent.** Delayed accuracy collapses on
   Q1/Q3/Q4 (floor) while Q2 stays 100% and Q5 rises — a form-difficulty artifact, **not**
   forgetting. Delayed comparisons are reported for completeness only.
3. **No control group, n=30, single class.** Findings are descriptive/exploratory, not causal.""")

md("## 1. EDA")
md("### 1.1 Sample counts and missingness")
code("""valid = long.groupby("phase", observed=True)["is_valid_response"].agg(["sum", "count"])
valid["n_absent_items"] = valid["count"] - valid["sum"]
# students with a usable total per phase
for col in ["pre_total", "post_total", "del_total"]:
    print(col, "valid students:", totals[col].notna().sum())
valid""")

md("### 1.2 Distribution of pre / post / delayed totals")
code("""fig, ax = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
for a, col, title in zip(ax, ["pre_total", "post_total", "del_total"], ["Pre", "Post", "Delayed"]):
    sns.histplot(totals[col].dropna(), bins=range(0, 7), ax=a, discrete=True)
    a.set_title(f"{title}  (mean={totals[col].mean():.2f})"); a.set_xlabel("score /5")
plt.tight_layout(); plt.savefig(FIG / "total_distributions.png", dpi=120); plt.show()
totals[["pre_total","post_total","del_total"]].describe().round(2)""")

md("### 1.3 Item-level % correct by phase (watch the delayed floor effect)")
code("""acc = (long.groupby(["phase", "item"], observed=True)
            .apply(lambda g: g.loc[g.is_valid_response, "correct"].mean(), include_groups=False)
            .unstack("phase"))[["pre", "post", "delayed"]]
acc_pct = (acc * 100).round(0)
acc_pct.to_csv(TAB / "item_accuracy_by_phase.csv")

ax = acc_pct.plot(kind="bar", figsize=(8, 4))
ax.set_ylabel("% correct (valid responses)"); ax.set_title("Item accuracy by phase")
ax.axhline(50, color="grey", lw=0.7, ls="--")
plt.tight_layout(); plt.savefig(FIG / "item_accuracy_by_phase.png", dpi=120); plt.show()
acc_pct""")

md("""The delayed bars for Q1/Q3/Q4 sit near the floor while Q2 stays at ceiling — the signature
of a non-equivalent parallel form, consistent with limitation 2.""")

md("### 1.4 Novel probe (Q6) pass rates")
code("""probe = pd.DataFrame({
    "post_q6": [totals["post_q6_novel"].mean()],
    "del_q6":  [totals["del_q6_novel"].mean()],
    "n_post":  [totals["post_q6_novel"].notna().sum()],
    "n_del":   [totals["del_q6_novel"].notna().sum()],
})
probe.round(2)""")

md("### 1.5 Proximal (Day 2) distributions and coding frequencies")
code("""fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
for a, col in zip(ax, ["partA_total", "partB_cs_quality", "partC_valid_total"]):
    sns.histplot(prox[col].dropna(), ax=a, discrete=True); a.set_title(col)
plt.tight_layout(); plt.savefig(FIG / "proximal_distributions.png", dpi=120); plt.show()

codes = pd.concat([prox["b1_cs_idea_code"], prox["b2_cs_idea_code"]]).dropna()
print("cs_idea_code frequency:\\n", codes.value_counts())
prox[["partA_total","partB_cs_quality","partC_valid_total"]].describe().round(2)""")

md("## 2. Statistics")
md("""### 2.1 Reliability of the 5-item scale (KR-20)

The pre-test items form the scale; KR-20 tells us how internally consistent the 5 binary items are
before we trust totals. A 5-item test rarely exceeds ~0.7, so read this as a floor caveat, not a verdict.""")
code("""pre_items = (long[long.phase == "pre"]
             .pivot(index="student_id", columns="item", values="correct"))
print("KR-20 (pre, complete cases):", round(st.kr20(pre_items), 3))""")

md("""### 2.2 Primary test — Pre → Post total (Q1-Q5)

Same students measured twice, bounded discrete 0-5 scores → **paired t-test** with **Wilcoxon
signed-rank** as the distribution-free check, plus **Cohen's dz** and a 95% CI. This is the main
learning measure (with the retest caveat from limitation 1).""")
code("""r = st.paired_change(totals["pre_total"], totals["post_total"])
print(f"n={r['n']}  mean diff={r['mean_diff']:+.2f}  95% CI [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]")
print(f"t={r['t']:.2f}  p(t)={r['p_ttest']:.3f}  p(Wilcoxon)={r['p_wilcoxon']:.3f}  Cohen dz={r['cohens_dz']:.2f}")""")

md("### 2.3 Delayed comparisons (reported for completeness — see limitation 2)")
code("""for label, a, b in [("Pre->Delayed", "pre_total", "del_total"),
                    ("Post->Delayed", "post_total", "del_total")]:
    r = st.paired_change(totals[a], totals[b])
    print(f"{label}: n={r['n']} diff={r['mean_diff']:+.2f} p(t)={r['p_ttest']:.4f} dz={r['cohens_dz']:.2f}")
print("\\nThese negative 'effects' reflect a harder, non-equivalent delayed form, not memory loss.")""")

md("""### 2.4 Item-level Pre → Post — McNemar exact test

Each item is binary and paired, so **McNemar's exact test** on the discordant pairs is the correct
per-item test. Family of 5 → **Holm-Bonferroni** correction.""")
code("""wide = long.pivot_table(index="student_id", columns=["phase", "item"],
                        values="correct", observed=True)
rows, pvals = [], {}
for item in load.ITEMS:
    m = st.mcnemar(wide[("pre", item)], wide[("post", item)])
    rows.append({"item": item, **m}); pvals[item] = m["p_exact"]
adj = st.holm(pvals)
mc = pd.DataFrame(rows).set_index("item")
mc["p_holm"] = pd.Series(adj)
mc.round(3)""")

md("""### 2.5 Proximal → distal transfer — Spearman

The strongest test of transfer that isn't contaminated by identical items: does higher in-game
(proximal) performance predict a larger Bebras gain? Small n and ordinal measures → **Spearman**.""")
code("""fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for a, col in zip(ax, ["partA_total", "partB_cs_quality", "partC_valid_total"]):
    r = st.spearman(mrg[col], mrg["gain_post_pre"])
    sns.regplot(x=mrg[col], y=mrg["gain_post_pre"], ax=a,
                scatter_kws={"alpha": 0.6}, line_kws={"color": "crimson"})
    a.set_title(f"{col}\\nrho={r['rho']:.2f}, p={r['p']:.3f}, n={r['n']}")
    a.set_ylabel("gain (post-pre)")
plt.tight_layout(); plt.savefig(FIG / "proximal_vs_gain.png", dpi=120); plt.show()""")

md("## 3. Results summary")
code("""summary = pd.DataFrame([
    {"comparison": "Pre->Post total", **{k: st.paired_change(totals.pre_total, totals.post_total)[k]
        for k in ["n", "mean_diff", "p_ttest", "cohens_dz"]}},
    {"comparison": "Pre->Delayed total", **{k: st.paired_change(totals.pre_total, totals.del_total)[k]
        for k in ["n", "mean_diff", "p_ttest", "cohens_dz"]}},
    {"comparison": "Post->Delayed total", **{k: st.paired_change(totals.post_total, totals.del_total)[k]
        for k in ["n", "mean_diff", "p_ttest", "cohens_dz"]}},
]).round(3)
summary.to_csv(TAB / "results_summary.csv", index=False)
summary""")

md("""### Interpretation

- **Pre → Post:** a small-to-medium, statistically detectable gain on the distal Bebras total,
  concentrated in the sequencing / pattern-rule items. Part of it is retest effect (identical items),
  so the novel probe and the proximal-transfer correlation carry the real transfer argument.
- **Delayed:** uninterpretable as retention — the parallel form is not equivalent. Validate or
  rescore the delayed form before making any durability claim.
- **Transfer:** the proximal→gain correlations are the cleanest evidence; interpret with the small n.

All limitations from the top cell stand.""")

nb["cells"] = c
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
with open("notebooks/01_eda_and_stats.ipynb", "w") as f:
    nbf.write(nb, f)
print("notebook written")
