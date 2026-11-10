"""Generates notebooks/03_visualizations_with_AI.ipynb.

Usage (from the repo root):  python notebooks/make_nb03.py notebooks/03_visualizations_with_AI.ipynb
The notebook is written without outputs; run it afterwards (Jupyter, VS Code or Colab) to create the plots.
"""
import json
import sys
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


# ----------------------------------------------------------------------------------------------------------------
md(r"""
# VIS4AI · Notebook 03: Where small models fail, and what the reasoning really does
*Visualizations of the AI-aided pair (see [`AI_USAGE_LOG.md`](../AI_USAGE_LOG.md)).*

Notebook 02 compares the two models **on average** (accuracy, tokens, correlations). This notebook looks at
**single problems and inside the outputs** of the same run (`run_4`, 50 GSM8K problems):

| Model | What it is |
|---|---|
| **Qwen2.5-Math-1.5B** | math *base* model: short textbook-style solutions, often with Python code |
| **R1-Distill-1.5B** | the *same* model, fine-tuned on reasoning traces of DeepSeek-R1: thinks at length in `<think> … </think>` first |

Because R1-Distill **is** Qwen2.5-Math plus reasoning training, every difference below is an effect of that training.

**Story line**
1. **Who fails where?** The failures are nested: R1 solves everything Qwen solves, plus 3 more.
2. **Why do they fail?** Every failure, labelled by type. Reasoning fixes *calculation* errors, not *understanding*
   errors. Repetition maps show how the cut-off answers get stuck.
3. **Case studies:** the exact words where it goes wrong.
4. **Inside R1's thinking:** when does the correct answer appear, and what happens afterwards?
5. **Experiment A, early answering:** stop R1's thinking early and force an answer. How much thinking is needed, how
   sure is R1 along the way, and in which layer of the model does the answer appear?
6. **Experiment B, rewording:** are the shared mistakes caused by the wording of the questions?
7. **Summary**

### How to run
* **Colab:** `Runtime → Run all`. Results are read from (and the experiments saved to) Google Drive `VIS4AI/results/<RUN>`.
* **Locally:** put the run folder in `data/` at the repo root (e.g. `data/run_4/`).
* Sections 1–4 only need the saved results. Sections 5–6 need a GPU (Colab T4, or Apple Silicon) **only if their
  result files `nb03_*.jsonl` are not in the run folder yet**; otherwise they are just loaded.
""")

# ----------------------------------------------------------------------------------------------------------------
md("## 0. Setup and loading the results")

code(r"""
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RUN = "run_4_2026-11-08_15-40"   # the run analysed in Notebook 02

try:                               # on Colab: read from Google Drive
    from google.colab import drive
    drive.mount("/content/drive")
    RESULTS_DIR = Path("/content/drive/MyDrive/VIS4AI/results") / RUN
except ImportError:                # locally: read from data/ in the repo
    data_dir = Path.cwd().parent / "data" if Path.cwd().name == "notebooks" else Path.cwd() / "data"
    RESULTS_DIR = next(p for p in [data_dir / RUN, data_dir / RUN.split("_2")[0]] if p.exists())
print("Reading results from:", RESULTS_DIR)

R1, QWEN = "R1-Distill-1.5B", "Qwen2.5-Math-1.5B"
MODELS = {"r1-distill": R1, "qwen-math": QWEN}   # file name -> short label

# Colours: outcome = status colours, always paired with a symbol so colour is never the only cue
COLORS  = {"correct": "#0ca30c", "wrong": "#d03b3b", "cut off": "#a8a79e", "no answer": "#e1e0d9"}
SYMBOLS = {"correct": "✓", "wrong": "✗", "cut off": "…", "no answer": "–"}
INK, INK_MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e1e0d9"
BLUE, BLUE_LIGHT = "#2a78d6", "#b9d3f3"              # R1's thinking: before / after the answer first appears
MARK_KEY, MARK_BAD, MARK_GOOD = "#fde3a7", "#f6c7c7", "#c9ecc9"   # text highlights

plt.rcParams.update({"text.parse_math": False,       # questions contain "$" signs
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED, "xtick.color": INK_MUTED,
                     "ytick.color": INK_MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})
""")

code(r"""
def outcome(row):
    # Same categories as the comparison table in Notebook 01
    if not row["finished"]:
        return "cut off"
    if row["answer"] is None or pd.isna(row["answer"]):
        return "no answer"
    return "correct" if row["correct"] else "wrong"


def thinking(text):
    # R1's thinking = everything before </think> (all of it if it was cut off while thinking)
    return text.split("</think>")[0]


results = []
for key, label in MODELS.items():
    df = pd.read_json(RESULTS_DIR / f"{key}.jsonl", lines=True, dtype={"gold": str, "answer": str})
    df["model"] = label
    df["outcome"] = df.apply(outcome, axis=1)
    results.append(df)
results = pd.concat(results, ignore_index=True)

# One row per problem, one column per model
outcomes = results.pivot(index="id", columns="model", values="outcome")[[R1, QWEN]]
answers  = results.pivot(index="id", columns="model", values="answer")[[R1, QWEN]]
outputs  = results.pivot(index="id", columns="model", values="output")[[R1, QWEN]]
tokens   = results.pivot(index="id", columns="model", values="n_tokens")[[R1, QWEN]]
problems = results.drop_duplicates("id").set_index("id")[["question", "gold"]]

results.groupby("model")["outcome"].value_counts().unstack(fill_value=0)
""")

# ----------------------------------------------------------------------------------------------------------------
md(r"""
## 1. Who fails where?
The table counts the problems by the outcome of **both** models at once (rows: R1-Distill, columns: Qwen2.5-Math).
""")

code(r"""
# How sure is "R1 is better"? Only the problems that exactly one model solves carry information (exact sign test).
only_r1 = int(((outcomes[R1] == "correct") & (outcomes[QWEN] != "correct")).sum())
only_qwen = int(((outcomes[QWEN] == "correct") & (outcomes[R1] != "correct")).sum())
n_diff = only_r1 + only_qwen
p_sign = min(1.0, 2 * sum(math.comb(n_diff, k) for k in range(min(only_r1, only_qwen) + 1)) / 2 ** n_diff)
print(f"Solved by R1 only: {only_r1}, by Qwen only: {only_qwen}  ->  exact sign test p = {p_sign:.2f}")

pd.crosstab(outcomes[R1], outcomes[QWEN], margins=True, margins_name="total")
""")

md(r"""
Each column below is one problem, sorted into groups by what happened to the two models.
`✓` correct, `✗` wrong, `…` cut off at the token limit.
""")

code(r"""
def group(row):
    ok_r1, ok_qwen = row[R1] == "correct", row[QWEN] == "correct"
    if ok_r1 and ok_qwen:
        return "both correct"
    if ok_r1:
        return f"only {R1} correct"
    if ok_qwen:
        return f"only {QWEN} correct"
    return "both failed"


GROUP_ORDER = ["both correct", f"only {R1} correct", f"only {QWEN} correct", "both failed"]
groups = outcomes.apply(group, axis=1)
order = sorted(outcomes.index, key=lambda i: (GROUP_ORDER.index(groups[i]), i))

fig, ax = plt.subplots(figsize=(15, 2.6))
for x, pid in enumerate(order):
    for y, model in enumerate([R1, QWEN]):
        o = outcomes.loc[pid, model]
        ax.add_patch(plt.Rectangle((x + 0.06, y + 0.06), 0.88, 0.88, color=COLORS[o]))   # gap between cells
        ax.text(x + 0.5, y + 0.5, SYMBOLS[o], ha="center", va="center", fontsize=9,
                color=INK if o in ("cut off", "no answer") else "white")

x = 0
for g in GROUP_ORDER:                      # group labels and separators above the grid
    n = int((groups == g).sum())
    if n == 0:
        continue
    label = g.replace(R1, "R1").replace(QWEN, "Qwen").replace(" correct", "\ncorrect").replace("both failed", "both\nfailed")
    ax.text(x + n / 2, -0.15, f"{label} ({n})", ha="center", va="bottom", fontsize=8.5, color=INK_MUTED)
    if x > 0:
        ax.axvline(x, color=INK, lw=1)
    x += n

ax.set_xlim(0, len(order))
ax.set_ylim(2, -1.1)
ax.set_yticks([0.5, 1.5], [R1, QWEN], color=INK)
ax.set_xticks(np.arange(len(order)) + 0.5, order, fontsize=7)
ax.set_xlabel("GSM8K problem id")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.legend(handles=[Patch(color=c, label=f"{SYMBOLS[o]} {o}") for o, c in COLORS.items() if (outcomes == o).any().any()],
          loc="upper left", bbox_to_anchor=(1.005, 1), frameon=False)
ax.set_title("Outcome per problem: failures are nested, Qwen never solves a problem R1 misses",
             loc="left", fontsize=12, color=INK)
plt.tight_layout()
plt.show()
""")

md(r"""
**Takeaway:** 40 problems are easy for both. R1 adds 3 problems on top, and loses none. The remaining 7 are hard for
*both* models, so the hard problems are a property of the questions, not of one model.

*How sure is this?* The nested pattern is clear, but "R1 is better" rests on 3 problems. With 3 against 0 the exact sign
test gives p = 0.25, so on 50 problems the difference in accuracy is **not statistically secured**. This notebook therefore
describes *how* the two models differ, not by how much.
""")

# ----------------------------------------------------------------------------------------------------------------
md(r"""
## 2. Why do they fail? Every failure, labelled by type
We read the full output of each of the 17 failures (10 Qwen, 7 R1) and gave it one type, from *understanding* the
question (top rows) to *carrying out* the solution (bottom rows). The short evidence for each label is in the table below the plot.

*The labels were proposed by the AI (Claude) after reading every output and **must be checked by the team**
(open an output with the cell at the end of this section).*
""")

code(r"""
FAILURE_TYPES = ["misread a number", "misunderstood the story", "ambiguous question",
                 "made-up code output", "went in circles (cut off)", "repeated itself (cut off)"]

FAILURES = [  # (model, problem id, type, evidence)
    (R1,   3,  "misread a number",          "restates the question as '3 sprints a week': 3 × 60 = 180"),
    (QWEN, 3,  "misread a number",          "sprints_per_week = 3: 3 × 60 = 180"),
    (QWEN, 4,  "misread a number",          "takes 3 cups per chicken per *meal* (not per day): 180 − 40 = 140"),
    (R1,   7,  "misunderstood the story",   "reads 40% as 'halfway' and adds a second 20-min restart: 100 + 20 + 20"),
    (QWEN, 7,  "misunderstood the story",   "plans a 'second restart': 140"),
    (QWEN, 8,  "misunderstood the story",   "adds the drive back home to the distance away: 180 + 15 + 120 = 315"),
    (R1,   12, "ambiguous question",        "finds that year 13 is the first year with profit, but answers 12 (break-even)"),
    (QWEN, 12, "ambiguous question",        "break-even 90 / 7.5 = 12, while the gold answer counts the first year with profit"),
    (R1,   37, "ambiguous question",        "the question says he sells *all* his toys, so it answers 0 sets left"),
    (QWEN, 11, "made-up code output",       "correct code, but writes the output 716 (true result 694)"),
    (QWEN, 17, "made-up code output",       "correct code, but writes the output 52500 (true result 57500)"),
    (QWEN, 26, "made-up code output",       "correct code, but writes the output 234.0 (true result 243, digits swapped)"),
    (R1,   4,  "went in circles (cut off)", "unsure if 3 cups are per meal or per day, keeps re-deciding until 8192 tokens"),
    (R1,   8,  "went in circles (cut off)", "tries 20, 115, … miles and re-reads the question until 8192 tokens"),
    (R1,   21, "went in circles (cut off)", "mixes up whose son it is and keeps finding contradictions until 8192 tokens"),
    (QWEN, 21, "repeated itself (cut off)", "sets up a contradicting equation and restarts the same steps until 2048 tokens"),
    (QWEN, 37, "repeated itself (cut off)", "writes the same sentence over and over until 2048 tokens"),
]
failures = pd.DataFrame(FAILURES, columns=["model", "id", "type", "evidence"])

# Every failure of the run must have exactly one label
labelled = set(zip(failures["model"], failures["id"]))
actual = {(m, i) for i in outcomes.index for m in [R1, QWEN] if outcomes.loc[i, m] != "correct"}
assert labelled == actual, f"missing: {actual - labelled}, not a failure: {labelled - actual}"
""")

code(r"""
fig, ax = plt.subplots(figsize=(13, 4.6))
COL_X = {R1: 0.0, QWEN: 1.0}            # each model gets one unit of width; chips are placed left to right
CHIP_W, CHIP_H, STEP = 0.2, 0.64, 0.225

for row, ftype in enumerate(FAILURE_TYPES):
    if row % 2 == 0:   # zebra band to guide the eye along a row
        ax.add_patch(plt.Rectangle((-0.05, row - 0.5), 2.1, 1, color="#f3f2ee", zorder=0))
    for model, x0 in COL_X.items():
        other = QWEN if model == R1 else R1
        ids = failures[(failures.type == ftype) & (failures.model == model)]["id"].tolist()
        # problems the OTHER model failed with this type, but this model solved: hollow ✓ chip
        solved = [i for i in failures[(failures.type == ftype) & (failures.model == other)]["id"]
                  if outcomes.loc[i, model] == "correct"]
        for k, pid in enumerate(ids + solved):
            x = x0 + k * STEP
            if pid in ids:
                o = outcomes.loc[pid, model]
                ax.add_patch(plt.Rectangle((x, row - CHIP_H / 2), CHIP_W, CHIP_H, color=COLORS[o]))
                ax.text(x + CHIP_W / 2, row, f"{SYMBOLS[o]} #{pid}", ha="center", va="center", fontsize=10,
                        color="white" if o == "wrong" else INK)
            else:
                ax.add_patch(plt.Rectangle((x + 0.006, row - CHIP_H / 2 + 0.03), CHIP_W - 0.012, CHIP_H - 0.06,
                                           fc=SURFACE, ec=COLORS["correct"], lw=2))
                ax.text(x + CHIP_W / 2, row, f"✓ #{pid}", ha="center", va="center", fontsize=10, color=INK)

ax.set_yticks(range(len(FAILURE_TYPES)), FAILURE_TYPES, fontsize=10.5, color=INK)
ax.set_ylim(len(FAILURE_TYPES) - 0.5, -1.0)
ax.set_xlim(-0.05, 2.05)
for model, x0 in COL_X.items():
    n = (failures.model == model).sum()
    ax.text(x0, -0.85, f"{model}  ({n} failures)", fontsize=11, weight="bold", color=INK, va="center")
ax.set_xticks([])
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.legend(handles=[Patch(color=COLORS["wrong"], label="✗ wrong answer"),
                   Patch(color=COLORS["cut off"], label="… cut off at the token limit"),
                   Patch(fc=SURFACE, ec=COLORS["correct"], lw=1.6, label="✓ solved by this model (the other one failed)")],
          loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
ax.set_title("Why the models fail: R1's 3 extra wins are exactly Qwen's 3 made-up code outputs",
             loc="left", fontsize=12, color=INK, pad=10)
plt.tight_layout()
plt.show()
""")

code(r"""
pd.set_option("display.max_colwidth", 120)
failures.assign(gold=failures["id"].map(problems["gold"]),
                answer=[answers.loc[i, m] for m, i in zip(failures["model"], failures["id"])]
                ).sort_values(["type", "id"], key=lambda s: s.map(FAILURE_TYPES.index) if s.name == "type" else s
                ).set_index(["type", "id", "model"])[["gold", "answer", "evidence"]]
""")

code(r"""
# Read one full output to check a label (change the two values)
MODEL, PROBLEM_ID = QWEN, 11
print(f"#{PROBLEM_ID}:", problems.loc[PROBLEM_ID, "question"])
print(f"\ncorrect: {problems.loc[PROBLEM_ID, 'gold']} | {MODEL}: {answers.loc[PROBLEM_ID, MODEL]} "
      f"({outcomes.loc[PROBLEM_ID, MODEL]}, {tokens.loc[PROBLEM_ID, MODEL]} tokens)\n")
print(outputs.loc[PROBLEM_ID, MODEL])
""")

md(r"""
**Takeaways**
* The **understanding** errors (top three rows) are shared: whenever R1 misreads a question, Qwen does too, mostly
  with the same wrong answer. This fits the idea that R1 keeps these errors from its base model and that the reasoning
  training did not fix them, but one run cannot prove where they come from.
* The **carrying-out** errors differ. Qwen writes correct Python code but then *imagines* its output, and gets the
  arithmetic wrong (it never runs the code). R1 does the arithmetic step by step in text, and solves exactly these 3.
* R1's own typical failure is **going in circles**: it doubts its reading of the question, re-decides, and runs out of
  tokens. Qwen instead repeats the same text word for word. Section 2.1 measures this difference.
""")

md(r"""
### 2.1 Checking two labels with a measurement: repetition maps
"Went in circles" and "repeated itself" are judgements. A **repetition map** measures them: the output is split into
sentences, and both axes are the sentence number. A dot at (x, y) means *sentence x says the same as sentence y*
(dark = word for word, light = at least 80% of the words are the same).

How to read it: the diagonal is every sentence compared with itself. **Lines parallel to the diagonal are loops**: a whole
passage is written again. An empty map means that the text never repeats.
""")

code(r"""
def sentences(text):
    # Sentences (or lines) of an output, as sets of lower-case words; very short pieces are dropped
    parts = [s.strip().lower() for s in re.split(r"(?<=[.!?\n])\s+", text)]
    return [(s, frozenset(re.findall(r"\w+", s))) for s in parts if len(s) > 15]


def repetition_matrix(text):
    # 0 = different, 1 = at least 80% of the words shared (Jaccard), 2 = identical sentence
    sents = sentences(text)
    n = len(sents)
    sim = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i):
            (a, wa), (b, wb) = sents[i], sents[j]
            if a == b:
                sim[i, j] = sim[j, i] = 2
            elif len(wa & wb) >= 0.8 * len(wa | wb):
                sim[i, j] = sim[j, i] = 1
    return sim


rep = []
for pid in problems.index:
    for model in [R1, QWEN]:
        sim = repetition_matrix(outputs.loc[pid, model])
        earlier = np.tril(sim, -1)          # compare each sentence only with the sentences before it
        rep.append(dict(model=model, id=pid, outcome=outcomes.loc[pid, model], n_sentences=len(sim),
                        exact=(earlier.max(axis=1) == 2).mean() if len(sim) else 0.0,
                        repeated=(earlier.max(axis=1) >= 1).mean() if len(sim) else 0.0))
rep = pd.DataFrame(rep).set_index(["model", "id"])

MAPS = [(R1, tokens[R1][outcomes[R1] == "correct"].idxmax(), "correct, its longest output"),
        (R1, 4, "cut off, went in circles"), (R1, 8, "cut off, went in circles"),
        (R1, 21, "cut off, went in circles"), (QWEN, 21, "cut off, repeated itself"),
        (QWEN, 37, "cut off, repeated itself")]
fig, axes = plt.subplots(2, 3, figsize=(13, 9.2))
for ax, (model, pid, label) in zip(axes.flat, MAPS):
    sim = repetition_matrix(outputs.loc[pid, model])
    n = len(sim)
    size = float(np.clip((230 / n) ** 2, 3, 30))                         # dots stay visible in long outputs
    for level, color in [(1, BLUE_LIGHT), (2, BLUE)]:
        ys, xs = np.nonzero(sim == level)
        ax.scatter(xs + 0.5, ys + 0.5, s=size, marker="s", color=color, lw=0)
    ax.plot([0, n], [0, n], color=INK_MUTED, lw=0.8)                     # the diagonal
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    r = rep.loc[(model, pid)]
    o = outcomes.loc[pid, model]
    ax.set_title(f"{model} #{pid}: {label}\n{r.repeated:.0%} of {n} sentences repeat ({r.exact:.0%} word for word)",
                 loc="left", fontsize=9.5, color=INK)
    ax.set_xlabel("sentence number", fontsize=9)
    if ax in axes[:, 0]:
        ax.set_ylabel("sentence number", fontsize=9)
    ax.tick_params(length=2, labelsize=8)
fig.legend(handles=[Patch(color=BLUE, label="same sentence, word for word"),
                    Patch(color=BLUE_LIGHT, label="nearly the same (at least 80% of the words)")],
           loc="lower center", ncol=2, frameon=False)
fig.suptitle("Repetition maps: Qwen loops word for word, R1 comes back to the same statements in scattered places",
             x=0.01, ha="left", fontsize=12.5, color=INK)
plt.tight_layout(rect=(0, 0.035, 1, 0.96), h_pad=2.5, w_pad=2)
plt.show()
""")

md(r"""
**What the patterns look like in the text.** Below are three excerpts from the outputs in the maps, numbered with the
same sentence numbers as the map axes. `= #k` marks a sentence that is word for word the same as the earlier sentence k.
* **Qwen #37 (checkerboard):** two sentences, A and B, alternate until the token limit.
* **Qwen #21 (diagonal stripes):** a whole block (set up an equation, find a contradiction, "re-evaluate") is written
  again and again; only the variable changes once, from x to y.
* **R1 #4 (scattered dots):** R1 keeps coming back to the same doubt (3 cups per day or per meal?), but in new words
  each time, so almost no sentence is an exact copy.
""")

code(r"""
def numbered_sentences(text):
    # Same split as repetition_matrix, but keeping the original wording (the numbers match the map axes)
    parts = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text)]
    return [s for s in parts if len(s) > 15]


def show_excerpt(model, pid, numbers, title):
    sents = numbered_sentences(outputs.loc[pid, model])
    first_seen = {}
    for k, s in enumerate(sents):
        first_seen.setdefault(s.lower(), k)
    print(f"── {model} #{pid}: {title}  ({len(sents)} sentences in total)")
    last = None
    for k in numbers:
        if last is not None and k > last + 1:
            print("      …")
        earlier = first_seen[sents[k].lower()]
        mark = f"= #{earlier}" if earlier < k else ""
        print(f"{k:>4} {mark:>7}  {sents[k][:150]}")
        last = k
    print()


show_excerpt(QWEN, 37, range(9, 18), "falls into a loop of two sentences")
show_excerpt(QWEN, 21, list(range(11, 24)) + list(range(29, 40)), "writes the same block again and again")
r1_doubt = [k for k, s in enumerate(numbered_sentences(outputs.loc[4, R1])) if "1 cup per chicken" in s][:8]
show_excerpt(R1, 4, r1_doubt, "the same doubt about the 3 cups, in new words each time")
""")

md(r"""
The same measure for **all 100 outputs**: the share of sentences that repeat an earlier sentence, one dot per output.
""")

code(r"""
rng = np.random.default_rng(0)
ROWS = [(m, o) for m in [R1, QWEN] for o in ["correct", "wrong", "cut off"]]
fig, ax = plt.subplots(figsize=(13, 3.8))
ylabels = []
for k, (model, o) in enumerate(ROWS):
    d = rep.loc[model][rep.loc[model].outcome == o]
    ylabels.append(f"{model} · {SYMBOLS[o]} {o} ({len(d)})")
    jitter = rng.uniform(-0.2, 0.2, len(d)) if o == "correct" else np.zeros(len(d))   # only the crowded rows
    ax.scatter(d.repeated, k + jitter, s=44, color=COLORS[o], ec="white", lw=0.8, zorder=3)
    if o != "correct":   # name the failures; two close neighbours get their labels to the left and to the right
        named = d[d.repeated >= 0.1].sort_values("repeated")
        xs = named.repeated.tolist()
        for j, pid in enumerate(named.index):
            if j + 1 < len(xs) and xs[j + 1] - xs[j] < 0.035:
                offset, ha = (-7, -3), "right"
            elif j > 0 and xs[j] - xs[j - 1] < 0.035:
                offset, ha = (7, -3), "left"
            else:
                offset, ha = (0, 9), "center"
            ax.annotate(f"#{pid}", (xs[j], k), textcoords="offset points", xytext=offset, ha=ha, fontsize=8.5,
                        color=INK_MUTED)
ax.axhline(2.5, color=GRID, lw=1)
ax.set_yticks(range(len(ylabels)), ylabels, color=INK)
ax.set_ylim(len(ylabels) - 0.5, -0.6)
ax.set_xlim(-0.02, 1)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("share of sentences that repeat an earlier sentence (word for word or nearly)")
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
top = rep.reset_index().groupby(["model", "outcome"]).repeated.agg(["min", "max"])
ax.set_title(f"Qwen's cut-off outputs are mostly repetition ({top.loc[(QWEN, 'cut off'), 'min']:.0%} and more); "
             f"R1's cut-offs: {top.loc[(R1, 'cut off'), 'min']:.0%} to {top.loc[(R1, 'cut off'), 'max']:.0%}, "
             f"its correct answers: at most {top.loc[(R1, 'correct'), 'max']:.0%}", loc="left", fontsize=11, color=INK)
plt.tight_layout()
plt.show()

rep.reset_index().groupby(["model", "outcome"]).repeated.agg(["count", "min", "median", "max"]).round(2)
""")

md(r"""
**Takeaways (2.1)**
* The measurement supports both labels, and shows that they are two different things. Qwen's cut-off outputs are
  **loops**: about three quarters of the sentences are exact copies, in long lines parallel to the diagonal. Once it is
  in the loop, it never leaves.
* R1's cut-off outputs repeat much less, and mostly not word for word. It comes back to the same statements **in
  scattered places** (single dots and short lines), which is what "going in circles" looks like. Even its longest
  *correct* output (#30) shows blocks: R1 re-computes the same lines when it checks its work.
* **As a warning sign, repetition works for Qwen, but only weakly for R1.** A Qwen run that repeats most of its
  sentences could be stopped early. For R1 the cut-off outputs repeat more than every correct one, but the gap is small
  and two of its wrong answers (#7, #37) repeat about as much as the lowest cut-off.
""")

# ----------------------------------------------------------------------------------------------------------------
md(r"""
## 3. Case studies: the exact words where it goes wrong
For five problems: the question with the **key phrase** highlighted, and a quote from each model.
Red = where the model goes wrong, green = where it is right. Every quote is checked to appear in the saved output.
""")

code(r"""
CASES = [
    dict(id=3, title="Misread a number: both drop “3 times”",
         key=["3 sprints 3 times a week"],
         quotes={R1: ("First, the problem says he runs 3 sprints a week.", [("3 sprints a week", MARK_BAD)]),
                 QWEN: ("sprints_per_week = 3", [("sprints_per_week = 3", MARK_BAD)])}),
    dict(id=7, title="Misunderstood the story: both invent a second restart",
         key=["40% of the way through", "restart the download from the beginning"],
         quotes={R1: ("But halfway through the download, Windows forces her to restart the process because of some updates.",
                      [("halfway through", MARK_BAD)]),
                 QWEN: ("4. Add the time for the second restart.", [("second restart", MARK_BAD)])}),
    dict(id=12, title="Ambiguous question: R1 finds the gold answer 13 and still answers 12",
         key=["before he starts earning money"],
         quotes={R1: ("So, n=13 is the first integer where Total cash flow is positive.", [("n=13", MARK_GOOD)]),
                 QWEN: ("Divide the initial cost by the annual net profit.",
                        [("Divide the initial cost by the annual net profit", MARK_BAD)])}),
    dict(id=37, title="Ambiguous question: “sell all his toys”, but how many are left?",
         key=["plans to sell all his toys", "How many lego sets does he still have?"],
         quotes={R1: ("Wait, but the problem says he has 13 lego sets, so if he sells all of them, he has 0 left.",
                      [("if he sells all of them, he has 0 left", MARK_BAD)]),
                 QWEN: ("This means he must have sold all his lego sets and had $30 left before buying the video games.",
                        [("This means he must have sold all his lego sets and had $30 left before buying the video games.", MARK_BAD)])}),
    dict(id=11, title="Made-up code output: Qwen writes correct code, then invents its result",
         key=[],
         quotes={R1: ("$204 (donuts) + $160 (cupcakes) + $330 (cheesecakes)", [("$204", MARK_GOOD), ("$160", MARK_GOOD), ("$330", MARK_GOOD)]),
                 QWEN: ("```output\n716", [("716", MARK_BAD)])}),
]


def segments(text, marks):
    # Split text into (piece, highlight colour or None) using the marked substrings
    spans = sorted((text.index(s), text.index(s) + len(s), c) for s, c in marks)
    out, pos = [], 0
    for a, b, c in spans:
        out += [(text[pos:a], None), (text[a:b], c)]
        pos = b
    return out + [(text[pos:], None)]


def draw_rich_text(ax, x, y, parts, width, fontsize=10.5, color=INK):
    # Draw word-wrapped text with highlighted words, in axes coordinates. Returns the y of the next line.
    renderer = ax.figure.canvas.get_renderer()
    to_axes = ax.transAxes.inverted()
    line = fontsize * 1.75 / 72 / (ax.get_window_extent().height / ax.figure.dpi)

    def width_of(t):
        bb = t.get_window_extent(renderer)
        return to_axes.transform((bb.x1, 0))[0] - to_axes.transform((bb.x0, 0))[0]

    probe = ax.text(0, 0, "a", fontsize=fontsize)
    space = width_of(probe) * 0.55
    probe.remove()
    # words with a flag "glued": no white space before it (e.g. "." right after a highlighted phrase)
    words, prev_ends_with_space = [], True
    for text, bg in parts:
        for m in re.finditer(r"\S+", text):
            glued = m.start() == 0 and not prev_ends_with_space and bool(words)
            words.append((m.group(0), bg, glued))
        if text:
            prev_ends_with_space = text[-1].isspace()
    cx, cy = x, y
    for word, bg, glued in words:
        if glued:
            cx -= space
        t = ax.text(cx, cy, word, transform=ax.transAxes, fontsize=fontsize, va="top", color=color,
                    bbox=dict(boxstyle="square,pad=0.15", fc=bg, ec="none") if bg else None)
        w = width_of(t)
        if cx + w > x + width and cx > x:
            cx, cy = x, cy - line
            t.set_position((cx, cy))
        cx += w + space
    return cy - line


def plot_case(case):
    pid = case["id"]
    question = problems.loc[pid, "question"]
    for s in case["key"]:
        assert s in question, s
    fig, ax = plt.subplots(figsize=(13, 2.9))
    ax.axis("off")
    fig.canvas.draw()
    ax.text(0, 1, f"#{pid}  {case['title']}", transform=ax.transAxes, fontsize=12.5, weight="bold", color=INK, va="top")
    ax.text(0, 0.86, "question", transform=ax.transAxes, fontsize=10, color=INK_MUTED, va="top")
    y = draw_rich_text(ax, 0.17, 0.86, segments(question, [(s, MARK_KEY) for s in case["key"]]), 0.83) - 0.04
    for model in [R1, QWEN]:
        quote, marks = case["quotes"][model]
        assert quote in outputs.loc[pid, model], f"quote not found in {model} output for #{pid}: {quote}"
        o = outcomes.loc[pid, model]
        ans = answers.loc[pid, model] if o in ("correct", "wrong") else o
        ax.text(0, y, f"{model}\n{SYMBOLS[o]} {ans}", transform=ax.transAxes, fontsize=10, color=INK_MUTED, va="top")
        repeats = outputs.loc[pid, model].count(quote)
        shown = quote.replace("```output\n", "output: ") + (f"   (written {repeats}× in a row)" if repeats > 3 else "")
        y = draw_rich_text(ax, 0.17, y, segments(shown, marks), 0.83) - 0.07
    ax.text(0, y, f"correct answer: {problems.loc[pid, 'gold']}", transform=ax.transAxes, fontsize=10, color=INK,
            va="top", weight="bold")
    plt.show()


for case in CASES:
    plot_case(case)
""")

md(r"""
**Takeaways**
* In #3 and #7 the mistake is made **in the first sentence**, when the model restates the question, before any
  calculation. Everything after that is correct maths on the wrong problem, and R1's long thinking never questions it.
* #12 and #37 are arguably **badly worded questions** (#37 even contradicts itself). R1 actually considers the gold
  answer in #12, but picks the other reading.
* #11 shows the difference in *carrying out*: Qwen pretends to run code, R1 calculates in text.

Experiment B (section 6) tests the first two points: if we reword these questions clearly, do the models get them right?
""")

# ----------------------------------------------------------------------------------------------------------------
md(r"""
## 4. Inside R1's thinking
R1 writes ~1,400 tokens of thinking per problem (median) against ~350 for Qwen. **What is all that text for?**

We search the thinking for the moments where the **correct answer is stated as a result** (e.g. `= 18`, `is 18`,
`total of 18`, not just any mention of the number), and for **hesitation words** (`wait`, `hmm`, `let me check`, …).
Positions are given in tokens (estimated from the character position).
""")

code(r"""
HESITATION = re.compile(r"\b(wait|hmm|let me (?:double[- ])?check|let me verify|but let me)\b", re.IGNORECASE)


def stated_as_result(think, value):
    # Character positions where `value` is stated as the result of a step (commas removed from the text)
    pattern = r"(?:=|\bis|\bbe|\bget|\btotal of|\bequals)\s*\\?\$?\s*" + re.escape(value) + r"(?![\d]|\.\d)"
    return [m.end() for m in re.finditer(pattern, think)]


def number_str(answer):
    # '\\$1,234' -> '1234', '234.0' -> '234'; None if not a number
    if answer is None or (not isinstance(answer, str) and pd.isna(answer)):
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(answer).replace(",", ""))
    if not nums:
        return None
    v = float(nums[-1])
    return str(int(v)) if v.is_integer() else str(v)


rows = []
for pid in problems.index:
    text = outputs.loc[pid, R1]
    think = thinking(text).replace(",", "")
    think_tokens = tokens.loc[pid, R1] * len(thinking(text)) / len(text)
    to_tok = lambda char_pos: char_pos / len(think) * think_tokens
    gold_pos = stated_as_result(think, problems.loc[pid, "gold"])
    final = number_str(answers.loc[pid, R1])
    wrong_pos = stated_as_result(think, final) if final and final != problems.loc[pid, "gold"] else []
    # If the correct number also appears in the question (e.g. "20 chickens"), it can be "stated as a result"
    # by coincidence ("20 chickens * 1 cup = 20 cups"), so its first position is not reliable.
    in_question = re.search(r"(?<![\d.])" + re.escape(problems.loc[pid, "gold"]) + r"(?![\d]|\.\d)",
                            problems.loc[pid, "question"].replace(",", "")) is not None
    rows.append(dict(id=pid, outcome=outcomes.loc[pid, R1], think_tokens=think_tokens, gold_in_question=in_question,
                     gold_first=to_tok(gold_pos[0]) if gold_pos else np.nan, gold_count=len(gold_pos),
                     wrong_first=to_tok(wrong_pos[0]) if wrong_pos else np.nan,
                     hesitations=[to_tok(m.start()) for m in HESITATION.finditer(think)]))
r1_think = pd.DataFrame(rows).set_index("id")
r1_think["share_after"] = 1 - r1_think["gold_first"] / r1_think["think_tokens"]
print("Correct number also in the question (position unreliable, left out of the statistics):",
      [f"#{i}" for i in r1_think.index[r1_think.gold_in_question]])

ok = r1_think[(r1_think.outcome == "correct") & ~r1_think.gold_in_question]
print(f"Correct answers ({len(ok)}): the correct result is first stated after a median of "
      f"{(1 - ok.share_after).median():.0%} of the thinking ({ok.gold_first.median():.0f} tokens); "
      f"the remaining {ok.share_after.median():.0%} ({(ok.think_tokens - ok.gold_first).median():.0f} tokens) come after it.")
print(f"It is stated as a result {ok.gold_count.median():.0f} times (median) before R1 stops thinking.")
""")

md(r"""
### 4.1 Answer timeline
One bar per problem = R1's thinking, from the first token (left) to `</think>` (right end).
**Dark blue** = before the correct result first appears, **light blue** = after it (checking, repeating, doubting).
A grey bar means the correct result never appears.
""")

code(r"""
OUTCOME_ORDER = ["correct", "wrong", "cut off"]
data = r1_think.copy()
data["rank"] = data["outcome"].map(OUTCOME_ORDER.index)
data = data.sort_values(["rank", "think_tokens"])

fig, ax = plt.subplots(figsize=(13, 12))
y, yticks, ylabels, group_y = 0, [], [], {}
for o in OUTCOME_ORDER:
    part = data[data.outcome == o]
    group_y[o] = (y, y + len(part) - 1, len(part))
    for pid, r in part.iterrows():
        h = 0.72
        if np.isnan(r.gold_first):
            ax.barh(y, r.think_tokens, height=h, color=COLORS["cut off"])
        else:
            ax.barh(y, r.gold_first, height=h, color=BLUE)
            ax.barh(y, r.think_tokens - r.gold_first, left=r.gold_first, height=h, color=BLUE_LIGHT)
            if r.gold_in_question:   # hollow marker: position may be a coincidence
                ax.plot(r.gold_first, y, marker="o", ms=6.5, color=SURFACE, mec=COLORS["correct"], mew=1.8, zorder=3)
            else:
                ax.plot(r.gold_first, y, marker="o", ms=6.5, color=COLORS["correct"], mec="white", mew=1.2, zorder=3)
        if not np.isnan(r.wrong_first):
            ax.plot(r.wrong_first, y, marker="X", ms=8, color=COLORS["wrong"], mec="white", mew=1.0, zorder=3)
        yticks.append(y)
        ylabels.append(f"#{pid}" + ("*" if r.gold_in_question else ""))
        y += 1
    y += 1.2   # gap between groups

LIMIT = 8192
ax.axvline(LIMIT, color=INK_MUTED, lw=1, ls="--")
ax.text(LIMIT, -1.3, "token limit", ha="center", fontsize=9, color=INK_MUTED)
for o, (a, b, n) in group_y.items():
    ax.text(LIMIT * 1.03, (a + b) / 2, f"{SYMBOLS[o]} {o}\n({n})", va="center", fontsize=10.5, color=INK, weight="bold")
ax.text(0, y + 0.6, "* the correct number also appears in the question, so its first position (hollow dot) may be a "
        "coincidence; left out of the statistics", fontsize=8.5, color=INK_MUTED, va="top")

ax.set_yticks(yticks, ylabels, fontsize=7.5)
ax.set_ylim(y + 1.2, -2)
ax.set_xlim(0, LIMIT * 1.16)
ax.set_xlabel("position in R1's thinking (tokens)")
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.legend(handles=[Patch(color=BLUE, label="thinking before the correct result first appears"),
                   Patch(color=BLUE_LIGHT, label="thinking after it"),
                   Patch(color=COLORS["cut off"], label="correct result never appears"),
                   Line2D([], [], marker="o", ls="", color=COLORS["correct"], mec="white", ms=7, label="correct result first stated"),
                   Line2D([], [], marker="X", ls="", color=COLORS["wrong"], mec="white", ms=8, label="wrong final answer first stated")],
          loc="upper right", bbox_to_anchor=(0.86, 0.97), frameon=True, facecolor=SURFACE, edgecolor=GRID)
share = 1 - ok.share_after.median()
ax.set_title(f"R1 usually has the correct result after {share:.0%} of its thinking; the rest is spent after it",
             loc="left", fontsize=12.5, color=INK)
plt.tight_layout()
plt.show()
""")

md(r"""
### 4.2 Value lanes: what does R1 compute, and how often?
Every number R1 writes after an `=` sign, at its position in the thinking. Each **row is one value** (the 8 most
frequent, plus the correct answer and R1's final answer); each dot is one time R1 computed it.
A row of dots across the whole bar = R1 computes the same value **again and again**.
""")

code(r"""
RESULT = re.compile(r"=\s*\\?\$?\s*(-?\d+(?:\.\d+)?)")


def computed_values(pid):
    # (position in tokens, value as string) for every number written after "=" in R1's thinking
    text = outputs.loc[pid, R1]
    think = thinking(text).replace(",", "")
    think_tokens = tokens.loc[pid, R1] * len(thinking(text)) / len(text)
    return [(m.start() / len(think) * think_tokens, number_str(m.group(1))) for m in RESULT.finditer(think)], think_tokens


LANE_IDS = [(ok.think_tokens.idxmax(), "correct: the longest correct thinking"),
            (12, "wrong: switches between 12 and 13"),
            (7, "wrong: never computes the correct 160"),
            (8, "cut off: tries many distances, then runs out")]

fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
for ax, (pid, subtitle) in zip(axes.flat, LANE_IDS):
    values, think_tokens = computed_values(pid)
    gold, final = problems.loc[pid, "gold"], number_str(answers.loc[pid, R1])
    counts = pd.Series([v for _, v in values]).value_counts()
    lanes = list(counts.index[:8])
    for must in [gold, final]:
        if must and must not in lanes:
            lanes.append(must)
    lanes = sorted(lanes, key=float, reverse=True)
    ax.barh(range(len(lanes)), think_tokens, height=0.82, color="#f3f2ee", zorder=0)   # one track per value
    for k, v in enumerate(lanes):
        xs = [p for p, val in values if val == v]
        color = COLORS["correct"] if v == gold else COLORS["wrong"] if v == final else COLORS["cut off"]
        ax.scatter(xs, [k] * len(xs), s=34, color=color, ec="white", lw=0.7, zorder=3,
                   marker="X" if (v == final and v != gold) else "o")
    labels = [v + ("  correct" if v == gold else "  its answer" if v == final else "") for v in lanes]
    ax.set_yticks(range(len(lanes)), labels, fontsize=9)
    for t, v in zip(ax.get_yticklabels(), lanes):
        t.set_color(INK if v in (gold, final) else INK_MUTED)
        t.set_weight("bold" if v in (gold, final) else "normal")
    ax.set_ylim(len(lanes) - 0.5, -0.5)
    ax.set_xlim(0, max(think_tokens, 100) * 1.02)
    ax.set_title(f"#{pid}  {subtitle}", loc="left", fontsize=10.5, color=INK)
    ax.set_xlabel("position in R1's thinking (tokens)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=COLORS["correct"], mec="white", ms=7, label="correct answer"),
                    Line2D([], [], marker="X", ls="", color=COLORS["wrong"], mec="white", ms=8, label="R1's wrong final answer"),
                    Line2D([], [], marker="o", ls="", color=COLORS["cut off"], mec="white", ms=7, label="other value")],
           loc="upper right", bbox_to_anchor=(0.99, 0.995), ncol=3, frameon=False)
fig.suptitle("Value lanes: R1 re-computes the same values over and over, also when it is right",
             x=0.01, ha="left", fontsize=12.5, color=INK)
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()
""")

md(r"""
### 4.3 Hesitation: does doubting help?
Hesitation words (`wait`, `hmm`, `let me check`, …) per 1,000 tokens against the length of the output, one dot per
problem (R1 only: Qwen almost never uses these words, see the line printed below the plot).
""")

code(r"""
hes = []
for pid in problems.index:
    for model in [R1, QWEN]:
        text = outputs.loc[pid, model]
        hes.append(dict(model=model, id=pid, outcome=outcomes.loc[pid, model], n_tokens=tokens.loc[pid, model],
                        count=len(HESITATION.findall(text)),
                        per_1k=len(HESITATION.findall(text)) / tokens.loc[pid, model] * 1000))
hes = pd.DataFrame(hes)
h1 = hes[hes.model == R1].set_index("id")
rho_all = h1.n_tokens.rank().corr(h1.per_1k.rank())                      # Spearman rank correlation
h_ok = h1[h1.outcome == "correct"]
rho_ok = h_ok.n_tokens.rank().corr(h_ok.per_1k.rank())

MARKERS = {"correct": "o", "wrong": "X", "cut off": "s"}
fig, ax = plt.subplots(figsize=(13, 5.2))
for o in OUTCOME_ORDER:
    d = h1[h1.outcome == o]
    ax.scatter(d.n_tokens, d.per_1k, s=70 if o != "correct" else 46, marker=MARKERS[o], color=COLORS[o], ec="white",
               lw=0.9, zorder=3 if o == "correct" else 4, label=f"{SYMBOLS[o]} {o} ({len(d)})")
    if o != "correct":   # name the failures (problems at the same spot share one label)
        for (x, y), ids in d.groupby(["n_tokens", d.per_1k.round(1)]).groups.items():
            ax.annotate(" ".join(f"#{i}" for i in ids), (x, y), textcoords="offset points", xytext=(0, 9), ha="center",
                        fontsize=8.5, color=INK)
ax.set_xscale("log")
ax.set_xticks([500, 1000, 2000, 4000, 8192], ["500", "1,000", "2,000", "4,000", "8,192\n(limit)"])
ax.minorticks_off()
ax.set_xlim(450, 9500)
ax.set_ylim(0, None)
ax.set_xlabel("length of R1's output (tokens, log scale)")
ax.set_ylabel("hesitation words per 1,000 tokens")
ax.grid(color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False)
ax.set_title(f"The longer R1 thinks, the more it hesitates (rank correlation {rho_all:.2f}); "
             "its failures are simply among the long ones", loc="left", fontsize=12, color=INK)
plt.tight_layout()
plt.show()

q = hes[hes.model == QWEN]
print(f"Rank correlation length vs hesitation rate, R1: all 50 problems {rho_all:.2f}, correct answers only {rho_ok:.2f}")
print(f"Qwen: hesitation words in {(q['count'] > 0).sum()} of {len(q)} outputs ({q['count'].sum()} words in total; "
      f"R1: {h1['count'].sum()})")
h1.groupby("outcome").per_1k.agg(["count", "median"]).round(1)
""")

md(r"""
**Takeaways (section 4)**
* For correct answers, R1 states the correct result early and then **keeps going for most of its thinking**:
  re-checking, re-deriving, and repeating the same result several times.
* When R1 fails, the correct result sometimes *does* appear (#12, #8), but R1 does not settle on it; the value lanes
  show it **switching between a few candidates** instead of converging.
* **Doubt grows with length, and does not single out the failures.** R1 hesitates more per 1,000 tokens the longer it
  thinks, also among the correct answers. Its failures have a high rate (median 7.0 wrong, 8.8 cut off, against 5.2
  correct), but they are 7 problems and all of them are long, so **this run cannot separate "fails" from "thinks long"**.
  What can be said: more doubt does not go together with better answers.

This is descriptive, so it cannot tell us if the "checking" part is useful. Experiment A tests that directly.
""")

# ----------------------------------------------------------------------------------------------------------------
md(r"""
## 5. Experiment A: early answering, how much of the thinking does R1 need?
We take R1's saved thinking for each of the 50 problems, **cut it off** after 0%, 5%, 10%, … 100% of its tokens
(21 cut points), close the thinking and **force an answer** right away:

`<question> <first x% of the thinking> </think> The final answer is \boxed{` → the model completes the number.

At every cut point we record three things:
1. the **forced answer** (greedy: always the most likely next token), and if it is correct;
2. the **probability of the correct answer**: how likely the model is to write exactly the correct number here
   (its digits, with or without a thousands comma, followed by something that is not another digit);
3. a **layer read-out** for section 5.3: how likely the first digit of the correct answer is when we read the model's
   state after each of its 28 layers.

If the forced answer is already correct at 50%, the second half of the thinking was not needed for *this* answer.
(Method after Lanham et al., 2023, *Measuring Faithfulness in Chain-of-Thought Reasoning*.)

Results are saved to `nb03_early_answering_fine.jsonl` in the run folder; if the file is complete, nothing is re-run.
""")

code(r"""
import gc

try:
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList,
                              set_seed)
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
except ImportError:
    torch, DEVICE = None, None
print("Device for the experiments:", DEVICE or "none (saved experiment results are loaded, nothing is re-run)")

R1_NAME, QWEN_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen2.5-Math-1.5B"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
SEED = 42


def load_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to(DEVICE).eval()
    return model, tok


def free_memory():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()


def build_prompt(tok, question, chat_template):
    text = question + "\n" + INSTRUCTION
    if not chat_template:
        return text + "\n"
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


def boxed_answers(text):
    found = []
    for m in re.finditer(r"\\boxed\{", text):
        depth = 1
        for j in range(m.end(), len(text)):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                if text[m.end():j].strip():
                    found.append(text[m.end():j].strip())
                break
    return found


def is_correct(answer, gold):
    value = number_str(answer)
    return value is not None and math.isclose(float(value), float(gold), abs_tol=1e-6)
""")

code(r"""
FRACTIONS = [round(i / 20, 2) for i in range(21)]     # 0%, 5%, ..., 100% of the thinking
EARLY_PATH = RESULTS_DIR / "nb03_early_answering_fine.jsonl"
RUN_EARLY_ANSWERING = True     # set to False to never start the model (only load saved results)
FORCE = "\n</think>\n\nThe final answer is \\boxed{"

done = {(r["id"], r["fraction"]) for r in load_jsonl(EARLY_PATH)}
todo = [pid for pid in problems.index if any((int(pid), f) not in done for f in FRACTIONS)]
print(f"Early answering: {len(problems) - len(todo)}/{len(problems)} problems done")

if todo and RUN_EARLY_ANSWERING and DEVICE:
    model, tok = load_model(R1_NAME)
    force = tok(FORCE, add_special_tokens=False).input_ids
    vocab = [tok.decode([i]) for i in range(len(tok))]
    more_digits = torch.zeros(model.config.vocab_size, dtype=torch.bool)    # tokens that would continue a number
    more_digits[:len(vocab)] = torch.tensor([bool(re.match(r"[0-9,]", v)) for v in vocab])
    more_digits = more_digits.to(DEVICE)

    def read(ids, cache, all_layers=False):
        # Run the model on the new tokens only. `cache` holds everything it has read before, so each long thinking
        # is read once and re-used for all cut points (instead of 21 times).
        out = model.model(torch.tensor([ids], device=DEVICE), past_key_values=cache, use_cache=True,
                          output_hidden_states=all_layers)
        return out, out.past_key_values

    def log_probs(hidden):
        return torch.log_softmax(model.lm_head(hidden).float(), dim=-1)

    with torch.no_grad():
        for n, pid in enumerate(todo):
            gold = problems.loc[pid, "gold"]
            forms = [gold] + ([f"{int(gold):,}"] if gold.isdigit() and int(gold) >= 1000 else [])   # 70000 or 70,000
            first_digit = tok(gold, add_special_tokens=False).input_ids[0]
            prompt = tok(build_prompt(tok, problems.loc[pid, "question"], True), add_special_tokens=False).input_ids
            think = tok(thinking(outputs.loc[pid, R1]), add_special_tokens=False).input_ids
            _, cache = read(prompt, None)
            fed, rows = 0, []
            for f in FRACTIONS:
                kept = round(f * len(think))
                if kept > fed:                                   # read the next piece of the thinking
                    _, cache = read(think[fed:kept], cache)
                    fed = kept
                if (int(pid), f) in done:
                    continue
                out, cache = read(force, cache, all_layers=True)
                first = log_probs(out.last_hidden_state[0, -1])   # what the model would write first, after "\boxed{"

                # (3) layer read-out: turn the state after every layer into next-token probabilities ("logit lens")
                states = torch.stack([h[0, -1] for h in out.hidden_states[:-1]])
                per_layer = torch.softmax(model.lm_head(model.model.norm(states)).float(), dim=-1)[:, first_digit]
                layer_p = [round(p, 4) for p in per_layer.tolist()] + [round(first[first_digit].exp().item(), 4)]

                # (2) probability of the correct answer: feed its digits and multiply their probabilities
                p_correct = 0.0
                for form in forms:
                    digits = tok(form, add_special_tokens=False).input_ids
                    o, cache = read(digits, cache)
                    lp = log_probs(o.last_hidden_state[0])
                    logp = first[digits[0]].item() + sum(lp[j - 1, t].item() for j, t in enumerate(digits) if j > 0)
                    p_ends = 1 - lp[-1].exp()[more_digits].sum().item()      # ... and the number ends here
                    p_correct += math.exp(logp) * p_ends
                    cache.crop(-len(digits))                                 # forget the digits again

                # (1) greedy forced answer: write tokens until the closing brace
                written, nxt, n_fed = [], int(first.argmax()), 0
                for _ in range(12):
                    written.append(nxt)
                    if "}" in tok.decode(written):
                        break
                    o, cache = read([nxt], cache)
                    n_fed += 1
                    nxt = int(model.lm_head(o.last_hidden_state[0, -1]).argmax())
                cache.crop(-(len(force) + n_fed))                            # back to "prompt + thinking so far"
                answer = tok.decode(written).split("}")[0].strip()
                rows.append(dict(id=int(pid), fraction=f, kept_tokens=kept, think_tokens=len(think), answer=answer,
                                 correct=is_correct(answer, gold), kept_chars=len(tok.decode(think[:kept])),
                                 p_correct=round(p_correct, 4), layer_p=layer_p))
            append_jsonl(EARLY_PATH, rows)
            del cache
            free_memory()
            print(f"  {n + 1}/{len(todo)}  #{pid}: " + "".join("✓" if r["correct"] else "✗" for r in rows))
    del model, tok
    free_memory()

early = pd.DataFrame(load_jsonl(EARLY_PATH))
if len(early):
    display(early[early.fraction.isin([0, 0.25, 0.5, 0.75, 1])].groupby("fraction")
            .agg(correct=("correct", "sum"), problems=("correct", "count"), accuracy=("correct", "mean"),
                 mean_probability=("p_correct", "mean")).round(2))
""")

md(r"""
### 5.1 Accuracy and confidence along the thinking
**Left:** the share of the 50 problems with a correct forced answer, at each cut point.
**Right:** one row per problem; the colour is the **probability of the correct answer** at each cut point (white = 0%,
dark blue = 100%). The ring marks where the correct result is first written in the thinking (section 4.1).
""")

code(r"""
if len(early) == 0:
    print("No early-answering results yet: run the cell above on a GPU.")
else:
    piv = early.pivot(index="id", columns="fraction", values="correct").astype(bool)
    prob = early.pivot(index="id", columns="fraction", values="p_correct")

    def locks_in(row):
        # first cut point from which the forced answer stays correct until the end (NaN = never)
        for f in FRACTIONS:
            if row.loc[f:].all():
                return f
        return np.nan

    lock = piv.apply(locks_in, axis=1)
    order_e = sorted(piv.index, key=lambda i: (OUTCOME_ORDER.index(outcomes.loc[i, R1]),
                                               2 if np.isnan(lock[i]) else lock[i], i))
    PROB_CMAP = plt.matplotlib.colors.LinearSegmentedColormap.from_list("prob", ["#ffffff", BLUE_LIGHT, BLUE, "#123a6b"])

    fig = plt.figure(figsize=(13, 10.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.22)

    # Left: accuracy curve
    ax = fig.add_subplot(gs[0])
    acc = piv.mean()
    ax.plot(FRACTIONS, acc.values, color=BLUE, lw=2, marker="o", ms=5.5, mec="white", mew=1, zorder=3)
    for f in [0, 0.25, 0.5, 0.75, 1.0]:
        ax.annotate(f"{acc[f]:.0%}", (f, acc[f]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=10,
                    color=INK)
    full = (outcomes[R1] == "correct").mean()
    ax.axhline(full, color=INK_MUTED, ls="--", lw=1)
    ax.text(0.02, full + 0.015, f"R1 answering normally: {full:.0%}", fontsize=9, color=INK_MUTED)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("share of the thinking kept before forcing the answer")
    ax.set_ylabel("forced answer correct (50 problems)")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Accuracy when the thinking is cut early", loc="left", fontsize=11.5, color=INK)

    # Right: probability of the correct answer, per problem
    ax = fig.add_subplot(gs[1])
    step = FRACTIONS[1] - FRACTIONS[0]
    im = ax.imshow(prob.loc[order_e, FRACTIONS].values, cmap=PROB_CMAP, vmin=0, vmax=1, aspect="auto",
                   extent=(-step / 2, 1 + step / 2, len(order_e), 0), interpolation="nearest")
    for yy in range(1, len(order_e)):
        ax.axhline(yy, color=SURFACE, lw=1.2)
    for yy, pid in enumerate(order_e):
        g = r1_think.loc[pid, "gold_first"] / r1_think.loc[pid, "think_tokens"]     # share of the characters
        if not np.isnan(g) and not r1_think.loc[pid, "gold_in_question"]:
            # the cut points are shares of the tokens: convert with the number of characters kept at each cut point
            kept_chars = early[early.id == pid].set_index("fraction").kept_chars.loc[FRACTIONS].values
            g = np.interp(g * kept_chars[-1], kept_chars, FRACTIONS)
            ax.plot(g, yy + 0.5, marker="o", ms=6, color="none", mec="#eb6834", mew=1.8, zorder=3)
    bounds = np.cumsum([0] + [int((outcomes.loc[order_e, R1] == o).sum()) for o in OUTCOME_ORDER])
    for o, a, b in zip(OUTCOME_ORDER, bounds, bounds[1:]):
        if b > a:
            ax.text(1.05, (a + b) / 2, f"{SYMBOLS[o]} {o}\nin the full run", va="center", fontsize=9.5, color=INK)
            if b < len(order_e):
                ax.axhline(b, color=INK, lw=1.2)
    ax.set_xlim(-step / 2, 1 + step / 2)
    ax.set_ylim(len(order_e), 0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticks(np.arange(len(order_e)) + 0.5, [f"#{i}" for i in order_e], fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("share of the thinking kept")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.025, pad=0.06, aspect=40)
    cb.set_ticks([0, 0.5, 1], labels=["0%", "50%", "100%"])
    cb.set_label("probability of the correct answer", fontsize=9, color=INK_MUTED)
    cb.outline.set_visible(False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="none", mec="#eb6834", mew=1.8, ms=7,
                              label="correct result first written in the thinking (section 4.1)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9, handletextpad=0.2, borderpad=0)
    ax.set_title("Per problem: how sure is R1 of the correct answer?", loc="left", fontsize=11.5, color=INK, pad=22)

    half = acc[0.5]
    fig.suptitle(f"Early answering: with half of its thinking R1 is already right {half:.0%} of the time "
                 f"(no thinking: {acc[0]:.0%})", x=0.01, ha="left", fontsize=12.5, color=INK)
    plt.show()

    sure = early.p_correct
    print(f"The probability is rarely in between: {((sure < 0.1) | (sure > 0.9)).mean():.0%} of the {len(early)} "
          f"measurements are below 10% or above 90%.")
    unsure = early[early.correct & (early.p_correct < 0.5)]
    print(f"Correct forced answers with a probability under 50% (correct, but not sure): {len(unsure)}, "
          f"of which {int((unsure.fraction <= 0.25).sum())} in the first quarter of the thinking. "
          f"Wrong forced answers with a probability of 50% or more: {int((~early.correct & (early.p_correct >= 0.5)).sum())}.")
    print(f"Forced answer correct from some point until the end: {lock.notna().sum()} of {len(lock)} problems.")
""")

md(r"""
### 5.2 Does R1 need to write the result before it can answer?
For each problem we compare two cut points:
* **locked in**: the first cut point from which the forced answer is correct and stays correct;
* **written**: the first cut point at which the correct number appears **anywhere** in the text R1 was shown.

If R1 only repeats what it has written, it locks in at the cut point where the number is written, or later.
(Left out: the 4 problems where the correct number is already in the question, and the problems that never lock in.)
""")

code(r"""
def contains_number(text, value):
    return re.search(r"(?<![\d.])" + re.escape(value) + r"(?![\d]|\.\d)", text.replace(",", "")) is not None


if len(early):
    chars = early.pivot(index="id", columns="fraction", values="kept_chars")
    rows = []
    for pid in problems.index:
        if r1_think.loc[pid, "gold_in_question"] or np.isnan(lock[pid]):
            continue
        think, gold = thinking(outputs.loc[pid, R1]), problems.loc[pid, "gold"]
        shown = {f: think[:chars.loc[pid, f]] for f in FRACTIONS}          # the text R1 was shown at each cut point
        written = next((f for f in FRACTIONS if contains_number(shown[f], gold)), np.nan)
        rows.append(dict(id=pid, outcome=outcomes.loc[pid, R1], locked=lock[pid], written=written,
                         p_at_lock=prob.loc[pid, lock[pid]], text_end="…" + shown[lock[pid]][-110:].replace("\n", " ")))
    ahead = pd.DataFrame(rows).set_index("id").dropna(subset=["written"])
    ahead["delay"] = (ahead.locked - ahead.written).round(2)                # < 0: locked in before it is written
    n_before, n_same, n_later = (ahead.delay < 0).sum(), (ahead.delay == 0).sum(), (ahead.delay > 0).sum()
    no_thinking = int((ahead.locked == 0).sum())

    step = FRACTIONS[1] - FRACTIONS[0]
    counts = ahead.delay.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(13, 4.6))
    GROUPS = [("locked in before the number is written", BLUE, counts[counts.index < 0]),
              ("at the cut point where it is first written", "#123a6b", counts[counts.index == 0]),
              ("only later", COLORS["cut off"], counts[counts.index > 0])]
    for label, color, part in GROUPS:
        ax.bar(part.index, part.values, width=step * 0.86, color=color, label=f"{label} ({int(part.sum())})")
        for x, v in part.items():
            ax.text(x, v + 0.15, str(v), ha="center", fontsize=9, color=INK)
    ax.axvline(-step / 2, color=INK_MUTED, lw=1, ls="--")
    ax.set_xticks(np.arange(-0.5, 0.41, 0.1))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else f"{v:+.0%}"))
    ax.set_xlabel("cut point where the answer locks in  minus  cut point where the number is first written  "
                  "(share of the thinking)")
    ax.set_ylabel("problems")
    ax.set_ylim(0, counts.max() + 1.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(f"R1 is ahead of its own text: in {n_before} of {len(ahead)} problems it gives the correct answer "
                 "before the number is written", loc="left", fontsize=12.5, color=INK)
    plt.tight_layout()
    plt.show()

    print(f"{len(ahead)} problems: locked in before the number is written {n_before} (of these {no_thinking} with no "
          f"thinking at all), at the same cut point {n_same}, later {n_later}. "
          f"Median: {-ahead.delay.median():.0%} of the thinking before it is written.")
    # The end of the text R1 had been shown when its answer locked in (the correct number is not in it yet)
    EXAMPLES = [0, 10, 11, 24]
    pd.set_option("display.max_colwidth", 130)
    display(ahead.loc[EXAMPLES, ["locked", "written", "text_end"]].assign(correct_answer=problems.loc[EXAMPLES, "gold"]))
""")

md(r"""
### 5.3 A look inside: in which layer does the answer appear?
R1 has 28 layers. Normally only the output of the last layer is turned into the next token. Here we read out the model's
state **after every layer** in the same way (a "logit lens") at the position right after `\boxed{`, and ask: how likely
is the **first digit of the correct answer**? The plot shows the average over the problems that R1 answers correctly with
its full thinking.

*Read it with care: only the first digit is checked, and the early layers are not made to be read out like this, so a
low value there means "not readable this way", not "not there".*
""")

code(r"""
if len(early):
    solved = early[(early.fraction == 1.0) & early.correct].id
    lens = early[early.id.isin(solved)]
    grid = np.array([np.mean(np.stack(lens[lens.fraction == f].layer_p.to_list()), axis=0) for f in FRACTIONS]).T
    n_layers = grid.shape[0] - 1                                   # row 0 = input embeddings, rows 1..28 = layers
    readable = int(np.argmax(grid[:, -1] >= 0.5))                  # first layer with 50% at the full thinking
    quiet = int(np.max(np.nonzero(grid.max(axis=1) < 0.01)[0]))    # up to this layer: below 1% at every cut point
    FIRST_SHOWN = 17
    step = FRACTIONS[1] - FRACTIONS[0]

    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(grid[FIRST_SHOWN:], cmap=PROB_CMAP, vmin=0, vmax=1, aspect="auto", origin="lower",
                   extent=(-step / 2, 1 + step / 2, FIRST_SHOWN - 0.5, n_layers + 0.5), interpolation="nearest")
    for yy in range(FIRST_SHOWN, n_layers):
        ax.axhline(yy + 0.5, color=SURFACE, lw=1.2)
    ax.axhline(readable - 0.5, color=INK, lw=1, ls="--")
    ax.text(0.0, readable - 0.62, f"from layer {readable} on, the first digit of the answer can be read out",
            fontsize=9.5, color=INK, va="top")
    ax.text(0.0, FIRST_SHOWN + 0.3, f"layers 1 to {quiet}: below 1% at every cut point (layers 1 to {FIRST_SHOWN - 1} are not shown)",
            fontsize=9.5, color=INK_MUTED, va="center")
    ax.set_yticks(range(FIRST_SHOWN, n_layers + 1))
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of the thinking kept")
    ax.set_ylabel(f"layer ({n_layers} = last)")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_ticks([0, 0.5, 1], labels=["0%", "50%", "100%"])
    cb.set_label("probability of the first digit of the correct answer", fontsize=9, color=INK_MUTED)
    cb.outline.set_visible(False)
    ax.set_title(f"The answer becomes readable only in the last {n_layers - readable + 1} of {n_layers} layers, "
                 f"however much thinking is kept (average over {len(solved)} problems)", loc="left", fontsize=12.5,
                 color=INK)
    plt.tight_layout()
    plt.show()
""")

md(r"""
**Takeaways (section 5)**
* **The thinking does real work.** Forced to answer with no thinking at all, R1 is right on only 6 of 50 problems (12%),
  and on 3 of these 6 its probability for the correct answer is below 50%. It cannot solve these problems "in its head".
* **R1 needs about three quarters of its thinking.** Accuracy rises step by step from 12% to 90% at three quarters of the
  thinking (36% at one quarter, 74% at half). **The last quarter changes no answer**: from 75% on, the same 45 problems
  are correct at every cut point.
* **R1 does not become sure slowly, it flips.** In 90% of the 1,050 measurements the probability of the correct answer
  is below 10% or above 90%. In the per-problem plot each row turns from white to dark blue within one or two cut points
  and then stays dark (45 of 50 problems lock in).
* **R1 is ahead of its own text.** In 26 of 41 problems the answer locks in **before the correct number is written
  anywhere** in the text R1 was shown (3 of them need no thinking at all), in 9 at the cut point where it is written,
  and in only 6 later. The table shows what happens: once the intermediate results are on the page, R1 does the
  remaining step "in its head". So the thinking is needed to get there, but writing down the final result is not, and
  what comes after it (60% of the thinking, section 4) is checking that does not change the answer.
* **R1 sometimes knows the answer and does not stop.** With its full thinking the forced answer is right on 45 problems,
  2 more than in the normal run: in #8 R1 is sure of the correct 45 from 55% of its thinking on, but kept going and was
  cut off. (#4 is correct almost from the start because its answer, 20, is the number of chickens in the question.)
* **Inside the model, the answer appears late.** The first digit of the answer can be read out only in the last 5 of
  the 28 layers. More thinking does not make it appear in an earlier layer, it only makes it more certain.

*These runs were made on an Apple M3 GPU (MPS). The forced answers are greedy, so the six cut points of an earlier,
coarser run (0, 10, 25, 50, 75, 100%) were reproduced exactly (300 of 300 answers).*
""")

# ----------------------------------------------------------------------------------------------------------------
md(r"""
## 6. Experiment B: are the shared mistakes caused by the wording?
For the five problems where the models misread the question or where the question is ambiguous, we wrote a
**clearer version with the same correct answer** and ran both models on both versions, with the same settings as
Notebook 01. R1 samples (temperature 0.6), so it is run **4 times** per version; Qwen is greedy, so once is enough.
""")

code(r"""
REWORDED = {
    3:  "James runs 3 sprints per session, and he does 3 sessions every week. Each sprint is 60 meters. "
        "How many total meters does he run a week?",
    4:  "Every day, Wendi gives each of her chickens three cups of feed in total, split over three meals. "
        "In the morning, she gives her flock 15 cups of feed. In the afternoon, she gives them another 25 cups. "
        "She has 20 chickens. How many cups of feed does she need to give them in the final meal of the day?",
    7:  "Carla is downloading a 200 GB file at 2 GB/minute. When 40% of the file is downloaded, Windows restarts the "
        "computer, which takes 20 minutes. After the restart she downloads the whole file again from the beginning, "
        "without any interruption. How many minutes does it take in total, from the start until the download is finished?",
    12: "Carlos is planting a lemon tree. The tree costs $90 to plant. Each year it grows 7 lemons, which he sells for "
        "$1.5 each. It costs $3 a year to water and feed the tree. After how many years will he have earned more "
        "money from the tree than he has spent on it?",
    37: "John has 13 lego sets. He sells some of them for $15 each and uses all of that money to buy 8 video games "
        "for $20 each, after which he has $5 left. How many lego sets does he still have?",
}
N_SAMPLES_R1 = 4
REWORD_PATH = RESULTS_DIR / "nb03_rewording.jsonl"
RUN_REWORDING = True
MODEL_CFG = {R1:   dict(name=R1_NAME,   chat_template=True,  sampling=True,  max_new_tokens=8192, n=N_SAMPLES_R1, stop_after_answer=False),
             QWEN: dict(name=QWEN_NAME, chat_template=False, sampling=False, max_new_tokens=2048, n=1, stop_after_answer=True)}

pd.DataFrame({"original": problems.loc[list(REWORDED), "question"], "reworded": pd.Series(REWORDED),
              "gold": problems.loc[list(REWORDED), "gold"]})
""")

code(r"""
if torch is not None:
    class StopAfterAnswer(StoppingCriteria):
        # Stops each sequence once it contains a \boxed{...} answer (Qwen base model, as in Notebook 01)
        def __init__(self, tok, prompt_length):
            self.tok, self.prompt_length = tok, prompt_length

        def __call__(self, input_ids, scores, **kwargs):
            texts = self.tok.batch_decode(input_ids[:, self.prompt_length:], skip_special_tokens=True)
            return torch.tensor([bool(boxed_answers(t)) for t in texts], device=input_ids.device)


def generate(model, tok, questions, cfg):
    # Same decoding as Notebook 01. Returns (text, n_tokens, finished) per question.
    prompts = [build_prompt(tok, q, cfg["chat_template"]) for q in questions]
    inputs = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
    plen = inputs["input_ids"].shape[1]
    stop_ids = [tok.eos_token_id]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != tok.unk_token_id:
        stop_ids.append(im_end)
    decoding = (dict(do_sample=True, temperature=0.6, top_p=0.95) if cfg["sampling"]
                else dict(do_sample=False, temperature=None, top_p=None, top_k=None))
    criteria = StoppingCriteriaList([StopAfterAnswer(tok, plen)] if cfg["stop_after_answer"] else [])
    with torch.no_grad():
        out = model.generate(**inputs, **decoding, max_new_tokens=cfg["max_new_tokens"], pad_token_id=tok.pad_token_id,
                             eos_token_id=stop_ids, stopping_criteria=criteria)
    res = []
    for seq in out[:, plen:].tolist():
        end = next((k for k, t in enumerate(seq) if t in stop_ids), None)
        n = end if end is not None else len(seq)
        text = tok.decode(seq[:n], skip_special_tokens=True)
        finished = end is not None or (cfg["stop_after_answer"] and bool(boxed_answers(text))) or n < cfg["max_new_tokens"]
        res.append((text, n, finished))
    return res


def final_answer(text, thinks):
    if thinks:
        if "</think>" not in text:
            return None
        boxes = boxed_answers(text.split("</think>")[-1])
        return boxes[-1] if boxes else None
    boxes = boxed_answers(text)
    return boxes[0] if boxes else None


done = {(r["model"], r["id"]) for r in load_jsonl(REWORD_PATH)}
for model_label, cfg in MODEL_CFG.items():
    todo = [pid for pid in REWORDED if (model_label, pid) not in done]
    print(f"Rewording, {model_label}: {len(REWORDED) - len(todo)}/{len(REWORDED)} problems done")
    if not (todo and RUN_REWORDING and DEVICE):
        continue
    model, tok = load_model(cfg["name"])
    for pid in todo:   # one batch per problem: n samples of the original + n of the reworded question
        set_seed(SEED + int(pid))
        versions = [("original", problems.loc[pid, "question"])] * cfg["n"] + [("reworded", REWORDED[pid])] * cfg["n"]
        outs = generate(model, tok, [q for _, q in versions], cfg)
        rows = []
        for k, ((wording, q), (text, n, finished)) in enumerate(zip(versions, outs)):
            ans = final_answer(text, thinks=model_label == R1)
            rows.append(dict(model=model_label, id=int(pid), wording=wording, sample=k % cfg["n"], output=text,
                             answer=ans, correct=is_correct(ans, problems.loc[pid, "gold"]), n_tokens=n, finished=finished))
        append_jsonl(REWORD_PATH, rows)
        print(f"  #{pid}: " + " ".join(f"{r['wording'][:4]}:{'✓' if r['correct'] else ('…' if not r['finished'] else '✗')}"
                                       for r in rows))
    del model, tok
    free_memory()
""")

code(r"""
reword = pd.DataFrame(load_jsonl(REWORD_PATH))
if len(reword) == 0:
    print("No rewording results yet: run the cell above on a GPU.")
else:
    reword["outcome"] = reword.apply(lambda r: "cut off" if not r["finished"] else
                                     ("no answer" if r["answer"] is None else ("correct" if r["correct"] else "wrong")), axis=1)
    COLS = [(R1, "original"), (R1, "reworded"), (QWEN, "original"), (QWEN, "reworded")]
    pids = list(REWORDED)

    fig, ax = plt.subplots(figsize=(13, 0.95 * len(pids) + 1.6))
    for yy, pid in enumerate(pids):
        for k, (m, w) in enumerate(COLS):
            d = reword[(reword.model == m) & (reword.id == pid) & (reword.wording == w)].sort_values("sample")
            cw = 0.86 / N_SAMPLES_R1
            x0 = k + 0.07 if len(d) > 1 else k + 0.5 - cw / 2   # a single run is centred in its column
            for s, (_, r) in enumerate(d.iterrows()):
                o = r["outcome"]
                ax.add_patch(plt.Rectangle((x0 + s * cw + 0.01, yy + 0.15), cw - 0.02, 0.7, color=COLORS[o]))
                label = f"{SYMBOLS[o]}\n{number_str(r['answer'])}" if o in ("correct", "wrong") else SYMBOLS[o]
                ax.text(x0 + s * cw + cw / 2, yy + 0.5, label, ha="center", va="center", fontsize=8.5,
                        color=INK if o in ("cut off", "no answer") else "white", linespacing=1.1)
        ax.text(-0.05, yy + 0.5, f"#{pid}\ncorrect: {problems.loc[pid, 'gold']}", va="center", ha="right", fontsize=10,
                color=INK)
    for k, (m, w) in enumerate(COLS):
        ax.text(k + 0.5, -0.12, f"{m}\n{w}" + (f"  ({N_SAMPLES_R1} runs)" if m == R1 else "  (1 run)"),
                ha="center", va="bottom", fontsize=10, color=INK, weight="bold" if w == "reworded" else "normal")
    ax.axvline(2, color=INK, lw=1)
    ax.set_xlim(-0.75, 4.05)
    ax.set_ylim(len(pids), -0.9)
    ax.axis("off")
    hits = reword.groupby(["model", "wording"]).correct.agg(["sum", "count"])
    frac = lambda m, w: f"{hits.loc[(m, w), 'sum']} of {hits.loc[(m, w), 'count']}"
    ax.set_title(f"Rewording: R1 correct in {frac(R1, 'original')} runs → {frac(R1, 'reworded')}; "
                 f"Qwen in {frac(QWEN, 'original')} → {frac(QWEN, 'reworded')}",
                 loc="left", fontsize=12.5, color=INK, pad=12)
    ax.legend(handles=[Patch(color=COLORS[o], label=f"{SYMBOLS[o]} {o}") for o in ["correct", "wrong", "cut off"]],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    plt.tight_layout()
    plt.show()
""")

md(r"""
**Takeaways (section 6)**
* **Qwen's mistakes here are caused by the wording:** with the clearer questions it goes from 0 to 4 of 5 correct.
* **R1 improves from 10 to 17 of 20 runs.** The misread "3 sprints 3 times a week" (#3) and the self-contradicting
  #37 are fixed completely (0/4 → 4/4 and 1/4 → 4/4).
* **#7 stays hard for both**, even when the restart is spelled out: there the problem is reasoning about the story,
  not reading it.
* **A single R1 run can be unlucky.** On the *original* #4 and #12, where R1 failed in run_4, all 4 new runs are correct.
  R1 samples, so its run_4 failures are partly chance, while Qwen (greedy) fails the same way every time: it gave exactly
  the same wrong answers as in run_4. This also checks that the local setup reproduces the Colab run.

*Note: these runs were made on an Apple M3 GPU (MPS), not the Colab T4. The reworded questions were written by the AI
and should be checked by the team: a rewording must make the question clearer, not easier.*
""")

# ----------------------------------------------------------------------------------------------------------------
md(r"""
## 7. Summary
**Can a small reasoning model solve grade-school math?** Mostly (43 of 50), and here is what the reasoning changes:

1. **The failures are nested.** R1-Distill solves every problem its base model Qwen2.5-Math solves, plus 3 more
   (section 1). On 50 problems this difference is not statistically secured (p = 0.25), so what follows is about *how*
   the models differ.
2. **Reasoning fixes *carrying out*, not *understanding*.** R1's 3 extra wins are exactly Qwen's 3 made-up code outputs:
   R1 does the arithmetic in text instead of pretending to run code (sections 2–3). When the question is misread,
   both models misread it the same way, which fits R1 keeping these errors from its base model.
3. **Many "hard" problems are hard because of their wording.** Clearer versions of the misread or ambiguous questions
   fix 4 of 5 for Qwen and raise R1 from 10 to 17 correct runs out of 20 (section 6).
4. **The thinking does real work, but too much of it.** Without thinking R1 gets 12% right, with three quarters of it
   90%, and the last quarter changes no answer (section 5). R1 is even ahead of its text: in 26 of 41 problems it can
   give the correct answer before the number is written. After that it keeps re-computing the same values (section 4).
5. **The two models get stuck differently.** Qwen falls into a loop and repeats the same sentences word for word; R1
   goes in circles with new wording, doubts more the longer it thinks, and runs out of tokens, sometimes after already
   having the answer (#8) (sections 2.1 and 4).

**Limits:** 50 problems and one run per model; R1 samples, so single failures can be chance (section 6). The answer
positions come from text matching (sections 4 and 5.2), the layer read-out only checks the first digit (section 5.3),
and the failure types and rewordings are judgements made with the AI that the team has checked.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
Path(sys.argv[1]).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("written", sys.argv[1])
