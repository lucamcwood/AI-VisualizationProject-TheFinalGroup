# AI Usage Log (AI-aided pair)

Record of how generative AI was used in this project, as required for the AI-aided pair.
Tool: Claude Code (Claude Opus 5.5). Entries marked _TODO_ are for the team to fill in.

## 1. Interaction log

| # | Date | Prompt (short) | What the AI did | Decision (accepted / modified / rejected) and why |
|---|---|---|---|---|
| 1 | 2026-11-05 | Full project brief: build the first Colab notebook (R1-Distill-Qwen-1.5B on GSM8K + GSM-Symbolic, behaviour metrics, early answering, logit lens) | Checked the real dataset name/configs (`apple/GSM-Symbolic`: `main`, `p1`, `p2`), wrote a large notebook with all sections | **Rejected / reduced**: too much at once. We wanted a working setup first |
| 2 | 2026-11-05 | "Keep it really simple: get the dataset and try the problems with one model, then the other" | Rewrote as a minimal notebook: setup, GSM8K, run both models, side-by-side table; resumable saving to Drive | **Accepted** |
| 3 | 2026-11-05 | Uploaded run 1 outputs: "anything wrong / weird?" | Found: Qwen-Math did not stop and invented new questions (wrong answers taken from them); R1 cut off at 2048 tokens in 10/50. Raised R1 budget to 4096, stop Qwen after its first `\boxed{}`, clearer table | **Accepted**. R1: 38 → 43/50 |
| 4 | 2026-11-05 | Uploaded run 2 outputs: "and now?" | Found Qwen-Math writing junk tokens and looping: it is a base model and never learned the chat template. Switched it to a plain-text prompt | **Accepted**. Qwen-Math: 21 → 40/50 |
| 5 | 2026-11-05 | Uploaded run 3 outputs | No setup bugs left. Pointed out Qwen "imagines" Python code output, causing several wrong answers | **Accepted** as a finding to analyse later |
| 6 | 2026-11-09 | "Create a notebook for the AI visualizations" (with the run_4 result files) | Read the raw results first and found that the failures are nested (Qwen never solves a problem R1 misses) and that both models give the same wrong answer in 3 problems. Built notebook 03 with an outcome grid of all 50 problems and a zoom on the 10 failed ones (answers + question). Ran it locally and fixed the layout bugs it saw in the rendered plots (overlapping labels, `$` in questions read as math) | **Accepted** |
| 7 | 2026-11-09 | "How should I continue?", then "think about smart ways to continue" | Proposed next steps; read the outputs of the shared wrong answers and measured where the correct number appears in R1's thinking (after ~35–40%). Proposed: failure types, case studies, answer timeline, early answering, rewording | **Accepted** (asked to build all of it) |
| 8 | 2026-11-09 | "Build everything in notebook 03, with an interesting story line and nice visualizations" | Read all 17 failed outputs and labelled each with a failure type (with evidence). Extended notebook 03 to 7 sections: outcome grid, failure types, case studies with highlighted quotes, answer timeline, value lanes, hesitation words, and two new experiments (early answering; reworded questions). Ran the experiments **locally on the Apple M3 GPU (MPS)** instead of Colab, and checked every rendered plot. Found and fixed its own mistakes: a plot whose title claimed something the data did not show (replaced by a new design), and false matches when the correct number also appears in the question (now flagged and excluded) | _TODO: accepted / modified / rejected and why_ |
| … | _TODO_ | | | |

## 2. How AI suggestions affected analytical decisions

- Each model is prompted the way it works best (R1: chat template + sampling; Qwen-Math base: plain text + greedy + stop after first answer). This is a methodological choice we must state in the report.
- R1 token budget 4096 instead of 2048: fewer cut-off answers, but runs ~2× slower.
- Notebook 03 looks at single problems instead of averages (Notebook 02 already covers those), starting with "do both models fail on the same problems?".
- Colour for the outcomes is backed by a symbol (✓ ✗ …) so the plots work without colour (colour-blind readers, print).
- **Failure types** (notebook 03, section 2) were proposed by the AI after reading every failed output. They are a judgement, so the team must check each label against the output.
- **Early answering** (cut the thinking at x % and force an answer) follows Lanham et al. (2023). It was suggested by the AI in the first brief (entry 1) and only used now.
- **Reworded questions** (Experiment B) were written by the AI. A rewording can also make a question *easier* (not only clearer), which would bias the result; the team should check that each one keeps the difficulty.
- The experiments of notebook 03 ran on an Apple M3 GPU, not on the Colab T4. R1 samples, so its exact outputs differ between machines; the conclusions should not depend on single outputs.
- _TODO: decisions made in later notebooks._

## 3. Failure patterns of AI assistance

- **Over-engineering:** the first version did far more than needed and was hard to check.
- **Fixes not verified before delivery:** the AI could not run the models itself (we run them only on Colab), so its first Qwen fix (system prompt) was wrong and needed another iteration with real outputs.
- **Accuracy alone hid problems:** the bugs only became visible when the AI read the raw model outputs, not the numbers.
- **Version control slip:** in one step the AI overwrote an earlier commit of its own and had to merge the lost parts back.
- **Claims in plot titles that the data does not support:** the first "number trace" plot was titled "correct thinking converges", but the rendered plot did not show that. Caught only by looking at the plot image, then redesigned.
- **Simple text matching gives false hits:** searching R1's thinking for the correct number also found it in restatements of the question ("20 chickens"). Needed a check and an explicit exclusion.
- **Small path mistakes in long runs:** the AI started the long experiment from a copy of the notebook in another folder, where the data could not be found; it failed at once and was restarted.
- _TODO: more patterns we notice._

## 4. Emerging best practices

- Start minimal, run it, then extend.
- Feed real outputs (notebook + result files) back to the AI and ask it to look at raw generations, not only scores.
- Let the AI verify external facts (dataset names, chat templates, model settings) against the source instead of assuming.
- Keep runs resumable and results on Drive, so iterations are cheap.
- Review every AI commit before building on it.
- Make the notebook check its own claims: every quoted sentence is asserted to appear in the saved output, and every failure must have exactly one label.
- Look at every rendered plot (not just the code) before accepting it; titles must state what the plot actually shows.
- _TODO._
