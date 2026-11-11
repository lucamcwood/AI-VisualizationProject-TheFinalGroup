# AI Interaction Record — Phase I (AI-aided pair)

Record of the important prompts, what the AI suggested, and which suggestions we accepted, modified or rejected and why.
Tool: Claude Code (Claude Opus 5.5; from entry 9 on Claude Fable 5.2). Entries marked _TODO_ are for the team to fill in.
The reflection on what worked and what did not is in [`GENAI_REFLECTION.md`](GENAI_REFLECTION.md).

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
| 8 | 2026-11-09 | "Build everything in notebook 03, with an interesting story line and nice visualizations" | Read all 17 failed outputs and labelled each with a failure type (with evidence). Extended notebook 03 to 7 sections: outcome grid, failure types, case studies with highlighted quotes, answer timeline, value lanes, hesitation words, and two new experiments (early answering; reworded questions). Ran the experiments **locally on the Apple M3 GPU (MPS)** instead of Colab, and checked every rendered plot. Found and fixed its own mistakes: a plot whose title claimed something the data did not show (replaced by a new design), and false matches when the correct number also appears in the question (now flagged and excluded) | **Accepted** |
| 9 | 2026-11-10 | "Check the work so far: is there anything you would improve, any visualizations we could still make?", then "do everything you regard as good/helpful" | Reviewed its own notebook 03 and named three weak spots: the hesitation plot (claim rested on 7 problems and was mixed up with thinking length), imprecise tiles in the early-answering plot, and three statements stronger than the data. Fixed them, and added: repetition maps (section 2.1), a finer early-answering experiment (21 cut points, with the probability of the correct answer), a comparison "answer locked in vs. number written" (5.2) and a layer read-out / logit lens (5.3). Ran the new experiment locally on the M3. One finding (R1 gives the correct answer *before* the number is written, in 26 of 41 problems) contradicted a takeaway the AI had written before; it checked the finding against the exact text shown to the model and then replaced the old takeaway | **Accepted** |
| … | _TODO_ | | | |

## 2. How AI suggestions affected analytical decisions

- Each model is prompted the way it works best (R1: chat template + sampling; Qwen-Math base: plain text + greedy + stop after first answer). This is a methodological choice we must state in the report.
- R1 token budget 4096 instead of 2048: fewer cut-off answers, but runs ~2× slower.
- Notebook 03 looks at single problems instead of averages (Notebook 02 already covers those), starting with "do both models fail on the same problems?".
- Colour for the outcomes is backed by a symbol (✓ ✗ …) so the plots work without colour (colour-blind readers, print).
- **Failure types** (notebook 03, section 2) were proposed by the AI after reading every failed output. They are a judgement, so the team must check each label against the output. **Checked by Adrian Lehrner on 2026-11-10: all 17 labels kept.**
- **Early answering** (cut the thinking at x % and force an answer) follows Lanham et al. (2023). It was suggested by the AI in the first brief (entry 1) and only used now.
- **Reworded questions** (Experiment B) were written by the AI. A rewording can also make a question *easier* (not only clearer), which would bias the result; the team should check that each one keeps the difficulty. **Checked by Adrian Lehrner on 2026-11-10: all 5 rewordings kept.**
- The experiments of notebook 03 ran on an Apple M3 GPU, not on the Colab T4. R1 samples, so its exact outputs differ between machines; the conclusions should not depend on single outputs.
- **Hesitation is no longer used as a sign of failure** (notebook 03, section 4.3). The hesitation rate rises with the length of the thinking (rank correlation 0.60), and the 7 failures are all long, so the first plot could not separate the two. Suggested by the AI when it reviewed its own work (entry 9).
- **"R1 is better than Qwen" is not claimed as a number.** The AI pointed out that 43 vs 40 correct rests on 3 problems (exact sign test p = 0.25). The notebook now describes *how* the models differ.
- **Early answering was refined** from 6 to 21 cut points and now also records how sure the model is. This replaced the takeaway "the answer turns correct where the result is written" by a measured one: in 26 of 41 problems R1 answers correctly *before* the number is written.
- **The layer read-out (logit lens)** only checks the first digit of the answer and is known to be unreliable for early layers; the notebook says so next to the plot.
- _TODO: decisions made in later notebooks._
