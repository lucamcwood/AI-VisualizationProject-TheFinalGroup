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
| … | _TODO_ | | | |

## 2. How AI suggestions affected analytical decisions

- Each model is prompted the way it works best (R1: chat template + sampling; Qwen-Math base: plain text + greedy + stop after first answer). This is a methodological choice we must state in the report.
- R1 token budget 4096 instead of 2048: fewer cut-off answers, but runs ~2× slower.
- _TODO: decisions made in later notebooks._

## 3. Failure patterns of AI assistance

- **Over-engineering:** the first version did far more than needed and was hard to check.
- **Fixes not verified before delivery:** the AI could not run the models itself (we run them only on Colab), so its first Qwen fix (system prompt) was wrong and needed another iteration with real outputs.
- **Accuracy alone hid problems:** the bugs only became visible when the AI read the raw model outputs, not the numbers.
- **Version control slip:** in one step the AI overwrote an earlier commit of its own and had to merge the lost parts back.
- _TODO: more patterns we notice._

## 4. Emerging best practices

- Start minimal, run it, then extend.
- Feed real outputs (notebook + result files) back to the AI and ask it to look at raw generations, not only scores.
- Let the AI verify external facts (dataset names, chat templates, model settings) against the source instead of assuming.
- Keep runs resumable and results on Drive, so iterations are cheap.
- Review every AI commit before building on it.
- _TODO._
