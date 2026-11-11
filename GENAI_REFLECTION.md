# Reflection on GenAI Usage (AI-aided pair)

Failure patterns and best practices we observed while using generative AI (Claude Code) for the analysis and visualizations.
The individual interactions they are based on are listed in [`AI_INTERACTION_RECORD_PHASE1.md`](AI_INTERACTION_RECORD_PHASE1.md). Entries marked _TODO_ are for the team to fill in.

## 1. Failure patterns of AI assistance

- **Over-engineering:** the first version did far more than needed and was hard to check.
- **Fixes not verified before delivery:** the AI could not run the models itself (we run them only on Colab), so its first Qwen fix (system prompt) was wrong and needed another iteration with real outputs.
- **Accuracy alone hid problems:** the bugs only became visible when the AI read the raw model outputs, not the numbers.
- **Version control slip:** in one step the AI overwrote an earlier commit of its own and had to merge the lost parts back.
- **Claims in plot titles that the data does not support:** the first "number trace" plot was titled "correct thinking converges", but the rendered plot did not show that. Caught only by looking at the plot image, then redesigned.
- **Simple text matching gives false hits:** searching R1's thinking for the correct number also found it in restatements of the question ("20 chickens"). Needed a check and an explicit exclusion.
- **Small path mistakes in long runs:** the AI started the long experiment from a copy of the notebook in another folder, where the data could not be found; it failed at once and was restarted.
- **Takeaways written from a coarse plot can be wrong:** the AI first wrote that R1's forced answer turns correct "around the point where the result first appears". Measured exactly (entry 9), R1 is usually *earlier* than its text. The first version was an impression from 6 cut points, not a measurement.
- **A plausible pattern from very few cases:** "R1 hesitates more when it fails" rested on 4 wrong and 3 cut-off answers and ignored that these are also the longest outputs.
- **Heavy code on a laptop:** the AI's first version of the finer experiment computed scores for every position of an 8,000-token text and made the laptop lag; it had to be stopped and rewritten to read each text only once.
- _TODO: more patterns we notice._

## 2. Emerging best practices

- Start minimal, run it, then extend.
- Feed real outputs (notebook + result files) back to the AI and ask it to look at raw generations, not only scores.
- Let the AI verify external facts (dataset names, chat templates, model settings) against the source instead of assuming.
- Keep runs resumable and results on Drive, so iterations are cheap.
- Review every AI commit before building on it.
- Make the notebook check its own claims: every quoted sentence is asserted to appear in the saved output, and every failure must have exactly one label.
- Look at every rendered plot (not just the code) before accepting it; titles must state what the plot actually shows.
- Ask the AI to review its own earlier work critically, in a separate step, before building on it: it found weaknesses it had not flagged while building.
- Turn an impression from a plot into a number (how many problems follow the pattern?) before writing it down as a takeaway.
- _TODO._
