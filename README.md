# AI-VisualizationProject-TheFinalGroup
Can a small reaseasoning model solve grade-school math problems?

## Notebooks

| # | Notebook | What it does |
|---|----------|--------------|
| 01 | [`notebooks/01_setup_and_first_run.ipynb`](notebooks/01_setup_and_first_run.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lucamcwood/AI-VisualizationProject-TheFinalGroup/blob/main/notebooks/01_setup_and_first_run.ipynb) | Setup + first run: both models (DeepSeek-R1-Distill-Qwen-1.5B and Qwen2.5-Math-1.5B) answer the same GSM8K problems. Runs on a free Colab T4, saves answers to Google Drive and resumes after disconnects. |
| 02 | [`notebooks/02_visualizations_without_AI.ipynb`](notebooks/02_visualizations_without_AI.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lucamcwood/AI-VisualizationProject-TheFinalGroup/blob/main/notebooks/02_visualizations_without_AI.ipynb) | Load experiment results from Google Drive and generates the visualizations of the subgroup that did not use AI. |
| 03 | [`notebooks/03_visualizations_with_AI.ipynb`](notebooks/03_visualizations_with_AI.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lucamcwood/AI-VisualizationProject-TheFinalGroup/blob/main/notebooks/03_visualizations_with_AI.ipynb) | Visualizations of the AI-aided subgroup: where the two models fail and why (failure types, case studies), what R1's long thinking does (answer timeline, value lanes, hesitation), and two experiments (early answering, reworded questions). Reads the run from Google Drive (Colab) or from `data/` (locally). |

## Phase 1 (Vis4AI) - How does the model behave?

Goal: Design a visualization that helps you observe patterns in what goes in and what comes out.

| Sub-group | Names | Used AI |
|-----------|-------|---------|
| 01 | Dal Bosco, Bhattarai | NO |
| 02 | Lehrner, Kühberger | YES |

## AI usage

The AI-aided pair documents prompts, decisions and lessons learned in [`AI_USAGE_LOG.md`](AI_USAGE_LOG.md).


## To do / Brainstorm ideas
 
### Visualization plots
 - Deeper dive in the false answers:
   - Are the two models making mistakes in the same questions or different ones?
   - Do the questions that cannot answered have something in common?
   - Can we engineer difficult questions that cannot be answered?

 - Deeper dive on chain of thought concept:
   - Can we somehow visulize the trajectories?
   - Can we prove that the model is really reasonikg other than counting the answer length?
   - Can we visualize the embeddings?

- Deeper dive on prompting
   - Can we somehow improve the model performance by prompting (asking please, being rude, ask to answer fast)
