# Engineering Online Experimentation — Samples and Code
This repository contains sample chapters and code the book project “Engineering Online Experimentation: Architecture, Pipelines, and Statistical Methods for Production‑Scale Systems.” 

## What’s inside

- [Sample Chapters](sample_chapters/)
  - [ch0_front_matter.md](sample_chapters/ch0_front_matter.md): Front matter and full Table of Contents]
  - [ch3_designing_trustworthy_experiments.md](sample_chapters/ch3_designing_trustworthy_experiments.md): Designing Trustworthy Experiments.
  - [ch13_ranking_experiments.md](sample_chapters/ch13_ranking_experiments.md): Evaluating Ranking Systems (interleaving, Wilcoxon, power simulation)
  - [ch16_contextual_bandits.md](sample_chapters/ch16_contextual_bandits.md): Contextual Multi-ARM Bandit
- [Code in notebooks](https://github.com/thunderbird2009/book_online_exp/tree/main/code/)

## Suggested review flow (10–30 minutes)
1) Skim Ch0 front matter and ToC (if present) to confirm scope and structure.
2) Read the two core samples in order:
   - Ch3 — experimental design and pitfalls (fixed horizon, decision rules, multiple testing, peeking)
   - Ch13 — interleaving for ranking with Wilcoxon and simulation‑based power
   - Ch16 — Contextual Bandit with LinTS, linUCB, Neural Bandits, etc.
   - Refer to python notebook ch3_*, ch13_* and ch16_* for working code examples.

## Code quickstart
If you want to run examples, use the minimal setup below. Otherwise, you can ignore this section.

Prerequisites
- Python 3.10+ (3.11/3.12 fine)
- Git

Setup (either pip or conda)
- Use venv kernel for the notebook.
- Install dependencies using the first cell in each notebook.

## Rights and data policy
- Code license: Apache‑2.0
- Figures: original unless noted; any third‑party images will be cleared prior to production
- Data: synthetic/anonymized samples; no proprietary datasets or confidential screenshots
- Accessibility: alt text for figures; colorblind‑safe palettes; semantic headings

