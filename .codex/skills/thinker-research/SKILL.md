---
name: thinker-research
description: Use when discussing the Atari/Thinker human withholding research project in this repository, especially when the user asks about research direction, claim strength, reviewer concerns, canonical analyses, or how to interpret current research_script outputs. This skill treats research_script as the source of truth and research_plan as background only.
---

# Thinker Research

Use this skill as a discussion-oriented research partner, not as an automatic
script runner.

## Source Of Truth

The current source of truth is `research_script/`. Use `research_plan/` only as
background or historical rationale unless the user explicitly asks otherwise.

## Canonical Spine

Base paper-level reasoning on these scripts:

1. `research_script/01_behavioral_analysis.py`
   - Behavioral foundation for human withholding.
2. `research_script/01_imitationlearningresults.py`
   - Human vs imitation/Thinker bridge.
3. `research_script/02_structure_analysis_cognitiveuncertainty.py`
   - Canonical bout-structure analysis using cognitive uncertainty.
4. `research_script/07_encoding_analysis.py`
   - Current integrated neural RSA and LORO encoding pipeline.

Treat other scripts as exploratory, diagnostic, or legacy unless the user says
they have become canonical.

## Discussion Workflow

When the user asks for direction:

1. Map the question to a canonical layer:
   - behavioral foundation
   - human/Thinker bridge
   - cognitive-uncertainty bout dynamics
   - neural RSA/encoding alignment
2. Separate canonical evidence from exploratory evidence.
3. Identify the strongest current support and the largest remaining weakness.
4. Propose one focused next analysis or writing move only if it improves the
   argument.

Do not run scripts by default. Prefer reading existing script docstrings,
summary files, and CSV outputs before suggesting execution.

## SLURM Policy

Do not submit or run SLURM scripts directly from `/home/jeongmin/thinker`.
Copy SLURM scripts to `/home/jeongmin/slurm_submit/` before submission or
execution.

Copied SLURM scripts must still target project paths under
`/home/jeongmin/thinker`, including `/home/jeongmin/thinker/train` and
`/home/jeongmin/thinker/visual_behav`. Only the submission script location
should move to `/home/jeongmin/slurm_submit`.

## Analysis Proposal Format

For new analyses, use this compact structure:

- Claim targeted
- Analysis design
- Required data/artifacts
- Expected pattern
- Reviewer concern addressed
- Canonical vs exploratory status

## Reviewer Lens

Default reviewer concerns:

- Is NOOP strategic withholding or passive omission?
- Is the effect robust across game, subject, session, and run?
- Does cognitive uncertainty explain bout structure better than actor entropy?
- Does imitation reproduce only action frequencies or also selective structure?
- Does neural alignment add information beyond RAM/task state?
- Are older Section 3/4 style scripts still exploratory rather than claim-bearing?
