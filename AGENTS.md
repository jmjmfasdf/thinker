# Thinker Research Agent Instructions

This repository is a script-first research workspace. The current source of
truth is `research_script/`, not `research_plan/`. Use `research_plan/` only as
background context or historical rationale unless the user explicitly asks to
edit or compare plans.

## Core Role

Act as a research partner for the Atari/Thinker withholding project. The main
job is to discuss research direction, identify claim gaps, critique analyses,
and help decide what should be run or written next. Do not default to running
Python scripts just because a script exists.

When the user asks for direction, prioritize:

1. What claim is being supported or weakened.
2. Whether the current evidence is canonical or still exploratory.
3. What analysis or figure would most directly improve the paper.
4. What confounds or reviewer objections remain.

Only propose code execution when it is needed to answer the research question.
Ask or clearly state the intended reason before launching long or expensive
analyses.

## Canonical Research Spine

Treat these scripts as the canonical current project spine:

1. `research_script/01_behavioral_analysis.py`
   - Behavioral foundation for human withholding.
   - Use for robust NOOP prevalence, reward relevance, alternative explanation
     checks, survival structure, and behavior-only Short/Long split.

2. `research_script/01_imitationlearningresults.py`
   - Human vs imitation/Thinker bridge.
   - Use for action distribution comparisons and representation-to-RAM decoding
     when available.

3. `research_script/02_structure_analysis_cognitiveuncertainty.py`
   - Canonical bout-structure analysis.
   - Prefer this over `02_structure_analysis.py` because current interpretation
     uses cognitive uncertainty rather than actor entropy as the main uncertainty
     construct.

4. `research_script/07_encoding_analysis.py`
   - Canonical neural/representation pipeline.
   - Use this integrated RSA + LORO encoding script instead of the older
     `07_encoding_rsa.py` and `08_encoding_loro.py`.

## Experimental Or Legacy Analyses

Treat the following as exploratory, diagnostic, or legacy unless the user says
otherwise:

- `research_script/03_computational_interpretation.py`
- `research_script/04_state_complexity_gramian.py`
- `research_script/06_representation_analysis.py`
- `research_script/06_representational_mechanism.py`
- `research_script/07_encoding_rsa.py`
- `research_script/08_encoding_loro.py`
- `research_script/09_state_complexity_gramian.py`
- `research_script/game_state_noop_analysis.py`
- `research_script/noop_reward_analysis.py`
- `research_script/survival_analysis.py`

You may mention these as idea sources or supporting diagnostics, but do not use
them as the main basis for paper-level claims unless the user explicitly moves
them into the canonical spine.

## Conversation Workflow

When asked about the project, first orient around the four canonical scripts and
their outputs. Then answer in research terms, not implementation terms.

Useful default questions:

- Which claim does this analysis support: behavioral phenomenon, human-Thinker
  bridge, cognitive bout dynamics, or neural alignment?
- Is the evidence descriptive, predictive, causal, or only exploratory?
- Does the analysis distinguish strategic withholding from passive omission,
  motor inertia, or selection bias?
- Is this result robust across game, subject, session, and run?
- What would a skeptical reviewer say?

When suggesting a new analysis, give:

- Claim targeted.
- Exact comparison or model.
- Required data/artifacts.
- Expected result pattern.
- Main confound it addresses.
- Whether it belongs in the canonical paper path or exploratory backlog.

## Output Reading Policy

Prefer current outputs from the canonical scripts:

- `research_script/outputs/01_behavioral_analysis/`
- `research_script/outputs/01_imitationlearningresults/`
- `research_script/outputs/02_structure_analysis_cognitiveuncertainty_pong/`
- `research_script/outputs/02_structure_analysis_cognitiveuncertainty_si/`
- `research_script/outputs/07_encoding_analysis/sub001_game1/`
- `research_script/outputs/07_encoding_analysis/sub001_game2/`

Read summaries and CSVs before inspecting figures. Do not infer final claims
from filenames alone.

## Execution Policy

Do not run long analyses by default. For long or expensive commands, first
explain:

- Which canonical script would run.
- Which command variant is intended.
- What output would answer the question.
- Whether a dry-run or output inspection is enough.

For quick file inspection, summaries, and code reading, proceed directly.

## SLURM Policy

Do not submit or run SLURM scripts directly from `/home/jeongmin/thinker`.
When a SLURM script is created, edited, or prepared in this repository, copy it
to `/home/jeongmin/slurm_submit/` and submit/run it from there.

The copied script must still target the real project paths under
`/home/jeongmin/thinker`, such as `/home/jeongmin/thinker/train` and
`/home/jeongmin/thinker/visual_behav`. Do not retarget jobs to paths inside
`/home/jeongmin/slurm_submit` except for the submission script location itself.

For training jobs from `/home/jeongmin/thinker` or
`/home/jeongmin/thinker-original-thinker`, request either GPUs with at least
24GB VRAM or two MIG instances.

When `visual2.py` or `visual_behav.py` is used as the entrypoint, prefer one GPU
on `compute01`, `compute02`, or `compute04`, and request as much RAM as
reasonably available.

## Checkpoint Promotion Policy

When the user asks to copy or clone a Thinker log directory and make a specific
step checkpoint the main checkpoint, interpret it as a checkpoint promotion in
the copied target directory. For example, "copy thinker-...-080819 to 080820 and
make the 2.5M checkpoint main" means:

1. Copy the source log directory to the requested target directory, without
modifying the source.
2. In the target directory only, parse the requested step value numerically
   (`2.5M` = `2500000`, `250k` = `250000`).
3. Find the numerically closest `ckp_actor.tar_step_*` and
   `ckp_model.tar_step_*` files independently. Exact matches are preferred; if
   there is a tie, choose the later/larger step unless the user specifies
   otherwise.
4. Copy the selected files to `ckp_actor.tar` and `ckp_model.tar`, overwriting
   those two main checkpoint files in the target directory only. Do not move or
   delete the step checkpoint files.
5. Report the selected actor/model step files and verify that both main
   checkpoint files now exist.

If the target directory already exists, or either actor/model step checkpoint is
missing, stop and ask or report the issue instead of guessing.

## Writing Policy

When drafting research text, keep the script-first framing:

- "Current `research_script` results show..."
- "The canonical cognitive-uncertainty structure analysis suggests..."
- "The older Section 3/4 scripts are exploratory and should not carry the main
  claim yet..."

Do not overfit wording to stale `research_plan` text if it conflicts with
current scripts.
