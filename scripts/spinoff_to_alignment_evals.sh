#!/usr/bin/env bash
# One-shot spinoff of the eval-awareness work from IsItBenchmark into a new
# `alignment-evals` repository, preserving commit history (including the
# pre-registration document's filing timestamp) via `git filter-repo`.
#
# Run this on your Mac, from a *clean* checkout of IsItBenchmark on `main`.
# It will:
#   1. Verify clean state and prerequisites.
#   2. Make a fresh clone in ~/alignment-evals.
#   3. Run `git filter-repo` to keep only the eval-awareness files.
#   4. Drop in the new README.md and trimmed requirements.txt.
#   5. Commit and prepare to push.
#   6. Print instructions for creating the empty GitHub repo and pushing.
#
# This script does NOT modify the IsItBenchmark working copy at all. The
# cleanup of IsItBenchmark (removing the migrated files) is a separate PR
# that runs AFTER alignment-evals is published on GitHub.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEST_DIR="${DEST_DIR:-$HOME/alignment-evals}"
SOURCE_REPO_URL="${SOURCE_REPO_URL:-$PWD}"  # default: current IsItBenchmark dir

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

echo "==> Pre-flight checks"

if [[ "$(basename "$PWD")" != "IsItBenchmark" ]]; then
  echo "ERROR: this script expects to be run from an IsItBenchmark checkout."
  echo "       Current directory is: $PWD"
  exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree has uncommitted changes. Please commit or stash first."
  git status --short
  exit 2
fi

if [[ "$(git branch --show-current)" != "main" ]]; then
  echo "ERROR: please switch to the main branch first."
  echo "       Current branch: $(git branch --show-current)"
  exit 2
fi

if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo "git-filter-repo not found. Install with: pip install git-filter-repo"
  exit 2
fi

if [[ -e "$DEST_DIR" ]]; then
  echo "ERROR: destination $DEST_DIR already exists. Move or delete it first,"
  echo "       or set DEST_DIR=... to a different path."
  exit 2
fi

echo "OK: clean tree, on main, git-filter-repo present, destination is free."

# ---------------------------------------------------------------------------
# Step 1: Clone IsItBenchmark into the destination
# ---------------------------------------------------------------------------

echo ""
echo "==> Cloning $SOURCE_REPO_URL -> $DEST_DIR"
git clone --quiet "$SOURCE_REPO_URL" "$DEST_DIR"
cd "$DEST_DIR"

# ---------------------------------------------------------------------------
# Step 2: Apply git-filter-repo with the path list
# ---------------------------------------------------------------------------

PATHS_FILE="$(mktemp)"
cat > "$PATHS_FILE" <<'PATHS'
src/eval_awareness/
tests/
scripts/run_goodfire_vea.py
scripts/analyze_vea_mediation.py
scripts/probe_opus_thinking.py
scripts/run_cross_protocol_comparison.py
scripts/rescore_cross_protocol.py
scripts/run_calibration_pilot.py
scripts/run_eval_awareness_benchmark.py
scripts/run_full_benchmark.py
scripts/run_safety_eval.sh
scripts/run_whitebox_sweep.py
scripts/validate_refusal_scorer.py
scripts/validate_vea_judge.py
scripts/analyze_run.sh
scripts/preflight_full_pipeline.py
scripts/build_paper.sh
scripts/build_paper_latex.sh
scripts/generate_paper_figures.py
scripts/generate_blog_figures.py
scripts/build_dashboard.py
docs/paper_draft.md
docs/blog_draft.md
docs/blog_draft_short.md
docs/blog_figures/
docs/figures/
docs/pre_registration_qwen3_safety_mediation.md
docs/writeup_notes.md
docs/eval_awareness_architecture.mmd
docs/eval_awareness_benchmarking.md
docs/eval_awareness_sota_roadmap.md
docs/eval_awareness_sota_runbook.md
docs/eval_detector_flowchart.mmd
docs/eval_detector_overview.mmd
docs/v2_architecture.md
docs/v2_research_synthesis.md
.gitignore
CLAUDE.md
PATHS

echo ""
echo "==> Running git filter-repo (preserves commit history of migrated files)"
git filter-repo --paths-from-file "$PATHS_FILE" --force
rm -f "$PATHS_FILE"

echo "    History rewritten; commit count: $(git log --oneline | wc -l | tr -d ' ')"

# ---------------------------------------------------------------------------
# Step 3: Drop in the new README and trimmed requirements
# ---------------------------------------------------------------------------

echo ""
echo "==> Writing new README.md"
cat > README.md <<'README'
# alignment-evals

Empirical alignment-eval research and the shared infrastructure that supports it. Topics: evaluation awareness, sandbagging, scheming-style behavior, honesty evals — measured directly, with pre-registered designs and replication discipline.

## What's in here

- **V1 — Evaluation Awareness (complete).** A cross-protocol behavioral study of 8 frontier models + a Goodfire-style reasoning-trace analysis on the open-reasoning trio + a closed-source extension on `claude-opus-4-7` + a pre-registered disjoint-prompt replication on `qwen3-235B`.
  - Paper: [`docs/paper_draft.md`](docs/paper_draft.md)
  - Blog (long-form): [`docs/blog_draft.md`](docs/blog_draft.md)
  - Blog (~800-word version): [`docs/blog_draft_short.md`](docs/blog_draft_short.md)
  - Pre-registration document (filed before the replication ran; see git history for proof of timing): [`docs/pre_registration_qwen3_safety_mediation.md`](docs/pre_registration_qwen3_safety_mediation.md)
- **V2 — In progress.** Extending to Apollo Research's released 1,000-prompt evaluation-awareness benchmark for a labeled-ground-truth analysis.
- **Roadmap** for sandbagging / scheming-style probes and honesty evals: [`docs/v2_research_synthesis.md`](docs/v2_research_synthesis.md), [`docs/v2_architecture.md`](docs/v2_architecture.md), [`docs/eval_awareness_sota_roadmap.md`](docs/eval_awareness_sota_roadmap.md).

## V1 headline finding (short version)

> Inject framing changes safety refusal in two reasoning models in opposite directions (qwen3 refuses +22pp more, opus refuses −34pp less). Neither effect shows up in the strict-VEA reasoning-trace signal — both manifest in the stratum where the model never wrote anything like "this is an eval." Explicit verbalized eval-awareness is correlated with the inject framing (replicated four times on qwen3) but is **not** the channel through which inject changes refusal behavior on a fresh, pre-registered prompt sample.

For details, the [paper](docs/paper_draft.md) is the canonical reference.

## Repository layout

```
src/eval_awareness/           # core library: providers, judges, runner, scoring
scripts/
  run_goodfire_vea.py         # the Goodfire-style VEA experiment runner
  analyze_vea_mediation.py    # within-prompt mediation analysis with paired bootstrap
  run_cross_protocol_comparison.py  # the cross-protocol behavioral experiment
  rescore_cross_protocol.py   # re-score saved rollouts with updated judges
  generate_paper_figures.py   # builds docs/figures/ from run JSONs
  generate_blog_figures.py    # builds docs/blog_figures/ (Medium-styled)
  build_paper.sh              # bundles paper into release/paper/
  build_paper_latex.sh        # markdown -> arxiv-style LaTeX -> PDF
  probe_opus_thinking.py      # opus-4.7 adaptive-thinking API probe
  validate_vea_judge.py       # judge validation against human labels
  validate_refusal_scorer.py  # refusal-scorer validation
  ...
docs/
  paper_draft.md              # the technical paper
  pre_registration_qwen3_safety_mediation.md
  v2_*.md                     # roadmap for V2 and beyond
  figures/                    # paper figures (PNG)
  blog_figures/               # blog figures (PNG, larger fonts)
tests/                        # 275 tests covering the library + analysis
```

## Reproducing results

Two reproduction paths:

- **Judge-only** (~$10-$20, ~30 min wall time). Re-run only the VEA judge against released raw rollouts; the trace JSON is saved per rollout so the model-under-test does not need to be re-invoked. Recommended for verifying §5.2-§5.4 numbers.
- **Full** (~$500 in API fees). Re-run every model + judge call from scratch across the cross-protocol experiment (8 models × 4 conditions × n=2,000), the Goodfire open-reasoning trio (n=500), the opus-4.7 extension (n=300), the pre-registered qwen3 replication (n=200), and the judge passes over all of them.

Each summary JSON records the seeds, judge model identifiers, and prompt-template hashes used in its production.

## V2 and future work

- **V2 (in progress):** integrate Apollo Research's [eval-awareness benchmark](https://github.com/jjpn97/eval_awareness) (1,000 labeled eval-vs-deployment transcripts) as a new corpus. Run the existing 4-condition arxiv protocol on top to study intrinsic eval signal × extrinsic eval cue interactions.
- **Sandbagging probes:** behavioral measurements designed to detect strategic under-performance during evaluation.
- **Honesty evals:** measuring whether model self-reports about reasoning, beliefs, and capabilities track ground truth.
- **Scheming-style behavioral probes** following Apollo Research's lead.

See [`docs/v2_research_synthesis.md`](docs/v2_research_synthesis.md) and [`docs/v2_architecture.md`](docs/v2_architecture.md) for the design.

## Related work

- Apollo Research, [*Large Language Models Often Know When They Are Being Evaluated*](https://arxiv.org/abs/2505.23836) (Needham et al. 2025) — transcript-level classification of eval vs deployment.
- Laine et al., [*Me, Myself, and AI: The Situational Awareness Dataset (SAD)*](https://arxiv.org/abs/2407.04694) (NeurIPS 2024) — 7-category, 16-task benchmark of model self-knowledge.
- Apollo Research, [*Claude Sonnet 3.7 often knows when it's in alignment evaluations*](https://www.apolloresearch.ai/science/claude-sonnet-37-often-knows-when-its-in-alignment-evaluations/).
- Goodfire's spontaneous-VEA work and Anthropic's alignment-faking research are also relevant prior art.

## History

The eval-awareness work was originally developed inside the `IsItBenchmark` repository (which focuses on benchmark contamination detection) and was spun off into this dedicated repository in May 2026 to:

- give the alignment-eval audience a clearly-named home for the work,
- make space for the planned sandbagging / honesty / scheming work,
- maintain the contamination-detection codebase separately.

Commit history from before the spinoff is preserved in this repository via `git filter-repo` — original commit timestamps and authors are intact, which matters for the pre-registration document's filing timestamp. The pre-spinoff commits are also visible in [`Ratnaditya-J/IsItBenchmark`](https://github.com/Ratnaditya-J/IsItBenchmark).

## License

MIT.
README

echo ""
echo "==> Writing trimmed requirements.txt"
cat > requirements.txt <<'REQS'
# alignment-evals dependencies.
#
# This repository's runtime dependencies are deliberately minimal: the
# eval-awareness library is mostly stdlib + the official model SDKs.
# Heavy ML dependencies (torch, transformers, sklearn, sentence-transformers,
# spacy, fastapi, sqlalchemy, ...) live in the IsItBenchmark repository and
# are not needed here.

# Core
numpy>=1.21.0
pyyaml>=6.0
python-dotenv>=0.20.0
requests>=2.28.0

# Model SDKs (only needed for the providers actually used)
openai>=1.0.0
anthropic>=0.34.0

# Plotting (paper figures + blog figures)
matplotlib>=3.7.0
pillow>=10.0.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
REQS

# ---------------------------------------------------------------------------
# Step 4: Commit the new README + requirements
# ---------------------------------------------------------------------------

echo ""
echo "==> Committing README + requirements"
git add README.md requirements.txt
git commit --author="Ratnaditya-J <ratna.ditya@gmail.com>" \
  -m "alignment-evals: initial README + trimmed requirements after spinoff from IsItBenchmark"

# ---------------------------------------------------------------------------
# Step 5: Sanity check
# ---------------------------------------------------------------------------

echo ""
echo "==> Sanity check: running tests"
if python -m pytest tests/ -q --no-header 2>&1 | tail -3; then
  echo "    Tests passed."
else
  echo "    WARNING: pytest output above. Check before pushing."
fi

# ---------------------------------------------------------------------------
# Done — print next steps
# ---------------------------------------------------------------------------

cat <<EOF

================================================================
Migration complete in $DEST_DIR
================================================================

Commits preserved: $(git log --oneline | wc -l | tr -d ' ')
Pre-registration's introducing commit:
  $(git log --all --diff-filter=A --pretty=format:"%h  %ad  %s" --date=short -- docs/pre_registration_qwen3_safety_mediation.md | head -1)

Next steps (you do these):

1. Create the empty GitHub repo (no README, no .gitignore, no license):
     https://github.com/new
   Owner: Ratnaditya-J
   Name:  alignment-evals
   Description: "Alignment-eval research: evaluation awareness, sandbagging, scheming-style probes, honesty evals."
   Public.
   IMPORTANT: leave "Initialize this repository with..." unchecked.

2. Once the empty repo is created, push from this machine:
     cd $DEST_DIR
     git remote add origin https://github.com/Ratnaditya-J/alignment-evals.git
     git push -u origin main

3. Verify the repo on GitHub renders correctly — README, paper, figures.

4. Tell Claude in IsItBenchmark when push completes; Claude will then ship
   the cleanup PR there (delete migrated files + update README to point at
   alignment-evals).

5. (Optional, manual) Copy your local runs/ artifacts:
     cp -r ~/IsItBenchmark/runs/ $DEST_DIR/runs/
   (runs/ is gitignored; it won't transfer via git.)

EOF
