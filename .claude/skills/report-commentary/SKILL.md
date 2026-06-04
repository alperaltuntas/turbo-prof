---
name: report-commentary
description: Add interpretive commentary to a turbo-prof facts-only benchmark report. Use when the user wants conclusions, bottleneck analysis, or "what does this mean / what next" added to a generated report.md, or asks to annotate/comment on a report. Fills the <!-- commentary: NAME --> anchors gen_report.py leaves behind.
---

# Report commentary

`gen_report.py` emits a **facts-only** `report.md` (run parameters, plots, data
tables) with `<!-- commentary: NAME -->` anchors at each point where
interpretation belongs. This skill fills those anchors with commentary that is
**grounded strictly in the report's own numbers**. Background:
`docs/REPORTING.md`. Framing and timer caveats: `docs/METHODOLOGY.md`.

## Steps

1. **Resolve the report directory.** If the user named one, use it. Otherwise
   pick the most recent under `reports/` (the dir names sort chronologically:
   `ls -d reports/*/ | sort | tail -1`). Confirm the choice if ambiguous.

2. **Load the facts.** Read `report.md`, `results.csv`, and `provenance.json`
   from that dir. `results.csv` is the source of truth for numbers; `report.md`
   shows which figures are already surfaced and where the anchors are. Build
   your understanding of this run's *numbers* from these files **only** — do not
   pull numbers from memory, other reports, or prior conversations.
   `provenance.json` also gives you `stack_dir` and `mom6_commit`, the keys to
   the source code (next section).

3. **Find the anchors.** Each is a line `<!-- commentary: NAME -->`. An anchor
   already followed by a `<!-- commentary-body: NAME -->` … `<!-- /commentary-body: NAME -->`
   block has been filled; you are **replacing** that block (re-run / refresh),
   not appending a second one.

4. **Write commentary per anchor** (see "What each anchor wants" and "Grounding
   rules"). Insert it immediately after the anchor, fenced:

   ```
   <!-- commentary: NAME -->
   <!-- commentary-body: NAME -->
   > **Commentary.** …your text…
   <!-- /commentary-body: NAME -->
   ```

   Use the `> ` blockquote prefix on every line so the commentary renders as a
   visibly distinct overlay. Keep each block tight — a few sentences; this is
   analysis, not an essay. Skip an anchor (leave it bare) if the data genuinely
   supports no claim — say so to the user rather than padding.

5. **Report back.** Summarize which anchors you filled and which you skipped and
   why.

## What each anchor wants

- `key-finding` — the headline: the one or two sentences a reader should leave
  with. Typically the continuity-vs-whole-loop tension at the clean comparison
  point (read the breakdown table + head-to-head).
- `cpu-timing` — what the weak-scaling vs saturated-node slopes show.
- `throughput` / `speedup` — where the GPU wins vs loses, and the crossover
  relative to the production operating points (the dashed verticals).
- `head-to-head` — read the speedup column; name the crossover size.
- `breakdown` — which routines offload well vs poorly, and which dominate the
  loop. This drives the key finding.
- `barotropic` — which sub-step carries the cost; whether it's compute or
  communication (check the halo-update rows); the optimization implication.
- `continuity` — the reviewer's focus: its share of the loop on each platform
  and its isolated speedup trend across sizes.
- `init` — the GPU init trend and what it implies for the memory ceiling.
- `failures` — what the failure cause implies (e.g. the device memory ceiling).

## Grounding rules (non-negotiable)

- **Cite only numbers that appear in `report.md` or `results.csv`.** Never
  introduce a figure of your own. If you write "2.2x faster," that ratio must be
  in the tables. This is the entire point of the facts/commentary split.
- **Apply the standing caveats** (full text in `METHODOLOGY.md`):
  - A GPU routine timer folds in the OpenMP `target … map()` host⇄device
    transfers and overhead — it is end-to-end cost, **not** bare kernel time. To
    split them you need Nsight Systems (`run-profile.sh`). Never call a GPU timer
    "kernel time."
  - `n/a` / near-zero ratios on `message passing` and BT halo updates are
    **structural** (the lone GPU rank does no inter-rank exchange), not GPU wins.
    Never report them as speedups.
  - The whole model is OpenMP-offloaded, so a GPU regression is **not** an
    Amdahl / un-ported-remainder story; it's about which offloaded routines map
    well to one GPU.
  - The clean 1-node-vs-1-GPU comparison is the band `i ≥ 128`; ratios left of
    it mix scaling regimes — caveat them.
- **Distinguish fact from inference.** Hedge genuine inferences ("consistent
  with launch-bound kernels," "suggests") rather than stating them as measured.
  A mechanism you can't see in the numbers (e.g. kernel launch overhead) is a
  hypothesis to name, with the measurement that would confirm it.
- **Don't restate the caption.** The facts layer already says what the plot
  shows; commentary says what it *means*.

## Grounding mechanism in the source code

Numbers come from the report; **mechanism** — *why* a routine maps well or
poorly to the GPU — should be grounded in the actual MOM6 source, not guessed.

- **Locate the source.** `provenance.json` gives `stack_dir`; the MOM6 tree is
  `<stack_dir>/submodules/MOM6/src`. Before relying on it, sanity-check that the
  tree exists and that the checked-out commit matches `provenance.json`'s
  `mom6_commit` (`git -C <stack_dir>/submodules/MOM6 rev-parse HEAD`). If it is
  gone or has moved on, say so and fall back to numbers-only commentary — do not
  describe code you cannot read.
- **Map a timer to its routine.** Every FMS clock name in the report is a
  *literal string* in the source. Grep it to find the routine the timer wraps:

  ```
  grep -rn "(Ocean continuity equation)" <stack_dir>/submodules/MOM6/src
  grep -rn "(Ocean BT pre-calcs only)"    <stack_dir>/submodules/MOM6/src
  ```

  Then read that routine to ground the claim.
- **What the code grounds.** Algorithm structure (e.g. the barotropic
  free-surface sub-cycle loop and how many sub-steps it fires per baroclinic
  step); the density and placement of `!$omp target teams loop` regions and
  their `map(...)` clauses; whether a per-substep `pass_var`/halo update sits
  inside the hot loop. These justify mechanistic statements — "many small
  offloaded kernels per step," "the cost is in the offloaded pre-calc, not
  communication" — instead of asserting them blind. Cite what you lean on as
  `file:line`.
- **The boundary stays firm.** Source grounds *mechanism*; it never supplies
  *numbers*, and it never overrides the measured timings. If a code-based
  hypothesis conflicts with the data, the data wins — report the tension rather
  than the guess.
