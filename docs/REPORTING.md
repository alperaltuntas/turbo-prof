# Report structure: facts vs. commentary

A turbo-prof report is generated in **two layers**, kept deliberately separate:

1. **Facts** — produced by `gen_report.py` from the run logs. Run parameters,
   plots, and data tables, plus captions that describe *what is drawn*, not what
   it means. This layer is deterministic and reproducible from the logs: same
   logs in, byte-for-byte same `report.md` out.
2. **Commentary** — the interpretation: which routine is the bottleneck, what a
   ratio implies, what to profile next. This is run-specific judgement, added on
   top of the facts by the **`report-commentary` skill** (see
   `.claude/skills/report-commentary/`).

## Why split them

The interpretation depends on *what the numbers turned out to be*. If
`gen_report.py` hard-codes "the barotropic solver is the bottleneck," the next
run with different data will confidently lie. Worse, a reader can't tell which
sentences are machine-emitted facts and which are a judgement call. Splitting the
layers fixes both: the facts report is reproducible and auditable, and the
commentary is visibly a separate, model-written overlay grounded in those facts.

## The anchor / marker convention

`gen_report.py` leaves an HTML-comment anchor at each point where commentary
belongs (invisible in rendered Markdown, so the facts report is complete and
valid on its own):

```
<!-- commentary: NAME -->
```

The current anchors are: `key-finding`, `cpu-timing`, `throughput`, `speedup`,
`head-to-head`, `breakdown`, `barotropic`, `continuity`, `init`, `failures`.

The skill fills an anchor by inserting a fenced block immediately after it:

```
<!-- commentary: breakdown -->
<!-- commentary-body: breakdown -->
> **Commentary.** Continuity offloads well (2.19x) while the barotropic solver
> regresses (0.40x) and ...
<!-- /commentary-body: breakdown -->
```

The fences make the two layers mechanically separable.
`scripts/strip_commentary.py` removes every fenced body and leaves the bare
anchors — recovering exactly what `gen_report.py` produced. That is also how you
verify reproducibility: strip an annotated report and diff it against a fresh
`gen_report.py` run; only the commentary should differ.

## Workflow

```bash
# 1. Facts layer (under the npl conda env, for matplotlib)
python3 gen_report.py --cpu-dir "$STACK/examples/double_gyre" \
    --gpu-dir "$STACK/examples/double_gyre" --stack-dir "$STACK"

# 2. Commentary layer (optional) — in Claude Code, point the skill at the dir
/report-commentary reports/<the-new-dir>

# (recover the facts-only text at any time)
python3 strip_commentary.py reports/<dir>/report.md
```

## The grounding rule

The commentary layer must cite **only** numbers that appear in the facts
`report.md` or `results.csv`. It may not introduce figures of its own. This is
the whole point of the split — otherwise we'd have traded stale hard-coded
conclusions for freshly hallucinated ones. *Mechanism* (why a routine maps well
or poorly to the GPU) is grounded separately, in the actual MOM6 source: the
skill locates the tree via `provenance.json`'s `stack_dir`/`mom6_commit` and
greps a report's FMS clock name — a literal string in the source — to find the
routine it wraps. Source grounds the *why*; it never supplies numbers. The skill enforces this; see its
`SKILL.md` for the standing caveats it always applies (e.g. a GPU routine timer
includes OpenMP `map()` transfers, not just the kernel; `n/a` ratios are
structural, not GPU wins). The framing and timer caveats themselves live in
`METHODOLOGY.md`, the single source for that prose.
