#!/usr/bin/env python3
"""Strip skill-added commentary from a report, recovering the facts-only text.

`gen_report.py` emits a facts-only `report.md` with `<!-- commentary: NAME -->`
anchors. The report-commentary skill fills each anchor with an interpretation
block, fenced by `<!-- commentary-body: NAME -->` / `<!-- /commentary-body: NAME -->`
markers so the two layers stay mechanically separable. This script removes those
fenced blocks, leaving the bare anchors -- i.e. the exact text `gen_report.py`
produced.

Uses:
- verify reproducibility: `strip_commentary.py annotated.md` should match a fresh
  `gen_report.py` run (the facts layer is deterministic; commentary is not).
- re-annotate from scratch: strip, then re-run the skill.

    python3 strip_commentary.py REPORT.md            # print stripped text
    python3 strip_commentary.py REPORT.md -i          # rewrite in place
"""

import argparse
import re
import sys

# A filled commentary block: the body fence and everything between it. Removes
# the fence's own trailing newline but nothing more, so the blank line the facts
# layer placed after the anchor survives. The bare `<!-- commentary: NAME -->`
# anchor is preserved.
_BODY_RE = re.compile(
    r"<!-- commentary-body: (?P<name>[^>]+?) -->\n"
    r".*?"
    r"<!-- /commentary-body: (?P=name) -->\n",
    re.DOTALL,
)


def strip(text):
    """Return (stripped_text, n_blocks_removed)."""
    out, n = _BODY_RE.subn("", text)
    return out, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", help="path to report.md")
    ap.add_argument("-i", "--in-place", action="store_true",
                    help="rewrite the file instead of printing to stdout")
    args = ap.parse_args()

    with open(args.report) as fh:
        text = fh.read()
    stripped, n = strip(text)

    if args.in_place:
        with open(args.report, "w") as fh:
            fh.write(stripped)
        print(f"Removed {n} commentary block(s) from {args.report}", file=sys.stderr)
    else:
        sys.stdout.write(stripped)


if __name__ == "__main__":
    main()
