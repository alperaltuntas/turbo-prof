"""Shared report-output conventions, so every report generator names its output
directory the same way.

The convention: unless an explicit ``--outdir`` is given, a report lands in
``<reports-dir>/<YYYY-MM-DD-HHMMSS>-<label>``. The seconds-resolution stamp keeps
multiple reports per day distinct and chronologically sorted; on the rare
same-second collision a ``-2``, ``-3``, ... suffix is appended so a report is
never clobbered. ``DEFAULT_REPORTS_DIR`` is ``<repo>/reports``.
"""

import os

# <repo>/reports, resolved from this file (scripts/turboprof/reporting.py).
DEFAULT_REPORTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "reports"))


def resolve_outdir(args, now):
    """Pick the report directory from parsed args (.outdir/.reports_dir/.label).

    With ``--outdir`` set, honor it verbatim (explicit escape hatch). Otherwise
    build ``<reports-dir>/<YYYY-MM-DD-HHMMSS>-<label>``, appending -2, -3, ... on a
    same-second collision so an existing report is never overwritten.
    """
    if args.outdir:
        return args.outdir
    stamp = now.strftime("%Y-%m-%d-%H%M%S")
    base = os.path.join(args.reports_dir, f"{stamp}-{args.label}")
    outdir, n = base, 1
    while os.path.exists(outdir):
        n += 1
        outdir = f"{base}-{n}"
    return outdir
