"""turboprof: reusable building blocks for turbo-prof performance reports.

These modules hold the parts that any report type shares, independent of which
report is being generated:

- ``parsing``     read MOM6/FMS run logs into canonical per-run records
- ``provenance``  capture and render the software-stack state for reproducibility

Report-type-specific logic (which plots, tables, and prose to emit) lives in
the report generators, not here.
"""
