"""Capture and render the software-stack state for reproducible reports.

Every report type should record what produced it. ``gather_provenance`` returns
a plain dict (also serialized to provenance.json), and ``render_provenance``
turns it into a Markdown section. Both are report-type agnostic.
"""

import datetime
import os
import socket
import subprocess


def _git(stack_dir, *args):
    """Run a git command in stack_dir; return stripped stdout or None."""
    try:
        out = subprocess.run(["git", "-C", stack_dir, *args],
                             capture_output=True, text=True, check=False)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, ValueError):
        return None


def gather_provenance(stack_dir, note, date):
    """Capture the software-stack state needed to reproduce this report.

    Records the turbo-stack commit (with dirty flag), the MOM6 submodule commit
    (the actual model code), the full `git submodule status` snapshot, and the
    GPU build flags from build-utils/makefile-templates/ncar-nvhpc.mk -- since
    those flags (mem:separate, HAVE_FC_DO_CONCURRENT_LOCAL) materially change
    generated code, the commit hash alone is not enough if the tree is dirty.
    """
    prov = {
        "date_generated": date or datetime.date.today().isoformat(),
        "host": socket.gethostname(),
        "note": note,
        "stack_dir": None,
        "stack_describe": None,
        "stack_dirty": None,
        "mom6_commit": None,
        "mom6_describe": None,
        "submodule_status": None,
        "offload_flags": None,
    }
    if not stack_dir or not os.path.isdir(stack_dir):
        return prov
    prov["stack_dir"] = os.path.realpath(stack_dir)
    prov["stack_describe"] = _git(stack_dir, "describe", "--always", "--dirty", "--tags")
    porcelain = _git(stack_dir, "status", "--porcelain")
    prov["stack_dirty"] = bool(porcelain) if porcelain is not None else None
    prov["submodule_status"] = _git(stack_dir, "submodule", "status")

    mom6 = os.path.join(stack_dir, "submodules", "MOM6")
    if os.path.isdir(mom6):
        prov["mom6_commit"] = _git(mom6, "rev-parse", "HEAD")
        prov["mom6_describe"] = _git(mom6, "describe", "--always", "--tags")

    mk = os.path.join(stack_dir, "build-utils", "makefile-templates",
                      "ncar-nvhpc.mk")
    if os.path.isfile(mk):
        flags = []
        with open(mk, errors="replace") as fh:
            for line in fh:
                s = line.strip()
                if s.startswith(("FFLAGS +=", "CFLAGS +=", "FPPFLAGS")) and \
                        ("gpu" in s or "HAVE_FC" in s):
                    flags.append(s)
        prov["offload_flags"] = flags or None
    return prov


def gather_provenance_multi(stacks, note, date):
    """Capture provenance for several stacks at once (comparison-sweep reports).

    ``stacks`` maps a short name (e.g. 'dev-turbo', 'iturbo-cpu') to a
    turbo-stack checkout path. Returns the usual top-level stamp fields plus a
    ``stacks`` dict of per-stack gather_provenance() records (their duplicate
    stamp fields trimmed).
    """
    prov = {
        "date_generated": date or datetime.date.today().isoformat(),
        "host": socket.gethostname(),
        "note": note,
        "stacks": {},
    }
    for name, stack_dir in stacks.items():
        sub = gather_provenance(stack_dir, None, date)
        for k in ("date_generated", "host", "note"):
            sub.pop(k, None)
        prov["stacks"][name] = sub
    return prov


def render_stamp(prov):
    """One-line 'generated on' stamp for the top of a report.

    The full Provenance section (commits, build flags, submodule snapshot) is
    reproducibility detail that belongs at the end; only this lightweight stamp
    leads the report.
    """
    return f"**Generated:** {prov['date_generated']} on `{prov['host']}`\n"


def render_provenance(prov, include_stamp=True):
    """Markdown 'Provenance' section from a gather_provenance() dict.

    With ``include_stamp=False`` the leading generated-on line is omitted (the
    caller is rendering it separately at the top via ``render_stamp``).
    """
    L = ["## Provenance\n"]
    if include_stamp:
        L.append(f"- **Generated:** {prov['date_generated']} on `{prov['host']}`")
    if prov.get("note"):
        L.append(f"- **Note:** {prov['note']}")
    if prov.get("stack_dir"):
        dirty = " (dirty working tree)" if prov.get("stack_dirty") else ""
        L.append(f"- **turbo-stack:** `{prov['stack_describe']}`{dirty} "
                 f"(`{prov['stack_dir']}`)")
        if prov.get("mom6_describe"):
            L.append(f"- **MOM6 submodule:** `{prov['mom6_describe']}` "
                     f"(`{prov['mom6_commit']}`)")
        if prov.get("offload_flags"):
            L.append("- **GPU build flags** (ncar-nvhpc.mk):")
            L.append("  ```make")
            L += [f"  {f}" for f in prov["offload_flags"]]
            L.append("  ```")
        if prov.get("submodule_status"):
            L.append("- **Submodule snapshot:**")
            L.append("  ```")
            L += [f"  {ln}" for ln in prov["submodule_status"].splitlines()]
            L.append("  ```")
    else:
        L.append("- _Software-stack provenance not recorded "
                 "(re-run with `--stack-dir`)._")
    if prov.get("stack_dirty"):
        L.append("\n> **Warning:** the turbo-stack working tree had uncommitted "
                 "changes when this report was generated, so the commit hash "
                 "does not fully capture the build. The GPU build flags above "
                 "are recorded explicitly for this reason.")
    return "\n".join(L) + "\n"


def render_provenance_multi(prov, include_stamp=True):
    """Markdown 'Provenance' section from a gather_provenance_multi() dict.

    One subsection per stack, each rendered with the single-stack logic.
    """
    L = ["## Provenance\n"]
    if include_stamp:
        L.append(f"- **Generated:** {prov['date_generated']} on `{prov['host']}`")
    if prov.get("note"):
        L.append(f"- **Note:** {prov['note']}")
    L.append("")
    for name, sub in prov.get("stacks", {}).items():
        L.append(f"### Stack: {name}\n")
        # Strip the sub-renderer's own "## Provenance" heading line; the
        # per-stack dirty-tree warning (if any) comes along with the body.
        body = render_provenance(sub, include_stamp=False)
        L.append(body.split("\n", 2)[2].rstrip())
        L.append("")
    if not prov.get("stacks"):
        L.append("- _Software-stack provenance not recorded "
                 "(re-run with `--stack NAME=PATH`)._")
    return "\n".join(L) + "\n"
