"""Regenerate THIRD-PARTY-NOTICES.md from what PyInstaller actually bundled.

Run after a build, before shipping:
    pyinstaller zeus.spec
    python installer/make_third_party_notices.py
"""

import importlib.metadata as md
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOC = ROOT / "build" / "zeus" / "PYZ-00.toc"
OUT = ROOT / "THIRD-PARTY-NOTICES.md"

HEADER = """# Third-party notices

ZEUS bundles the following packages. This file is generated from the build
output by `installer/make_third_party_notices.py` -- do not edit by hand.

ZEUS itself is distributed under the GNU General Public License v3; see
LICENSE. Two bundled libraries (pyimpspec, cvxopt) are GPLv3, which is what
requires the application as a whole to be GPLv3.

Qt is used under the LGPL v3 via PySide6. The bundle keeps Qt in separate DLLs
alongside the executable, so a recipient can replace them with their own build
of Qt, as the LGPL requires.

| Package | Version | License |
| --- | --- | --- |
"""


def bundled_top_level() -> set[str]:
    """Top-level module names present in the frozen archive."""
    text = TOC.read_text(encoding="utf-8", errors="replace")
    return {m.split(".")[0] for m in re.findall(r"\('([A-Za-z_][\w\.]*)',", text)}


def is_bundled(dist: str, present: set[str], name_to_dists) -> bool:
    """Whether `dist` really made it into the archive, judged on the
    distribution's main package: several projects ship extra top-level shims,
    and a stray one would list an excluded package as bundled.
    """
    owned = {n for n, dists in name_to_dists.items() if dist in dists}
    normalized = re.sub(r"[-_.]+", "_", dist).lower()
    primary = {n for n in owned if re.sub(r"[-_.]+", "_", n).lower() == normalized}
    # Projects whose import name differs from their distribution name (python-dateutil -> dateutil) have no primary, so any owned name counts.
    return bool((primary or owned) & present)


def license_of(meta) -> str:
    expr = (meta.get("License-Expression") or "").strip()
    if expr:
        return expr
    classifiers = [
        c.rsplit("::", 1)[-1].strip()
        for c in (meta.get_all("Classifier") or [])
        if c.startswith("License")
    ]
    if classifiers:
        return "; ".join(classifiers)
    # Some projects paste the whole licence into the field; only the first line belongs in a table.
    raw = (meta.get("License") or "").splitlines()
    return raw[0][:60] if raw else "see project"


def main() -> None:
    if not TOC.exists():
        raise SystemExit(f"{TOC} not found -- run `pyinstaller zeus.spec` first")

    present = bundled_top_level()
    name_to_dists = md.packages_distributions()

    rows = {}
    for name in present:
        for dist in name_to_dists.get(name, []):
            try:
                meta = md.metadata(dist)
            except Exception:
                continue
            if not is_bundled(dist, present, name_to_dists):
                continue
            rows[dist] = (meta.get("Version", "?"), license_of(meta))

    lines = [HEADER]
    for dist in sorted(rows, key=str.lower):
        version, lic = rows[dist]
        lines.append(f"| {dist} | {version} | {lic} |\n")
    lines.append(
        f"\n{len(rows)} bundled packages. Full licence texts ship with each "
        "package inside `_internal/`, and upstream sources are linked from "
        "each project's PyPI page.\n"
    )
    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)}: {len(rows)} packages")


if __name__ == "__main__":
    main()
