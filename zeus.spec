# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for the ZEUS desktop app: `pyinstaller zeus.spec`.

Produces dist/ZEUS/ZEUS.exe. Onedir rather than onefile: the app already pays
several seconds of imports at startup (hence the splash in gui/app.py), and
onefile would add an unpack of the whole bundle to every launch on top of that.
"""

import tomllib
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH)

# pyproject.toml is the single source of the version: this writes it into the
# exe's resources, and installer/zeus.iss reads it back off the built exe.
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
COMPANY = "Aizad"
PRODUCT = "ZEUS"

# Windows wants a 4-part numeric version whatever the project uses.
_parts = tuple(int(p) for p in VERSION.split(".")) + (0, 0, 0, 0)
_vers = _parts[:4]

_version_file = ROOT / "build" / "version_info.txt"
_version_file.parent.mkdir(parents=True, exist_ok=True)
_version_file.write_text(
    f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={_vers}, prodvers={_vers}, mask=0x3f, flags=0x0,
                    OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', {COMPANY!r}),
      StringStruct('FileDescription', 'ZEUS - EIS batch analysis'),
      StringStruct('FileVersion', {VERSION!r}),
      StringStruct('InternalName', {PRODUCT!r}),
      StringStruct('LegalCopyright', 'Copyright (C) 2026 {COMPANY}. GPLv3.'),
      StringStruct('OriginalFilename', 'ZEUS.exe'),
      StringStruct('ProductName', {PRODUCT!r}),
      StringStruct('ProductVersion', {VERSION!r}),
    ])]),
    VarFileInfo([VarStruct('Translation', [0x0409, 1200])]),
  ],
)
""",
    encoding="utf-8",
)

# pyimpspec reaches some circuit elements and DRT methods through registries
# rather than plain imports, so sweep both packages in wholesale.
hiddenimports = collect_submodules("pyimpspec") + collect_submodules("schemdraw")

# cvxopt's extension modules load libopenblas.dll out of a sibling .libs
# directory, which the binary analysis does not follow on its own.
binaries = collect_dynamic_libs("cvxopt")

# gui/app.py resolves these as Path(__file__).parent / "assets", so the bundled
# copy has to keep the same relative position.
datas = [
    ("gui/assets", "gui/assets"),
    # GPLv3 and LGPLv3 both require the licence text to travel with the binary,
    # so it ships in the app folder rather than only in the installer.
    ("LICENSE", "."),
    ("THIRD-PARTY-NOTICES.md", "."),
]

# Nothing in core/ or gui/ imports matplotlib: schemdraw draws through its SVG
# backend and every plot is pyqtgraph. It arrives only as a lazy pyimpspec
# dependency, for plotting helpers this app never calls. Worth ~50 MB with PIL
# and fontTools. If a pyimpspec call ever raises ImportError, empty this list.
# pylab and mpl_toolkits are separate top-level modules from the matplotlib
# distribution, so excluding "matplotlib" alone leaves them behind as stubs
# that import a package no longer in the bundle.
UNUSED_PLOTTING = ["matplotlib", "pylab", "mpl_toolkits", "PIL", "fontTools"]

excludes = [
    "tkinter",
    # Build- and test-time only; never reached from gui.app.
    "pytest",
    "_pytest",
    "hatchling",
    "PyInstaller",
    "setuptools",
    "pkg_resources",
    # Guards against a second Qt binding being pulled in behind pyqtgraph.
    "PyQt5",
    "PyQt6",
    "PySide2",
    *UNUSED_PLOTTING,
]

a = Analysis(
    ["gui/app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ZEUS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="gui/assets/icon.ico",
    version=str(_version_file),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ZEUS",
)
