"""Fix macOS code signing for bundled dylibs (e.g. swmm-toolkit).

On macOS (Apple Silicon), some pip wheels ship dylibs with an invalid or
modified code signature. macOS security enforcement will SIGKILL the process
(exit code 137, no error message) when it tries to load such a library.

Run once after `uv sync`:
    uv run fix-macos-codesign
"""

import subprocess
import sys
from pathlib import Path


_PACKAGES_TO_FIX = [
    "swmm/toolkit",
]


def _check_signature(path: Path) -> bool:
    """Return True if the dylib signature is valid (or unsigned but ok)."""
    result = subprocess.run(
        ["codesign", "-vv", str(path)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and "invalid signature" not in result.stderr


def _resign(path: Path) -> bool:
    """Re-sign with an ad-hoc signature. Returns True on success."""
    result = subprocess.run(
        ["codesign", "--force", "--sign", "-", str(path)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def fix_codesign() -> None:
    if sys.platform != "darwin":
        print("Not on macOS — nothing to do.")
        return

    import site

    site_packages = Path(site.getsitepackages()[0])

    fixed: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []

    for pkg_subdir in _PACKAGES_TO_FIX:
        pkg_dir = site_packages / pkg_subdir
        if not pkg_dir.exists():
            print(f"[skip] {pkg_subdir} not found in site-packages")
            continue

        for dylib in sorted(pkg_dir.glob("*.dylib")):
            if _check_signature(dylib):
                skipped.append(dylib.name)
            elif _resign(dylib):
                fixed.append(dylib.name)
                print(f"[fixed] {dylib.name}")
            else:
                failed.append(dylib.name)
                print(f"[FAIL]  {dylib.name}")

    print()
    print(f"Fixed: {len(fixed)}  |  Already valid: {len(skipped)}  |  Failed: {len(failed)}")
    if failed:
        sys.exit(1)


def main() -> None:
    fix_codesign()


if __name__ == "__main__":
    main()
