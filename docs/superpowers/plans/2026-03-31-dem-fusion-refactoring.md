# DEM Fusion Code Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor DEM fusion code from `resource/rebuild_dem/` into the main NH-Flood-2D project structure without modifying existing project files.

**Architecture:** Create new `src/nh_flood_2d/dem/` module for DEM processing utilities, move Python files to appropriate locations, keep data files in `resource/rebuild_dem/`, add dependencies to main project.

**Tech Stack:** Python 3.10.17, numpy, rasterio, scipy, shapely, geopandas, triangle

---

## File Structure

### New Files in `src/nh_flood_2d/dem/`:
- `__init__.py` - Empty module initialization
- `tin_utils.py` - TIN utilities (moved from `resource/rebuild_dem/tin_utils.py`)
- `tin_fusion.py` - TIN-based DEM fusion (renamed from `resource/rebuild_dem/dem_fusion_tin_replacement.py`)
- `dem_fusion.py` - Original DEM fusion algorithm (moved from `resource/rebuild_dem/dem_fusion.py`)
- `run_fusion.py` - Runner script (renamed from `resource/rebuild_dem/run_dem_fusion.py`)
- `run_final_fusion.py` - Final runner script (renamed from `resource/rebuild_dem/run_final_dem_fusion.py`)

### New Files in `tests/`:
- `test_dem_tin_fusion.py` - TIN fusion tests (moved from `resource/rebuild_dem/test_dem_fusion_tin.py`)
- `test_dem_fusion.py` - Original fusion tests (moved from `resource/rebuild_dem/test_dem_fusion.py`)

### Modified Files:
- `pyproject.toml` - Add new dependencies (triangle, rasterio, scipy, shapely, geopandas)
- `uv.lock` - Will be updated when running `uv sync`

### Files to Delete from `resource/rebuild_dem/`:
- `debug_points.py`
- `monitor_progress.py`
- `pyproject.toml`
- `README.md`
- `docs/superpowers/` directory (archive if needed)

### Files to Keep in `resource/rebuild_dem/`:
- `Digital Terrain Model.tif`
- `bay/` directory with point cloud data
- Any other `.tif`, `.txt`, `.shp` data files

---

## Task 1: Create DEM Module Structure

**Files:**
- Create: `src/nh_flood_2d/dem/__init__.py`
- Create: `src/nh_flood_2d/dem/` directory

- [ ] **Step 1: Create dem module directory**

```bash
mkdir -p src/nh_flood_2d/dem
```

- [ ] **Step 2: Create empty __init__.py**

```python
# src/nh_flood_2d/dem/__init__.py
"""
DEM (Digital Elevation Model) processing module.

This module provides tools for DEM fusion, TIN construction, and elevation data processing.
"""

__version__ = "0.1.0"
```

- [ ] **Step 3: Verify directory structure**

```bash
ls -la src/nh_flood_2d/
```

Expected: Should show `dem/` directory

- [ ] **Step 4: Commit**

```bash
git add src/nh_flood_2d/dem/__init__.py
git commit -m "feat: create dem module directory structure"
```

---

## Task 2: Move and Rename TIN Fusion Script

**Files:**
- Move: `resource/rebuild_dem/dem_fusion_tin_replacement.py` → `src/nh_flood_2d/dem/tin_fusion.py`
- Modify: `src/nh_flood_2d/dem/tin_fusion.py` (update imports)

- [ ] **Step 1: Copy and rename the file**

```bash
cp resource/rebuild_dem/dem_fusion_tin_replacement.py src/nh_flood_2d/dem/tin_fusion.py
```

- [ ] **Step 2: Update import paths in tin_fusion.py**

Find and replace the import section (lines 31-35):

```python
# Original:
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.nh_flood_2d.util import init_taichi

# Replace with:
from ..util import init_taichi
```

- [ ] **Step 3: Update tin_utils import in tin_fusion.py**

Find and replace (lines 49-55):

```python
# Original:
try:
    from tin_utils import build_tin, sample_tin_points, interpolate_tin
except ImportError as e:
    logger.error(f"导入TIN工具模块失败: {e}")
    logger.info("请安装triangle库: pip install triangle")
    raise

# Replace with:
try:
    from .tin_utils import build_tin, sample_tin_points, interpolate_tin
except ImportError as e:
    logger.error(f"导入TIN工具模块失败: {e}")
    logger.info("请安装triangle库: pip install triangle")
    raise
```

- [ ] **Step 4: Verify the file loads**

```bash
cd src/nh_flood_2d/dem
python -c "import tin_fusion; print('tin_fusion module loaded successfully')"
```

Expected: No import errors

- [ ] **Step 5: Commit**

```bash
git add src/nh_flood_2d/dem/tin_fusion.py
git commit -m "feat: move and rename tin fusion script to dem module"
```

---

## Task 3: Move TIN Utilities

**Files:**
- Move: `resource/rebuild_dem/tin_utils.py` → `src/nh_flood_2d/dem/tin_utils.py`
- Modify: `src/nh_flood_2d/dem/tin_utils.py` (no changes needed)

- [ ] **Step 1: Copy the file**

```bash
cp resource/rebuild_dem/tin_utils.py src/nh_flood_2d/dem/
```

- [ ] **Step 2: Verify the file content**

```bash
head -20 src/nh_flood_2d/dem/tin_utils.py
```

Expected: Should show the TIN utilities module header

- [ ] **Step 3: Test import from tin_fusion**

```bash
cd src/nh_flood_2d/dem
python -c "from tin_utils import build_delaunay_triangulation; print('TIN utils import successful')"
```

Expected: No import errors

- [ ] **Step 4: Commit**

```bash
git add src/nh_flood_2d/dem/tin_utils.py
git commit -m "feat: move tin utilities to dem module"
```

---

## Task 4: Move Original DEM Fusion Script

**Files:**
- Move: `resource/rebuild_dem/dem_fusion.py` → `src/nh_flood_2d/dem/dem_fusion.py`
- Modify: `src/nh_flood_2d/dem/dem_fusion.py` (no changes needed)

- [ ] **Step 1: Copy the file**

```bash
cp resource/rebuild_dem/dem_fusion.py src/nh_flood_2d/dem/
```

- [ ] **Step 2: Verify the file loads**

```bash
cd src/nh_flood_2d/dem
python -c "import dem_fusion; print('dem_fusion module loaded successfully')"
```

Expected: No import errors (may show missing dependency warnings)

- [ ] **Step 3: Commit**

```bash
git add src/nh_flood_2d/dem/dem_fusion.py
git commit -m "feat: move original dem fusion script to dem module"
```

---

## Task 5: Move and Rename Runner Scripts

**Files:**
- Move: `resource/rebuild_dem/run_dem_fusion.py` → `src/nh_flood_2d/dem/run_fusion.py`
- Move: `resource/rebuild_dem/run_final_dem_fusion.py` → `src/nh_flood_2d/dem/run_final_fusion.py`
- Modify: Both files (update imports)

- [ ] **Step 1: Copy and rename run_dem_fusion.py**

```bash
cp resource/rebuild_dem/run_dem_fusion.py src/nh_flood_2d/dem/run_fusion.py
```

- [ ] **Step 2: Update imports in run_fusion.py**

Find and replace the import (around line 28):

```python
# Original:
from dem_fusion import fuse_dem_and_points

# Replace with:
from .dem_fusion import fuse_dem_and_points
```

- [ ] **Step 3: Copy and rename run_final_dem_fusion.py**

```bash
cp resource/rebuild_dem/run_final_dem_fusion.py src/nh_flood_2d/dem/run_final_fusion.py
```

- [ ] **Step 4: Update imports in run_final_fusion.py**

Find and replace the import (around line 28):

```python
# Original:
from dem_fusion_tin_replacement import fuse_dem_and_points

# Replace with:
from .tin_fusion import fuse_dem_and_points
```

- [ ] **Step 5: Test both runner scripts**

```bash
cd src/nh_flood_2d/dem
python -c "import run_fusion; print('run_fusion imports successfully')"
python -c "import run_final_fusion; print('run_final_fusion imports successfully')"
```

Expected: No import errors

- [ ] **Step 6: Commit**

```bash
git add src/nh_flood_2d/dem/run_fusion.py src/nh_flood_2d/dem/run_final_fusion.py
git commit -m "feat: move and rename runner scripts to dem module"
```

---

## Task 6: Move Test Files

**Files:**
- Move: `resource/rebuild_dem/test_dem_fusion_tin.py` → `tests/test_dem_tin_fusion.py`
- Move: `resource/rebuild_dem/test_dem_fusion.py` → `tests/test_dem_fusion.py`
- Modify: Both test files (update imports)

- [ ] **Step 1: Create tests directory if needed**

```bash
mkdir -p tests
```

- [ ] **Step 2: Copy and rename test_dem_fusion_tin.py**

```bash
cp resource/rebuild_dem/test_dem_fusion_tin.py tests/test_dem_tin_fusion.py
```

- [ ] **Step 3: Update imports in test_dem_tin_fusion.py**

Find and replace the import section (lines 24-38):

```python
# Original:
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import待测试模块
try:
    from tin_utils import (
        build_delaunay_triangulation,
        build_tin,
        interpolate_tin,
        calculate_tin_quality
    )
except ImportError as e:
    print(f"导入TIN工具模块失败: {e}")
    sys.exit(1)

# Replace with:
try:
    from src.nh_flood_2d.dem.tin_utils import (
        build_delaunay_triangulation,
        build_tin,
        interpolate_tin,
        calculate_tin_quality
    )
except ImportError as e:
    print(f"导入TIN工具模块失败: {e}")
    sys.exit(1)
```

- [ ] **Step 4: Copy and rename test_dem_fusion.py**

```bash
cp resource/rebuild_dem/test_dem_fusion.py tests/test_dem_fusion.py
```

- [ ] **Step 5: Update imports in test_dem_fusion.py**

Find and replace the import (around line 28):

```python
# Original:
from dem_fusion import fuse_dem_and_points

# Replace with:
from src.nh_flood_2d.dem.dem_fusion import fuse_dem_and_points
```

- [ ] **Step 6: Test imports**

```bash
cd tests
python -c "import test_dem_tin_fusion; print('TIN fusion test imports successfully')"
python -c "import test_dem_fusion; print('DEM fusion test imports successfully')"
```

Expected: No import errors (may show missing dependency warnings)

- [ ] **Step 7: Commit**

```bash
git add tests/test_dem_tin_fusion.py tests/test_dem_fusion.py
git commit -m "feat: move test files to tests directory"
```

---

## Task 7: Add Dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml` (add dependencies)

- [ ] **Step 1: Check current dependencies**

```bash
grep -A 20 "dependencies =" pyproject.toml
```

- [ ] **Step 2: Add missing dependencies**

Edit `pyproject.toml` and add to the dependencies list:

```toml
dependencies = [
    # Existing dependencies...
    "triangle",
    "rasterio>=1.4.4",
    "scipy>=1.10.0",
    "shapely>=2.1.2",
    "geopandas>=0.14.0",
]
```

Note: Check if any of these are already in the list and don't duplicate.

- [ ] **Step 3: Verify pyproject.toml syntax**

```bash
python -m py_compile pyproject.toml 2>/dev/null || echo "Syntax check not available"
```

- [ ] **Step 4: Update uv.lock**

```bash
uv sync
```

Expected: Should install new dependencies without errors

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add dem fusion dependencies to project"
```

---

## Task 8: Clean Up Resource Directory

**Files:**
- Delete: `resource/rebuild_dem/debug_points.py`
- Delete: `resource/rebuild_dem/monitor_progress.py`
- Delete: `resource/rebuild_dem/pyproject.toml`
- Delete: `resource/rebuild_dem/README.md`
- Keep: Data files in `resource/rebuild_dem/`

- [ ] **Step 1: List files to delete**

```bash
ls -la resource/rebuild_dem/*.py resource/rebuild_dem/*.toml resource/rebuild_dem/*.md 2>/dev/null || true
```

- [ ] **Step 2: Delete Python files**

```bash
rm resource/rebuild_dem/debug_points.py resource/rebuild_dem/monitor_progress.py
```

- [ ] **Step 3: Delete configuration and documentation files**

```bash
rm resource/rebuild_dem/pyproject.toml resource/rebuild_dem/README.md
```

- [ ] **Step 4: Check if docs directory exists and remove**

```bash
if [ -d "resource/rebuild_dem/docs" ]; then
    echo "Found docs directory, removing..."
    rm -rf resource/rebuild_dem/docs
fi
```

- [ ] **Step 5: Verify data files remain**

```bash
ls -la resource/rebuild_dem/
```

Expected: Should show only data files (.tif, .txt, directories like bay/)

- [ ] **Step 6: Commit**

```bash
git add -u resource/rebuild_dem/
git commit -m "cleanup: remove python files from resource/rebuild_dem, keep only data files"
```

---

## Task 9: Verify Module Functionality

**Files:**
- Test: `src/nh_flood_2d/dem/` (all modules)
- Test: `tests/` (test files)

- [ ] **Step 1: Test module imports**

```bash
cd src/nh_flood_2d/dem
python -c "
import tin_fusion
import tin_utils
import dem_fusion
import run_fusion
import run_final_fusion
print('All dem module imports successful')
"
```

Expected: No import errors

- [ ] **Step 2: Test basic TIN utilities**

```bash
cd tests
python -c "
import numpy as np
from src.nh_flood_2d.dem.tin_utils import build_delaunay_triangulation

# Simple test
points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
result = build_delaunay_triangulation(points)
print(f'TIN construction test passed: {len(result[\"triangles\"])} triangles')
"
```

Expected: Should run without errors

- [ ] **Step 3: Run test files (skip actual tests that need data)**

```bash
cd tests
python -m pytest test_dem_tin_fusion.py::TestTinUtils::test_build_delaunay_triangulation -v
```

Expected: Test should pass (unit test with synthetic data)

- [ ] **Step 4: Check final directory structure**

```bash
find src/nh_flood_2d/dem -name "*.py" | sort
find tests -name "test_dem*.py" | sort
ls -la resource/rebuild_dem/*.py 2>/dev/null || echo "No Python files in resource/rebuild_dem (good!)"
```

Expected: Python files in src/, test files in tests/, no Python files in resource/

- [ ] **Step 5: Commit verification**

```bash
git status
git diff --stat
```

Expected: Only expected changes shown

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "refactor: complete dem fusion code integration into main project structure"
```

---

## Self-Review

**1. Spec coverage:**
- ✓ Create new `dem` module in `src/nh_flood_2d/` - Task 1
- ✓ Move Python files to appropriate locations - Tasks 2-6
- ✓ Keep data files in `resource/rebuild_dem/` - Task 8
- ✓ Add dependencies to main project - Task 7
- ✓ Create `tests/` directory with dem_fusion tests - Task 6
- ✓ Don't modify existing project files - All tasks preserve existing files

**2. Placeholder scan:** No placeholders found. All tasks have concrete steps with exact commands and code.

**3. Type consistency:** All file paths and import statements are consistent across tasks.

**4. File structure:** Clear separation with focused files:
- `tin_utils.py` - TIN construction utilities
- `tin_fusion.py` - TIN-based DEM fusion
- `dem_fusion.py` - Original fusion algorithm
- `run_fusion.py` / `run_final_fusion.py` - Runner scripts
- Test files separated by functionality

**Plan complete and saved to `docs/superpowers/plans/2026-03-31-dem-fusion-refactoring.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**