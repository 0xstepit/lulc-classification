# Engineering Review — Outstanding Items

Findings from a full-repository engineering review, reduced to what is **still
open**. Docstring and test-coverage gaps are deliberately out of scope; this
file tracks engineering practice only.

Every claim below was verified against the working tree rather than inferred.
File references are `path:line` at the time of writing.

______________________________________________________________________

## Already closed

Recorded so nobody redoes them:

- CI invoked bare `ruff` and failed at exit 127 on every run — both jobs now use
  `uv run`, and `uv sync --frozen` fails the build on a stale lockfile.
- Ruff's `select` had replaced the default ruleset, silently disabling pyflakes
  (41 unused imports, 9 empty f-strings, 17 undefined names went unreported).
  The ruleset is now explicit and broad.
- `ReportEntry.timestamp` evaluated `datetime.now()` once at class-creation
  time, so every entry in a run shared one timestamp. Now `default_factory`.
- Pre-commit did not run ruff at all.
- The project was not installable: no `[build-system]`, imports working only via
  `PYTHONPATH=.` in the Makefile and `pythonpath = ["."]` in pytest. Now a real
  `src/lulc` package with a hatchling backend; both path hacks deleted.
- Two layering violations: `lulc.reporter` imported `lulc.data.sentinel2`, and
  `lulc.io` sat above `lulc.config` via `lulc.constants`. A dependency-free
  `lulc/domain.py` now holds the shared value objects. `import lulc.reporter`
  dropped from pulling
  `boto3, pystac, pystac_client, rasterio, tenacity, urllib3, numpy` to just
  `numpy`.
- `io.ROOT` silently pointed at `src/` after the package move, so every config
  path resolved wrong. Fixed to `parents[2]`.
- Pyright configured against the project venv and pinned as a dev dependency
  (the Homebrew `1.1.328` was reporting phantom errors a modern build does not).

______________________________________________________________________

## P1 — Correctness and design

### 1. `MSIConfig.num_bands` is not a dataclass field

`src/lulc/config/config.py:141` assigns `self.num_bands` inside `__post_init__`
without declaring it. It is therefore invisible to `dataclasses.asdict()`,
`replace()` and `__repr__` — and `asdict()` is exactly what
`reporter/writers.py:_json` uses to serialise. Any config snapshot written into
a report, or into a Phase 6 checkpoint, silently drops it.

**Fix:** make it a `@property`. While there, note `MSIConfig` and `AoiConfig`
are the only non-frozen configs in the package; the same `__post_init__` also
mutates `self.bands` after validation.

### 2. `get_scl_band_index()` returns a hardcoded `10`

`src/lulc/config/config.py:175`. CLAUDE.md flags band-order drift as the
cheapest guard worth having, because a mismatch computes the spectral indices
from the wrong channels and nothing downstream reveals it. The value is
derivable from `get_bands_list()`. `constants.py:seasonal_band_names` has the
same smell — a substring match on `"scl"` with a comment admitting it.

**Fix:** derive both from one source of truth, and add the assertion CLAUDE.md
already proposes:
`band_names.index("nir") == cfg.indices.get_channel("ndvi", "nir")`.

### 3. Normalization is recomputed on every `__getitem__`

`src/lulc/preprocessing/normalization.py:48` — `normalized_median` is a
property, so each call runs `np.clip` plus `_scale` across 52 channels; the
guarded reciprocal is recomputed too, and three full `[52, 256, 256]` float32
buffers (~13 MB each) are allocated per patch. None of it depends on the patch.
Per patch, per epoch, per worker, on the dataloader critical path.

**Fix:** precompute `(offset, inv_range, fill)` once in `__post_init__` or via a
cached property, leaving `__getitem__` with one clip, one fused multiply-add and
one `np.where`. Worth profiling before and after so the number can go in the
README.

### 4. Three hand-rolled TOML loaders

`load_config`, `load_reporter_config` and `load_viz_config` each re-implement
exists-check → `tomllib.load` → manual `**kwargs` splat. The int-key coercion
workaround appears four times as
`object.__setattr__(self, ..., {int(k): v ...})`. Failure modes are poor: an
unknown TOML key raises a bare `TypeError`, a missing one a `KeyError` with no
field path.

CLAUDE.md already commits to Hydra + OmegaConf for Phase 6. **Do not write a
fourth loader for training config.**

- **Pydantic v2 + pydantic-settings** — validation, automatic `dict[int, int]`
  key coercion, field-path error messages, JSON-Schema export. Deletes roughly
  150 lines including all four `object.__setattr__` blocks.
- **Hydra structured configs** — CLI overrides (`lulc-train model.lr=3e-4`),
  multirun sweeps, per-run output directories.

They compose: Hydra for composition and CLI, Pydantic for validating the
composed result. Worth naming that choice in the README.

**Caveat:** Pydantic evaluates annotations at model-build time, so a field type
hidden behind `if TYPE_CHECKING:` will fail unless `model_rebuild()` is called.
Relevant to `lulc/constants.py`.

### 5. `compute_nan_pct` returns `np.float16`

`src/lulc/data/utils.py:31,35`. Roughly three significant decimal digits, for a
number serialised into the reports as a data-quality statistic. Return `float`;
the memory saving is zero.

______________________________________________________________________

## P1 — Repository hygiene

### 6. `outputs/` is untracked by `.gitignore`; figures are ignored

`.gitignore:19` — `**/images/**` ignores `assets/images/`, which is where
`viz.images.store_figure` writes. Phase 8's deliverable is a README with
embedded figures, so those must be tracked. Meanwhile `outputs/` matches no
rule, so the first `05_train_unet.py` run will offer git a directory of
checkpoints.

```gitignore
data/raw/
data/processed/
data/labels/
data/patches/
outputs/
!outputs/figures/          # README figures are deliverables
mlruns/
```

Prefer anchored patterns over `**/raw/**`, which matches any directory named
`raw` anywhere in the tree.

### 7. Pipeline outputs are committed

Five files under `data/reports/` are tracked. These are run artifacts: they
churn on every re-run, conflict on merge, and are what DVC (planned for Phase
10\) or MLflow artifacts exist to hold. This gets worse in Phase 6 when per-run
metrics appear.

**Fix.** Untrack without deleting, then bring them under DVC:

```bash
git rm --cached data/reports/*.json
printf 'data/reports/\n' >> .gitignore

uv add --dev dvc
uv run dvc init
uv run dvc add data/reports          # writes data/reports.dvc, which IS tracked
git add data/reports.dvc .gitignore .dvc
```

If DVC is too big a step before Phase 10, the interim is just the first two
lines — untracked artifacts beat churning ones.

### 8. `src/lulc/data/rasterio.py` shadows the third-party `rasterio`

Resolves correctly under absolute imports, but it is a trap for the next reader.

**Fix.** Only two importers, so the rename is contained:

```bash
git mv src/lulc/data/rasterio.py src/lulc/data/raster_io.py
# update: scripts/02_create_seasonal_composite.py, scripts/04_extract_patches.py
```

Ruff's `TID` rules (already enabled) will flag nothing here, so grep to confirm:
`grep -rn "data.rasterio" --include='*.py' .` must come back empty.

______________________________________________________________________

## P2 — Practice

### 9. Import-time side effects in every script

`setup_logging()` and `load_dotenv()` run at module import, outside `main()`, at
seven sites: `00_select_aoi.py:24`, `01_download_sentinel2.py:24,26`,
`02_create_seasonal_composite.py:38,40`, `03_download_world_cover.py:31`,
`04_extract_patches.py:46`.

Same class of problem as the `mkdir` removed from `viz/images.py`. Importing any
of these scripts — which testing and the Phase 10 DVC stages both need —
reconfigures root logging and reads `.env` as a side effect.

**Fix.** Move both inside `main()`, and take the logger at module scope without
configuring it:

```python
import logging

logger = logging.getLogger(Path(__file__).stem)


def main() -> None:
    load_dotenv()
    setup_logging()
    ...


if __name__ == "__main__":
    main()
```

The module-level `logging.getLogger(...)` is fine — it only *creates* a logger
object. It is `setup_logging()`, which installs handlers on the root, that must
not run on import.

### 10. Nine open pyright errors — CLOSED

Resolved with eight narrow, rule-specific suppressions. `uv run pyright` now
reports **52 files, 0 errors, 0 warnings**.

> **Trap worth recording.** If pyright suddenly reports ~69 errors, check that
> `[tool.pyright]` in `pyproject.toml` is not commented out. Without a config
> block pyright analyses the entire project root, sweeping in the 15 jupytext
> files under `notebooks/` — exploratory cells with genuinely undefined names —
> and `filesAnalyzed` jumps 52 → 67. The `include` key is what keeps the scope
> honest. Wiring the type check into CI (item 11) turns this into an immediate,
> obvious failure instead of a confusing local one.

The original triage, kept for the reasoning: **six were third-party stub gaps,
three were intentional**.

- `scripts/04_extract_patches.py:165-168` and `src/lulc/data/patches.py:325` —
  `Window(col_off=..., ...)`. Verified at runtime: rasterio 1.5.0 ships **no
  `py.typed`**, and `inspect.signature(Window.__init__)` is
  `(self, col_off, row_off, width, height)`. Both call forms work. Pyright
  cannot see the generated `__init__`. Suppress narrowly with
  `# pyright: ignore[reportCallIssue]` and a comment naming the reason.
- `src/lulc/data/worldcover.py:43` — `.values` on a GeoDataFrame column; same
  category, imprecise geopandas stubs.
- `tests/test_augmentation.py:46` and `tests/test_unet.py:33` — deliberate
  frozen-dataclass assignments inside `pytest.raises(FrozenInstanceError)`. The
  existing comment in `test_unet.py` names `reportGeneralTypeIssues`, which
  pyright has since split. The correct rule for both, confirmed against
  `--outputjson`, is **`reportAttributeAccessIssue`**.
- `tests/test_viz_config.py:95` — passes `dict[str, str]` where the annotation
  says `dict[int, str]`, to exercise the coercion. The type model is the thing
  that is wrong here; item 4 fixes it properly by moving coercion into the
  loader.

**Fix.** Six narrow suppressions, each carrying its reason:

```python
# rasterio 1.5 ships no py.typed and Window's __init__ is generated, so pyright
# cannot see its parameters. Verified at runtime:
#   inspect.signature(Window.__init__) -> (self, col_off, row_off, width, height)
window = Window(  # pyright: ignore[reportCallIssue]
    col_off=block_col * block_size,
    ...
)
```

```python
# Asserting the dataclass is frozen; the assignment is the subject of the test.
UNetConfig().num_classes = 3  # pyright: ignore[reportAttributeAccessIssue]
```

Then wire `uv run pyright` into CI (item 11) so the count cannot drift back up.

### 11. CI is thin

Single OS, single Python, no coverage, no dependency cache, no type-check job.

- Add a `pyright` job (`uv run pyright`) now that it is pinned and clean at
  `standard`.
- Matrix over `[3.12, 3.13, 3.14]` — this also tests whether
  `requires-python = ">=3.14"` is genuinely necessary. It is an aggressive floor
  for a geospatial stack where GDAL wheels lag, and it blocks reproduction.
- Cache the uv download dir; add coverage reporting.

**Fix.** Collapse the two workflows into one `ci.yaml` with three jobs:

```yaml
name: CI
on:
  push: { branches: [main] }
  pull_request: { branches: [main] }

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with: { enable-cache: true }
      - run: uv sync --frozen
      - run: uv run ruff check --output-format=github .
      - run: uv run ruff format --check .
      - run: uv run pyright

  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        python: ["3.14"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with: { enable-cache: true }
      - run: uv sync --frozen
      - run: uv run pytest tests/ -v --cov=lulc --cov-report=term-missing
```

`macos-latest` is worth the minute: the project is developed on Apple silicon
with MPS, and it is the platform most likely to diverge. Widen the `python`
matrix only if you first lower `requires-python` — as it stands, `>=3.14` makes
a matrix meaningless.

### 12. No `tests/conftest.py`

No shared fixtures, no markers, no coverage configuration.

**Fix.** Create it with the fixtures the existing tests already rebuild by hand,
plus a marker for the tests that touch real rasters:

```python
"""Shared fixtures. Keep IO-free fixtures here; put raster fixtures behind a marker."""

import numpy as np
import pytest

from lulc.config import load_config
from lulc.io import GLOBAL_CONFIG


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_data: needs data/ on disk")


@pytest.fixture(scope="session")
def cfg():
    return load_config(GLOBAL_CONFIG)


@pytest.fixture
def rng():
    """Seeded generator — never let a test depend on global numpy state."""
    return np.random.default_rng(42)
```

```toml
[tool.pytest.ini_options]
addopts = "--strict-markers -ra"
testpaths = ["tests"]
```

`--strict-markers` turns a typo'd marker into an error rather than a silently
skipped test.

### 13. Add the dataclass-default convention check

The `ReportEntry` timestamp bug is invisible to both ruff and pyright: `RUF009`
resolves the callee to a dotted name, and one trailing `.isoformat()` defeats
it. Verified — no linter catches the pattern.

An `ast` sweep does, and when run against `src/` it found exactly one offender —
the real bug — with no false positives. The rule it enforces: **no call
expressions in dataclass field defaults, `field(...)` excepted.**

**Fix.** Save as `tools/check_defaults.py`:

```python
"""AST sweep: no dataclass field default may be a call expression.

Function calls in dataclass defaults are evaluated once, at class-creation time,
and the result is shared by every instance. Ruff's RUF009 catches only the case
where the callee resolves to a known dotted name, so `datetime.now()` is flagged
but `datetime.now(UTC).isoformat()` is not. This closes that gap.
"""

import ast
import sys
from collections.abc import Iterator
from pathlib import Path


def _is_dataclass(node: ast.ClassDef) -> bool:
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        name = (
            target.attr
            if isinstance(target, ast.Attribute)
            else getattr(target, "id", "")
        )
        if name == "dataclass":
            return True
    return False


def offenders(root: Path) -> Iterator[tuple[Path, int, str, str]]:
    """Yield (path, line, class name, source) for each offending default."""
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ClassDef) and _is_dataclass(node)):
                continue
            for stmt in node.body:
                if not (
                    isinstance(stmt, ast.AnnAssign) and isinstance(stmt.value, ast.Call)
                ):
                    continue
                # field(...) is the sanctioned escape hatch.
                fn = stmt.value.func
                if getattr(fn, "id", getattr(fn, "attr", "")) == "field":
                    continue
                yield path, stmt.lineno, node.name, ast.unparse(stmt)


def main() -> int:
    found = list(offenders(Path(sys.argv[1] if len(sys.argv) > 1 else "src")))
    for path, line, cls, src in found:
        print(f"{path}:{line}: {cls}: {src}")
    if found:
        print(f"\n{len(found)} offender(s); use field(default_factory=...)")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Wire it into `tests/test_conventions.py`:

```python
def test_no_call_expressions_in_dataclass_defaults():
    found = list(offenders(Path(__file__).parents[1] / "src"))
    assert not found, "use field(default_factory=...) instead:\n" + "\n".join(
        f"  {p}:{line}: {src}" for p, line, _, src in found
    )
```

and into pre-commit:

```yaml
- repo: local
  hooks:
    - id: dataclass-defaults
      name: no call expressions in dataclass defaults
      entry: python tools/check_defaults.py src
      language: system
      pass_filenames: false
```

`tests/test_conventions.py` is also the home for the band-order assertion from
item 2 — both are project invariants that no general-purpose tool can express.

### 14. Everything lands on `main`

No branches, no pull requests — so the `on: pull_request` half of both workflows
has never run. For a portfolio repository, three or four real PRs with
descriptions and green checks are a stronger signal than the same commits pushed
directly. Also `improvement:` is not a Conventional Commits type; `refactor:`
and `perf:` are the intended ones.

### 15. Optional — the CLI layer

Deferred from the packaging work. Moving each script's `main()` into
`src/lulc/cli/<name>.py` enables `[project.scripts]`, giving
`uv run lulc-extract-patches` from any directory, with `scripts/NN_*.py` kept as
three-line shims so the numbered pipeline stays readable. Worth doing when Phase
10 adds `dvc.yaml`, where `cmd: lulc-extract-patches` reads better than a path.

______________________________________________________________________

## Suggested order

1. Items 6 and 7 — before the first training run writes to `outputs/`.
1. Items 1, 2, 5, 9 — small, local, independent.
1. Item 13 — cheap, and prevents the recurrence of a bug class the linters miss.
1. Item 3 — profile first, then fix; the number belongs in the README.
1. Items 10, 11, 12 — get the type checker and CI green and keep them green.
1. Item 4 — fold into Phase 6 rather than doing it as standalone churn.
1. Items 8, 14, 15 — opportunistic.
