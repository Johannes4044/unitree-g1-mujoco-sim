# Contributing

## Set up

Python 3.11+. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras          # runtime deps + the dev tools (pytest, ruff)
```

or with pip:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"       # the dev tools: pytest, ruff
```

The dev tooling is declared twice on purpose — `[dependency-groups]` for `uv sync`
and `[project.optional-dependencies]` for `pip install -e ".[dev]"`. A test asserts
the two lists stay identical, so add new tools to both.

`docs/INSTALL.md` is the long version, including installing Python itself and an
offline install from a USB stick. Two hard constraints worth repeating here, because
they are not preferences:

- **macOS must be Apple Silicon and macOS 14+.** `mujoco` 3.12.0 publishes only
  `macosx_11_0_arm64` wheels and `onnxruntime` 1.30.0 only `macosx_14_0_arm64`.
  There is no Intel-macOS wheel for either, so an Intel Mac cannot install this.
- **Rendering on a headless Linux box needs an OpenGL context.** Set `MUJOCO_GL=egl`,
  or `MUJOCO_GL=osmesa` for pure software. Physics-only work needs neither.
  `MUJOCO_GL` is validated at `import mujoco`, and `egl`/`osmesa` are *Linux-only*
  values — setting either on macOS raises on import even if you never render. Leave
  it unset there.

## Run the tests and the linter

```bash
uv run pytest                 # the fast set - the default, and what CI gates on
uv run pytest -m slow         # only the slow set: rendering and long rollouts
uv run pytest -m ""           # everything
uv run ruff check . --exclude scripts --exclude workspace   # the lint gate
```

`pyproject.toml` sets `addopts = "-m 'not slow' --strict-markers"`, which is why a
bare `pytest` is the *fast* set and clearing the filter takes an empty marker
expression. Measured here: 176 passed in ~10 s fast, 186 in ~14 s for everything.

`ruff check .` is **not** clean, and that is expected. The findings all sit in
`scripts/` and `workspace/`; they came across with the extraction (see below) and are
a known baseline:

| ruff | findings | where |
|---|---|---|
| 0.13.3 – 0.15.x | 18 | `scripts/` 18, `workspace/` 0 (`E402` ×8, `E702` ×5, `E731`, `F401` ×2, …) |
| 0.16.0 – 0.16.10 | 9 | `scripts/` 6, `workspace/` 3 (`I001`, `C408`, `RUF046`, `RUF059`, `F401`) |
| any | 0 | everywhere else, `src/`, `tests/` and `conftest.py` included |

Measured version by version: the count is 18 up to and including 0.15.x and 9 from
0.16.0 on, because 0.16 changed ruff's *default* rule set — it dropped the
pycodestyle `E4`/`E7` checks that produced most of the 18 and added `I`, `C4` and
`RUF` rules that produce different ones. Nothing in this repo changed. If you have
seen the number "6" quoted for this tree, that is the `scripts/` subset under
0.16.x, not the whole count.

So the count is a function of the ruff version, `pyproject.toml` asks only for
`ruff>=0.5`, and a hard gate on `ruff check .` could therefore go red on a day
nobody touched the code. CI therefore gates on
the tree *minus those two directories*, which means new code anywhere — `src/`,
`tests/`, `conftest.py` — is gated from the moment it appears, and separately runs
`ruff check .` as an advisory step so the baseline stays visible. The excludes live
on the command line, not in `[tool.ruff]`. Clean `scripts/` and `workspace/` up and
drop them. Do not pay the debt down by loosening the ruff config.

Beyond the tests there is one end-to-end check, and it is the one that matters:

```bash
python scripts/check_scenarios.py     # expect `5/5 scenarios OK`
```

It compiles all five scenarios, settles physics, renders every camera and takes a
Mid-360 scan. It renders, so on headless Linux it needs `MUJOCO_GL` set.

## Layout

| | |
|---|---|
| `src/g1/robot/` | `RobotSource` (the interface), `SimSource` (MuJoCo), `RealSource` (the unimplemented seam), `lidar.py` (Mid-360 model) |
| `src/g1/wbc/` | whole-body controllers: `kinematic`, `holosoma` (ONNX), `pd_stand` |
| `models/` | the vendored scene `g1_pick_place.xml`, 122 meshes in `assets/`, compiled scenarios in `scenarios/`, provenance in `ASSETS.md` |
| `scenarios/` | the five scenario sources in the declarative DSL (`docs/scenario-dsl.md`) |
| `scripts/` | `build_scene.py`, `build_scenarios.py`, `check_scenarios.py`, `render_scene.py`, `smoke_test.py` |
| `workspace/` | the six commented examples; the actual point of the repo |
| `docs/` | `INSTALL.md`, `cheatsheet.md`, `scenario-dsl.md` |

## The one thing you cannot guess: this repo is an extraction

`g1-sim` is a stripped-down extract of a larger private project. **`src/g1/robot` and
`src/g1/wbc` exist there too, as independent copies.** They are not a submodule, not a
vendored dependency, and not generated — they are two files trees that happen to be the
same files.

That is why the package is called `g1` and not `g1_sim`: the import paths and the file
paths are identical on both sides, so a fix made in one place can be carried to the
other with a plain `diff -ru`. Please keep it that way:

- **Don't rename the package, and don't restructure `src/g1/robot` or `src/g1/wbc`**
  for tidiness alone. A rename here silently costs the other copy its diffability.
- **If you change behaviour in those two packages, say so in the commit message** in
  terms a person doing a `diff` will recognise. That person has no other signal.

Nothing enforces any of this. There is no CI check, no sync script and no shared
history — the two copies will drift unless somebody diffs them deliberately. Treat
divergence as a thing that will happen rather than a thing that might.

Code *outside* those two packages (`scripts/`, `scenarios/`, `workspace/`, `models/`,
`docs/`) is this repo's own and has no counterpart to stay compatible with.

## Regenerating the model is not a normal operation

`scripts/build_scene.py` **cannot run from a clone.** It needs three upstream
repositories that are deliberately not vendored (they are large and only needed to
regenerate), plus two packages that are not among this project's five dependencies:

```
MENAGERIE=...      google-deepmind/mujoco_menagerie  @ 8161bba
REVO2=...          BrainCoTech/revo2_description     @ 0ea3db7
UNITREE_EXTRA=...  unitreerobotics/unitree_ros       @ 7d6075f
pip install trimesh fast-simplification
```

It refuses to run without those variables. `models/g1_pick_place.xml` and
`models/assets/` are committed as *build output*, on purpose, so that one `git clone`
is enough on a machine with no network.

So regenerating meshes is a deliberate, pinned operation, not a build step: the commits
above are pinned because the output must stay reproducible and because every mesh has a
licence attached to its source. Read
[`models/ASSETS.md`](models/ASSETS.md) first — it maps every file in `models/assets/` to
the upstream repository, commit and licence it came from. If you regenerate, update that
table in the same commit.

`scripts/build_scenarios.py` is different: it compiles `scenarios/*.xml` into
`models/scenarios/` using nothing but this project's own dependencies, and you can run
it any time.

## Pull requests

- Keep the dependency list at five. The absences in `pyproject.toml` are deliberate and
  commented: no torch, no lerobot, no ffmpeg, no web framework. Anything copyleft is out.
- New third-party content needs a row in `THIRD-PARTY-NOTICES.md`, and new model assets
  need one in `models/ASSETS.md`.
- CI runs lint, the fast tests on three OS/Python legs, the full suite on Linux, and a
  clone-to-`5/5 scenarios OK` cold start with both documented install paths. The cold
  start is the promise of this repo; if it goes red, nothing else matters.

## Reporting a problem

Open an issue. If it is an install or run failure, the template asks for `python
--version`, `uname -srm`, your macOS version and `MUJOCO_GL` — those four distinguish
almost every real case. `python scripts/smoke_test.py` separates "physics is broken"
from "rendering is broken" and tells you which.
