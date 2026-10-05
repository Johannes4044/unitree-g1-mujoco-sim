# The scenario DSL

Reference for `scenarios/*.xml` and `models/objects/*.xml`. The short version — why scenarios
exist and how to add one — is in `workspace/README.md`; this is the full element list, the
randomisation attributes and what the build and check scripts actually enforce.

```
scenarios/*.xml              the declarative scenarios, ~40 lines each
models/objects/*.xml         the object library: one standalone MuJoCo model per part
models/scenarios/*.xml       the build output that SimSource loads
scripts/build_scenarios.py   scenario + library + robot -> a loadable model
scripts/check_scenarios.py   loads every built model through SimSource and proves it works
```

```bash
uv run python scripts/build_scenarios.py              # build all five
uv run python scripts/build_scenarios.py --seed 3     # + a seeded variant, <name>_s3.xml
uv run python scripts/check_scenarios.py              # verify all five
uv run python scripts/check_scenarios.py --png        # + write out/<name>_*.png
```

## The five scenarios

| Scenario | What it is for |
|---|---|
| `pick_place_table` | Baseline VLA pick-and-place. One work table with named objects and a tote on a packing table 2.5 m away. Same table geometry and heights as `models/g1_pick_place.xml`, so results carry over. |
| `cluttered_bench` | Grasping in clutter and language grounding: eight similar-sized objects within one arm's reach on a 0.9 × 1.8 m bench, palette-randomised colours, optional distractors. No locomotion. |
| `house_rooms` | Walking through a house: three rooms, two doorways (1.00 m and 0.95 m clear), furniture. The pick surface is a kitchen counter five metres away, so an episode must include locomotion before the grasp. LiDAR sees a closed room shape. |
| `warehouse_aisle` | Aisle navigation and crate handling: two runs of 1.8 m shelving forming a 2.8 m aisle (the degenerate case for LiDAR odometry), pallets and heavy crates as obstacles, and one free 1.2 kg crate on the floor for a bimanual floor pick. |
| `rubble_yard` | Uneven ground: a 5 × 5 m field of tiles 1–8 cm high tilted up to 3°, planks and bricks lying on it, two kerb steps, and a field bench past the rubble. |

Cost is dominated by **contacts, not geoms**: `rubble_yard` has 280 geoms and steps as cheaply as
the 164-geom table scene, while `cluttered_bench` is the most expensive because eight objects
rest on a bench. Expect 1.1×–1.8× the base scene's step cost. Rendering is unchanged within
noise, because everything added is untextured primitives.

## Scenario elements

A scenario is `<scenario name="...">` containing any of:

| Element | Meaning |
|---|---|
| `<doc>` | Free text: the purpose. Copied into a comment at the top of the generated model. |
| `<robot x y yaw/>` | Move the robot's spawn (default: origin, facing +x). Moves both the `stand` keyframe pelvis pose and the `base_mocap` weld target. |
| `<floor rgba="r g b a"/>` | Recolour the ground plane (default: the checkerboard from the robot model). |
| `<wall name from="x y" to="x y" height thickness rgba door="centre width" door_height/>` | A wall between two floor points. `door` cuts a doorway `width` wide whose centre is `centre` metres along the wall from `from`, and closes the gap above it with a lintel. |
| `<box name size="hx hy hz" pos yaw rgba/>` | A static box: kerbs, steps, pillars. |
| `<terrain name area="x0 y0 x1 y1" tile height tilt keepout="x y r" seed rgba/>` | A grid of static tiles of random height (1 cm to `height`) and tilt, for uneven ground. `keepout` leaves a flat circle where the robot spawns. |
| `<part type name pos yaw rgba .../>` | A **static** instance of `models/objects/<type>.xml`. A free joint in the part is stripped: `<part>` means scenery. |
| `<object type name pos yaw rgba .../>` | A **free** instance of `models/objects/<type>.xml`. The `name` is the body name — i.e. the VLA prompt vocabulary. |
| `<camera name pos target/xyaxes fovy/>` | Replaces a same-named world camera. `third_person` is the one most tooling renders. |

`pos` takes 2 or 3 numbers (z defaults to 0) in world metres; `yaw` is radians about +z. Object
origins are at their **bottom face**, so `pos="0.6 0.1 0.75"` sits exactly on a 0.75 m tabletop.
`on="<terrain name>"` makes `z` relative to the terrain surface under `(x, y)` instead of the floor.

**Every scenario needs a table-like `<part>` called `pick_table`.** `SimSource` looks for the geom
`pick_table_top` as its randomisation surface; without it the build fails loudly rather than
letting the simulator fall back to hard-coded constants.

Attachment uses MuJoCo's `MjSpec.attach` with a name prefix, so the same part file can be
instantiated any number of times without collisions: every geom, site, joint and material inside
is prefixed with the instance name, and the top body is renamed to exactly the instance name.

## The object library

Manipulable (free joint): `crate` (1.20 kg, cut-out handles), `box_parcel` (0.60 kg),
`cube` (0.12 kg), `ball` (0.15 kg), `can` (0.36 kg), `bottle` (0.55 kg), `hammer` (0.55 kg),
`wrench` (0.25 kg), `screwdriver` (0.09 kg), `brick` (2.3 kg), `plank` (2.0 kg).

Static: `work_table` (0.75 m surface), `packing_bench` (0.75 m, 0.9 × 1.8 m), `counter` (0.90 m),
`shelf_unit` (1.8 m, four decks), `pallet`, `tote` (place-into target, site `drop`), `chair`,
`crate_heavy` (18 kg obstacle).

Conventions a new object must follow: exactly one body in `<worldbody>`; origin at the bottom
face, centred in x and y; a material named `mat` for the main colour; table-like parts name their
working surface geom `top`. Give it real mass and friction — inertia is derived from the geom
layout, which is why the crate is panels rather than a solid box.

Friction is tuned per material class rather than left at the default: cardboard and timber
0.9–1.1, plastic 0.8, metal and aluminium 0.7, all with `condim=4` so torsional friction resists
a grasped object spinning in the fingers. Contacts use the robot model's elliptic cone and
`impratio=10`. Note that `condim=4` means MuJoCo ignores the rolling friction coefficient, so a
perfectly round object that gets nudged rolls until it leaves the table — the screwdriver's grip
is a flat-sided box for that reason, and `can` / `bottle` / `ball` are only stable because they
are placed upright and undisturbed.

Every file in `models/objects/` is a self-authored MuJoCo primitive — no mesh, no texture,
nothing to licence-clear. The only external assets in a built scenario are the ones already in
`models/g1_pick_place.xml` (see `models/ASSETS.md`). Nothing is downloaded at build or run time.

## Randomisation

Two layers, because they live at different times.

### Runtime, per episode — `SimSource.sim_randomise`

```python
info = sim.sim_randomise(seed=7)          # -> {"seed", "objects", "skipped", "placed", ...}
```

Deterministic for a seed; `seed=None` draws a fresh one and returns it, so an interesting
episode can be replayed exactly. Four independent axes, each separately seeded:

| axis | default | effect |
|---|---|---|
| `pose` | on | re-place on the pick surface: uniform xy (8 cm apart, 5 cm from the edge, both growing with the object's own radius), uniform yaw, upright, at the object's own rest height above its support |
| `colour` | off | ±0.15 per RGB channel around the object's own colour — bounded on purpose, so `red_box` stays recognisably red and a language-conditioned policy still has a referent |
| `mass` | off | ±10 % per object, inertia scaled with it |
| `friction` | off | ±20 % on all three friction coefficients, per geom |

`colour`, `mass` and `friction` change the *model*, so they persist until `sim_reset` restores
them. None of them add per-step cost.

Which objects move:

- `scope="surface"` (default) — the free bodies that *belong* on the pick surface: inside its
  footprint and above its top, either now or in the scene's initial pose. So
  `warehouse_aisle`'s floor crate and `rubble_yard`'s loose bricks stay where they are, and a
  cube knocked onto the floor mid-episode still comes back on the next randomisation.
- `objects=["green_cube", ...]` — exactly these, wherever they are.
- `scope="all"` — every free body.

Footprints honour `geom_xmat`, so a turned surface is measured in world axes.

Two scenario-side rules follow, and `check_scenarios.py` enforces both: **keep the tote off the
pick surface** (randomisation samples the whole surface, so a place-into target standing on it is
occasionally handed the object at episode start), and **leave room for the jitter**.

### Build time, per variant — `--seed N`

Produces `<name>_s<N>.xml` alongside the nominal model, and covers what the runtime hook cannot:

| Attribute | Effect (only with `--seed`) |
|---|---|
| `jitter="0.05"` | ±5 cm uniform on x and y |
| `yaw_jitter="3.14"` | ± that many radians of yaw |
| `optional="0.5"` | the part or object is present with probability 0.5 |
| `palette="r g b a\|r g b a\|..."` | pick one colour from the list |

```bash
for s in 0 1 2 3; do uv run python scripts/build_scenarios.py --seed $s; done
```

gives four independent worlds per scenario for a training sweep, each a normal model file with
no runtime cost. Without `--seed` the build is exactly nominal — every optional object present,
no jitter, first palette colour — so the checked-in scenarios are reproducible.

## What is checked, and why

Both scripts exist to stop a scenario claiming to test "pick the green cube" while the green
cube lies under the counter. Each of these checks has been seen to fail in practice.

`build_scenarios.py`, on the `stand` keyframe of every build (nominal and seeded):

- **Support.** The smallest signed distance from each free object to anything that is not itself
  must be between −1 mm and +1 cm. A larger positive gap means it is in mid-air and falls on
  load; a negative one means it starts inside something and the solver flings it away. Measured
  with `mj_geomDistance`, so it does not care which way up the object ended.
- **Jitter budget.** No two free objects may overlap anywhere in their jitter range, using the
  worst-case rotated footprint when `yaw_jitter` is set. Checked on the nominal layout, so one
  build vouches for every `--seed` variant instead of the bug surfacing on seed 1 only.
- Cameras, `pelvis`, `base_mocap` and `pick_table_top` exist; the surface is big enough for the
  number of free objects.

`check_scenarios.py`, after 2 s of physics through the real `SimSource`:

- no free object starts more than 1 mm inside anything;
- every free object starts on **scenery** — a table top, shelf deck, terrain tile or the floor —
  not held up only by another loose object, and after the settle it is still touching that same
  scenery. ("Not on the floor" is not enough: a cube that starts overlapping a crate settles
  *inside* the crate, 1 cm up, and would pass a floor-only check.)
- it is not on the floor unless it started there, and moved less than 5 cm sideways / 3 cm down;
- nothing is still drifting a second later;
- `sim_randomise` reports back the seed it was given, does not move the objects it skipped, and
  leaves every object it did place resting on the pick surface; `sim_reset` restores the keyframe.

Objects placed with `on="<terrain>"` are settled by 0.4 s of physics at build time and the result
is baked into the keyframe, because no arithmetic on a tile grid gets the resting pose of a brick
lying across two tilted tiles right.

## Conventions and units

- SI throughout: metres, kilograms, seconds, radians. Quaternions are wxyz.
- World frame: **+x forward, +y left, +z up**, origin on the floor where the robot spawns; the
  robot faces +x.
- Object origins are at the bottom face; furniture origins are at the floor.
- Surface heights: work table and packing bench 0.75 m, kitchen counter 0.90 m, shelf decks
  0.10 / 0.55 / 1.00 / 1.45 m, pallet deck 0.144 m.
- Table legs are visual only (`contype=0`), as in the original scene, so a hand cannot snag on
  them; shelf posts and decks do collide.
