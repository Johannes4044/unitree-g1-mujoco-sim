"""Cold-start check that does not render: did the install actually work?

    MUJOCO_GL=disabled python .github/scripts/install_check.py

Proves `import g1` resolves to an installed package, the vendored scene compiles,
physics runs and `state()` reports a standing robot. Deliberately touches no camera,
no depth buffer and no lidar, so it needs no OpenGL context at all.

This exists because a GitHub-hosted macOS runner cannot create one: there is no
window server session, so CGLChoosePixelFormat fails with "invalid pixel format"
and every render dies - on a real Mac the same code renders fine. Linux runners get
a working EGL context from Mesa and run the full `scripts/check_scenarios.py`
instead, which is the stronger check because it renders every camera.

Nothing in the repo depends on this file; it exists only for CI.
"""
import os
import sys

if os.environ.get("MUJOCO_GL", "").lower() not in ("disable", "disabled", "off", "false", "0", ""):
    sys.exit(f"refusing to run with MUJOCO_GL={os.environ['MUJOCO_GL']!r}: this check is "
             "physics-only and must not depend on a GL backend being selectable")

import g1  # noqa: E402  - imported for its location, before anything else from the package
from g1.robot.sim_source import SimSource, default_model_path  # noqa: E402

print(f"g1:    {g1.__file__}")
print(f"model: {default_model_path()}")

sim = SimSource(realtime=False)
try:
    before = sim.state()
    sim.step(2.0)
    after = sim.state()

    print(f"model: nq={sim.m.nq} nv={sim.m.nv} nu={sim.m.nu} "
          f"nbody={sim.m.nbody} ngeom={sim.m.ngeom}")
    print(f"state: t={after.t:.3f} s, {len(after.joints)} joints, "
          f"base at {[round(v, 3) for v in after.base_pos]}")

    # The scene starts from the `stand` keyframe, so 2 s of settling should leave the
    # pelvis near standing height. Loose bounds on purpose: this is "the install works
    # and physics is not NaN", not a physics regression test - tests/ owns those.
    z = after.base_pos[2]
    if not 0.4 < z < 1.2:
        sys.exit(f"pelvis at z={z:.3f} m after 2 s - expected roughly standing height")
    if after.t <= before.t:
        sys.exit(f"clock did not advance: {before.t} -> {after.t}")
    if not after.joints:
        sys.exit("state() reported no joints")
finally:
    sim.close()

print("OK: installed, scene compiles, physics runs. Rendering NOT checked here.")
