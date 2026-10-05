"""Load every built scenario through the real SimSource and prove it works.

    python scripts/check_scenarios.py                                   # all built scenarios
    python scripts/check_scenarios.py models/scenarios/house_rooms.xml

For each model: compile, settle 2 s of physics from the `stand` keyframe, check the robot did not
explode, render every camera the bridge uses, take a Mid-360 scan and sanity-check the geometry,
then exercise sim_randomise / sim_reset / sim_teleport. Prints step time and render cost.
Exit code 0 means every scenario compiles, settles and renders.
"""
from __future__ import annotations

import os
import sys
import time

import mujoco
import numpy as np

from g1.robot.robot_source import CAMERAS, DEPTHS
from g1.robot.sim_source import SimSource

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repo root; no Docker paths
BUILT = os.path.join(ROOT, "models", "scenarios")
SETTLE_S = 2.0
MAX_FALL = 0.03      # an object may sink this far into its support while settling [m]
MAX_SLIDE = 0.05     # ...and slide this far [m]
PICK_SURFACE = "pick_table_top"   # SimSource's randomisation surface
PENETRATION = 0.001  # an object may start at most this far inside something [m]


def object_geom_names(src):
    """Names of every geom that belongs to a free object - the complement is scenery."""
    m = src.m
    bodies = set(src._objects.values())
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
            for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies}


def penetrations(src):
    """[(object, other geom, depth m)] for every contact deeper than PENETRATION on a free object.

    One entry per (object, other geom) pair, at the deepest point: a box resting inside another
    produces up to four contact points, and four identical FAIL lines help nobody.
    """
    m, d = src.m, src.d
    owner = {g: nm for nm, b in src._objects.items() for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b}
    deepest = {}
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist >= -PENETRATION:
            continue
        a, b = int(c.geom1), int(c.geom2)
        for x, y in ((a, b), (b, a)):
            if x in owner and owner.get(y) != owner[x]:
                key = (owner[x], mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, y) or f"geom{y}")
                deepest[key] = max(deepest.get(key, 0.0), -float(c.dist))
    return [(nm, other, depth) for (nm, other), depth in sorted(deepest.items())]


def supports(src):
    """{object name: set of geom names it is touching} - what each free object is resting on."""
    m, d = src.m, src.d
    owner = {}
    for nm, b in src._objects.items():
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) == b:
                owner[g] = nm
    out = {nm: set() for nm in src._objects}
    for i in range(d.ncon):
        c = d.contact[i]
        a, b = int(c.geom1), int(c.geom2)
        for x, y in ((a, b), (b, a)):
            if x in owner and owner.get(y) != owner[x]:
                out[owner[x]].add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, y) or f"geom{y}")
    return out


def check(path, png=False):
    name = os.path.basename(path)
    src = SimSource(path, realtime=False)
    m = src.m
    ok = True
    try:
        z0 = src.state().base_pos[2]
        mujoco.mj_forward(src.m, src.d)          # contact list for the initial support
        init_pos = {nm: np.array(p, dtype=float) for nm, p in src.state().extra["objects"].items()}
        support_init = supports(src)
        loose = object_geom_names(src)
        pen_init = penetrations(src)
        t0 = time.perf_counter()
        src.step(SETTLE_S)
        wall = time.perf_counter() - t0
        st = src.state()
        support_now = supports(src)
        z1 = st.base_pos[2]
        steps = round(SETTLE_S / m.opt.timestep)
        rate = steps / wall
        qacc = float(np.abs(src.d.qacc).max())
        print(f"\n== {name}")
        print(f"   nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} "
              f"free objects={len(src._objects)}")
        print(f"   physics: {steps} steps in {wall:.2f} s = {rate:.0f} steps/s "
              f"({rate * m.opt.timestep:.1f}x realtime), {1e6 / rate:.0f} us/step")
        print(f"   settle:  pelvis z {z0:.3f} -> {z1:.3f} m, ncon={st.extra['ncon']}, "
              f"max|qacc|={qacc:.1f}")
        if not (0.5 < z1 < 1.2):
            print("   FAIL: pelvis height is not plausible after settling")
            ok = False
        if not np.isfinite(qacc) or qacc > 1e5:
            print("   FAIL: the model exploded (qacc)")
            ok = False
        for nm, p in st.extra["objects"].items():
            if not np.isfinite(p).all() or abs(p[0]) > 50 or abs(p[1]) > 50 or p[2] < -1 or p[2] > 5:
                print(f"   FAIL: object {nm} flew off to {p}")
                ok = False

        # An overlapping placement: the solver resolves it on the first step, usually by throwing
        # one of the two objects - or, if one is hollow, by dropping the other inside it.
        for nm, other, depth in pen_init:
            print(f"   FAIL: {nm} starts {depth * 1000:.0f} mm inside '{other}'")
            ok = False
        # Support: every object must still be resting on what it started on. This is the check
        # that catches "an object silently starts on the floor" - a scenario whose target object
        # falls off the counter on load still settles, still has tiny drift, and still renders.
        for nm, p in st.extra["objects"].items():
            p_init = init_pos[nm]
            here = support_now[nm]
            dz = p[2] - p_init[2]
            dxy = float(np.hypot(p[0] - p_init[0], p[1] - p_init[1]))
            print(f"   support {nm:14s} on {sorted(here) or 'NOTHING'}"
                  f"  moved {dxy * 100:.1f} cm / {dz * 100:+.1f} cm")
            if not here:
                print(f"   FAIL: {nm} is resting on nothing after {SETTLE_S} s")
                ok = False
            # The intended support is the scenery the object touches at t=0 (table top, shelf deck,
            # terrain, floor). It must still be touching some of it: "not on the floor" is not
            # enough - a cube that starts overlapping a crate settles *in the crate*, 1 cm up.
            intended = support_init[nm] - loose
            if support_init[nm] and not intended:
                print(f"   FAIL: {nm} starts held up only by other loose objects "
                      f"{sorted(support_init[nm])} - it was never placed on scenery")
                ok = False
            if intended and not (here & intended):
                print(f"   FAIL: {nm} is resting on {sorted(here) or 'nothing'}, not on its intended "
                      f"support {sorted(intended)}")
                ok = False
            if "floor" in here and "floor" not in support_init[nm]:
                print(f"   FAIL: {nm} ended up on the floor (it started on "
                      f"{sorted(support_init[nm]) or 'nothing'})")
                ok = False
            if dz < -MAX_FALL or dxy > MAX_SLIDE:
                print(f"   FAIL: {nm} moved {dxy * 100:.1f} cm sideways / {dz * 100:+.1f} cm down "
                      f"while settling - it was not placed on its support")
                ok = False

        # drift: nothing should still be moving after settling
        p0 = dict(st.extra["objects"])
        src.step(1.0)
        p1 = src.state().extra["objects"]
        drift = max((float(np.abs(np.array(p1[k]) - np.array(v)).max()) for k, v in p0.items()), default=0.0)
        print(f"   drift:   max object movement over 1 more second: {drift * 1e3:.1f} mm")
        if drift > 0.01:
            print("   FAIL: objects are still moving after settling")
            ok = False

        for cam in CAMERAS:
            t0 = time.perf_counter()
            img = src.rgb(cam)
            dt = (time.perf_counter() - t0) * 1e3
            if img is None or img.shape != (src.cam_h, src.cam_w, 3):
                print(f"   FAIL: camera {cam} did not render")
                ok = False
                continue
            print(f"   rgb {cam:13s} {img.shape} mean={img.mean():6.1f} {dt:5.1f} ms")
            if img.std() < 1.0:
                print(f"   FAIL: camera {cam} rendered a flat image (std={img.std():.2f})")
                ok = False
        for cam in DEPTHS:
            d = src.depth(cam)
            finite = d[np.isfinite(d)]
            print(f"   depth {cam:11s} {d.shape} range {finite.min():.2f}..{finite.max():.2f} m")
            if not (finite.min() > 0.0):
                print(f"   FAIL: depth {cam} has non-positive values")
                ok = False

        t0 = time.perf_counter()
        pw, pos, _quat = src.lidar()
        dt = (time.perf_counter() - t0) * 1e3
        d = np.linalg.norm(pw - pos, axis=1)
        print(f"   lidar: {len(pw)} points in {dt:.0f} ms, sensor at "
              f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), range {d.min():.2f}..{d.max():.2f} m, "
              f"z {pw[:, 2].min():.2f}..{pw[:, 2].max():.2f} m")
        if len(pw) < 2000:
            print("   FAIL: implausibly few LiDAR returns")
            ok = False
        if d.max() < 1.0:
            print("   FAIL: LiDAR sees nothing beyond 1 m - the scenario has no geometry")
            ok = False

        try:
            # sampled now, not before the drift step: on rough ground an object can still be
            # creeping a fraction of a millimetre, and that is not the randomiser's doing
            before = {nm: np.array(p, dtype=float)
                      for nm, p in src.state().extra["objects"].items()}
            res = src.sim_randomise(seed=3, colour=True)
            sel, skipped = set(res["objects"]), set(res["skipped"])
            print(f"   sim_randomise: seed={res['seed']} placed {sorted(sel)}"
                  + (f", left alone {sorted(skipped)}" if skipped else ""))
            if res["seed"] != 3:
                print("   FAIL: sim_randomise did not report back the seed it was given")
                ok = False
            after = src.state().extra["objects"]
            for nm in skipped:
                if not np.allclose(before[nm], after[nm], atol=1e-6):
                    print(f"   FAIL: {nm} was not selected but moved anyway")
                    ok = False
            src.step(0.5)
            on = supports(src)
            for nm in sel:
                if PICK_SURFACE not in on[nm]:
                    print(f"   FAIL: {nm} was randomised onto the pick surface but is resting on "
                          f"{sorted(on[nm]) or 'nothing'}")
                    ok = False
        except Exception as e:  # noqa: BLE001
            print(f"   FAIL: sim_randomise: {e!r}")
            ok = False
        src.sim_reset()
        back = src.state().extra["objects"]
        # sim_reset restores the *keyframe* pose, which is where each object started
        restored = all(np.allclose(init_pos[nm], back[nm], atol=1e-6) for nm in back)
        print(f"   sim_reset: t={src.state().t:.3f} s, {len(back)} objects restored"
              f"{'' if restored else ' (POSES DIFFER)'}")
        if not restored:
            print("   FAIL: sim_reset did not put the objects back where they started")
            ok = False
        try:
            src.set_mode("walk")
            for tx, ty in ((-0.8, 0.0), (-0.8, -1.0), (0.0, -1.4)):
                try:
                    src.sim_teleport(tx, ty, 0.3)   # candidates: some scenarios have a table at (1, 0)
                    break
                except RuntimeError:
                    continue
            src.set_base_velocity(0.3, 0.0, 0.0)
            src.step(0.5)
            print(f"   teleport+walk: pelvis at {np.round(src.state().base_pos, 3).tolist()}")
        except Exception as e:  # noqa: BLE001
            print(f"   FAIL: teleport/walk: {e!r}")
            ok = False
        if png:
            import imageio.v2 as imageio
            out = os.path.join(ROOT, "out")
            os.makedirs(out, exist_ok=True)
            stem = os.path.splitext(name)[0]
            big = SimSource(path, cam_w=960, cam_h=540, realtime=False)
            try:
                big.step(SETTLE_S)
                for cam in ("third_person", "scene_wide", "d435i_rgb"):
                    img = big.rgb(cam) if cam in CAMERAS else None
                    if img is None:
                        big_r = big._renderer()
                        big_r.update_scene(big._snapshot(), camera=cam)
                        img = big_r.render().copy()
                    imageio.imwrite(os.path.join(out, f"{stem}_{cam}.png"), img)
            finally:
                big.close()
            print(f"   wrote {out}/{stem}_*.png")
    finally:
        src.close()
    print(f"   {'OK' if ok else 'FAILED'}: {name}")
    return ok


def main():
    png = "--png" in sys.argv
    if png:
        sys.argv.remove("--png")
    files = sys.argv[1:] or sorted(
        os.path.join(BUILT, f) for f in os.listdir(BUILT) if f.endswith(".xml"))
    if not files:
        raise SystemExit(f"no built scenarios in {BUILT} - run build_scenarios.py first")
    bad = [f for f in files if not check(f, png=png)]
    print(f"\n{len(files) - len(bad)}/{len(files)} scenarios OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
