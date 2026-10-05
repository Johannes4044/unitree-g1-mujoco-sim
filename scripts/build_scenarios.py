"""Compose a MuJoCo scenario (G1 + world + objects) from a short declarative file.

    python scripts/build_scenarios.py                          # build all
    python scripts/build_scenarios.py scenarios/house_rooms.xml
    python scripts/build_scenarios.py --seed 7                 # randomised variants

Input:  scenarios/<name>.xml   - a few dozen lines (see the README for the schema)
Output: models/scenarios/<name>.xml (or <name>_s<seed>.xml), a self-contained model that
        `SimSource` and `scripts/render_scene.py` load unchanged.

Why a build step and not plain <include>: an include is fine for one extra body - SimSource repairs
short keyframes at compile time, so the added body keeps its pose through __init__ and sim_reset.
What an include cannot do is instantiate the same part several times without name collisions, cut a
doorway out of a wall, lay out terrain, rebuild the `stand` keyframe around the composed joint
layout, produce seeded variants, or check that every object is actually resting on something. That
is what this script is for.

Everything else is MuJoCo's own MjSpec attach mechanism: each part in models/objects/ is a
standalone model, attached under a frame with a name prefix, so parts can be reused any number of
times without name collisions. No new dependencies: mujoco + numpy, both already in the `sim` extra.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # the repo root; every path below is relative to it, no Docker paths
ROBOT_MODEL = os.environ.get("ROBOT_MODEL", os.path.join(ROOT, "models", "g1_pick_place.xml"))
OBJECT_DIR = os.path.join(ROOT, "models", "objects")
SCENARIO_DIR = os.path.join(ROOT, "scenarios")
OUT_DIR = os.path.join(ROOT, "models", "scenarios")

# World bodies of the robot model that are the robot itself; everything else in its worldbody is
# the demo pick-and-place furniture and is dropped before the scenario builds its own world.
ROBOT_BODIES = {"pelvis", "base_mocap"}
PICK_SURFACE_GEOM = "pick_table_top"   # SimSource.PICK_TABLE_GEOM: the randomisation surface
SUPPORT_GAP = 0.01      # max gap between an object and whatever holds it up [m]
SUPPORT_PROBE = 0.10    # how far to look for that support [m]
PENETRATION = 0.001     # max allowed interpenetration at the keyframe [m]
CLEARANCE = 0.01        # min gap between two free objects across their whole jitter range [m]


def floats(node, attr, n=None, default=None):
    raw = node.get(attr)
    if raw is None:
        return default
    v = [float(x) for x in raw.replace(",", " ").split()]
    if n is not None and len(v) != n:
        raise SystemExit(f"{attr}='{raw}': expected {n} numbers, got {len(v)}")
    return v


def pos3(node, attr="pos"):
    v = floats(node, attr, default=[0.0, 0.0, 0.0])
    if len(v) == 2:
        v = v + [0.0]
    if len(v) != 3:
        raise SystemExit(f"{attr}='{node.get(attr)}': expected 2 or 3 numbers")
    return v


def yaw_quat(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


class Builder:
    def __init__(self, scenario_path, seed=None):
        self.path = scenario_path
        self.root = ET.parse(scenario_path).getroot()
        self.name = self.root.get("name") or os.path.splitext(os.path.basename(scenario_path))[0]
        self.doc = " ".join((self.root.findtext("doc") or "").split())
        self.seed = seed
        self.rng = np.random.default_rng(seed) if seed is not None else None
        self.parts = {}     # part type -> MjSpec (loaded once, copied on attach)
        self.radii = {}     # part type -> horizontal half-extents (x, y) [m]
        self.settle = []    # objects placed on terrain: settled by physics before the keyframe
        self.footprints = []  # (name, x, y, reach_x, reach_y) incl. the jitter budget, for check()
        self.terrain = {}   # terrain name -> (x0, y0, tile, heights[nx, ny]) for on="<terrain>" snapping
        self.notes = []

    # --- helpers -------------------------------------------------------------
    def jitter(self, node, x, y, yaw):
        """Build-time pose randomisation (only with --seed)."""
        if self.rng is None:
            return x, y, yaw
        j = float(node.get("jitter", 0.0))
        jy = float(node.get("yaw_jitter", 0.0))
        if j:
            x += float(self.rng.uniform(-j, j))
            y += float(self.rng.uniform(-j, j))
        if jy:
            yaw += float(self.rng.uniform(-jy, jy))
        return x, y, yaw

    def present(self, node):
        """`optional="0.5"` = build-time presence randomisation of a distractor."""
        p = node.get("optional")
        if p is None:
            return True
        if self.rng is None:
            return True     # nominal build keeps every distractor
        return bool(self.rng.random() < float(p))

    def colour(self, node):
        """rgba, or one drawn from `palette="r g b a | r g b a | ..."` when seeded."""
        pal = node.get("palette")
        if pal and self.rng is not None:
            options = [p.split() for p in pal.split("|") if p.strip()]
            return [float(v) for v in options[int(self.rng.integers(len(options)))]]
        rgba = floats(node, "rgba", 4)
        if rgba is None and pal:
            return [float(v) for v in pal.split("|")[0].split()]
        return rgba

    def part_spec(self, kind):
        if kind not in self.parts:
            f = os.path.join(OBJECT_DIR, f"{kind}.xml")
            if not os.path.exists(f):
                raise SystemExit(f"{self.path}: unknown part type '{kind}' (no {f})")
            self.parts[kind] = mujoco.MjSpec.from_file(f)
        return self.parts[kind]

    def part_extent(self, kind, yaw=0.0):
        """Horizontal half-extents (x, y) of a part after a yaw rotation, from its compiled AABB."""
        if kind not in self.radii:
            # A fresh model, not self.part_spec(kind).compile(): compiling the cached spec after it
            # has been attached once silently drops the attach frame of every later instance.
            m = mujoco.MjModel.from_xml_path(os.path.join(OBJECT_DIR, f"{kind}.xml"))
            ax = ay = 0.0
            for g in range(m.ngeom):
                c, half = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
                rot = np.abs(m.geom_quat[g] * 0 + 0)   # placeholder, replaced below
                del rot
                rm = np.zeros(9)
                mujoco.mju_quat2Mat(rm, np.asarray(m.geom_quat[g], dtype=float))
                rm = rm.reshape(3, 3)
                centre = m.geom_pos[g] + rm @ c
                reach = np.abs(rm) @ half
                ax = max(ax, abs(centre[0]) + reach[0])
                ay = max(ay, abs(centre[1]) + reach[1])
            self.radii[kind] = (ax, ay)
        ax, ay = self.radii[kind]
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        return c * ax + s * ay, s * ax + c * ay

    # --- build ---------------------------------------------------------------
    def build(self):
        spec = mujoco.MjSpec.from_file(ROBOT_MODEL)
        spec.copy_during_attach = True
        spec.modelname = self.name if self.seed is None else f"{self.name}_s{self.seed}"

        # 1. remember the robot's stand pose, then drop the keyframe (its size is about to change)
        m0 = spec.compile()
        k0 = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if k0 < 0:
            raise SystemExit(f"{ROBOT_MODEL} has no 'stand' keyframe")
        stand = {}
        for j in range(m0.njnt):
            a = m0.jnt_qposadr[j]
            n = 7 if m0.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
            stand[mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, j)] = m0.key_qpos[k0, a:a + n].copy()
        mpos = m0.key_mpos[k0].copy() if m0.nmocap else np.zeros(0)
        mquat = m0.key_mquat[k0].copy() if m0.nmocap else np.zeros(0)
        for key in list(spec.keys):
            spec.delete(key)

        # 2. strip the demo world (tables, tote, the four demo objects); keep the robot
        world = spec.worldbody
        for b in list(world.bodies):
            if b.name not in ROBOT_BODIES:
                spec.delete(b)
        # ...and the materials only the demo furniture used ('tote_mat', 'table_mat'), so that a
        # scenario part attached with prefix 'tote_' can own the name 'tote_mat' itself.
        keep = {g.material for g in spec.geoms if g.material} | default_materials(ROBOT_MODEL)
        for mat in list(spec.materials):
            if mat.name not in keep:
                spec.delete(mat)

        # 3. floor
        floor = self.root.find("floor")
        if floor is not None:
            g = next((g for g in world.geoms if g.name == "floor"), None)
            rgba = self.colour(floor)
            if g is not None and rgba is not None:
                mat = spec.add_material(name="floor_mat", rgba=rgba)
                g.material = mat.name

        # 4. architecture: walls (with optional doorways) and plain boxes
        for i, w in enumerate(self.root.findall("wall")):
            self.add_wall(spec, world, w, i)
        for i, b in enumerate(self.root.findall("box")):
            self.add_box(spec, world, b, i)
        for i, t in enumerate(self.root.findall("terrain")):
            self.add_terrain(spec, world, t, i)

        # 5. parts (static furniture) and objects (free bodies)
        for node in self.root.findall("part") + self.root.findall("object"):
            self.attach_part(spec, world, node, free=node.tag == "object")

        # 6. cameras declared by the scenario replace same-named world cameras
        for c in self.root.findall("camera"):
            nm = c.get("name")
            for cam in list(world.cameras):
                if cam.name == nm:
                    spec.delete(cam)
            p = pos3(c)
            xy = floats(c, "xyaxes", 6)
            if xy is None:
                xy = look_at(p, floats(c, "target", 3, [0.0, 0.0, 0.8]))
            world.add_camera(name=nm, pos=p, xyaxes=xy, fovy=float(c.get("fovy", 45)))

        # 7. robot placement + rebuilt 'stand' keyframe
        rb = self.root.find("robot")
        dx, dy, dyaw = 0.0, 0.0, 0.0
        if rb is not None:
            dx, dy = float(rb.get("x", 0)), float(rb.get("y", 0))
            dyaw = float(rb.get("yaw", 0))
        for q in stand.values():
            if len(q) == 7:                       # the pelvis free joint
                q[0] += dx
                q[1] += dy
                if dyaw:
                    q[3:7] = mul_quat(yaw_quat(dyaw), q[3:7])
        if len(mpos):
            mpos[0] += dx
            mpos[1] += dy
            if dyaw:
                mquat[:] = mul_quat(yaw_quat(dyaw), mquat)
            mocap = next((b for b in world.bodies if b.name == "base_mocap"), None)
            if mocap is not None:
                mocap.pos = list(mpos)
                mocap.quat = list(mquat)

        model = spec.compile()
        qpos = model.qpos0.copy()                 # free objects: qpos0 is exactly where we placed them
        for j in range(model.njnt):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if nm in stand:
                a = model.jnt_qposadr[j]
                qpos[a:a + len(stand[nm])] = stand[nm]
        ctrl = np.array([qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]] for i in range(model.nu)])
        kw = {}
        if len(mpos):
            kw = {"mpos": mpos.tolist(), "mquat": mquat.tolist()}
        spec.add_key(name="stand", qpos=qpos.tolist(), ctrl=ctrl.tolist(), **kw)
        model = spec.compile()
        if self.settle:
            qpos = self.settle_on_terrain(model, qpos)
            next(k for k in spec.keys if k.name == "stand").qpos = qpos.tolist()
            model = spec.compile()
        self.check(model)
        return spec, model

    def settle_on_terrain(self, model, qpos):
        """Bake the resting pose of objects placed with on="<terrain>" into the keyframe.

        A brick lying across tilted tiles touches them at one corner; no arithmetic on the tile
        grid gets that pose right, so let the solver find it once at build time instead of letting
        every single load start with a small fall. Deterministic: same model, same 0.4 s rollout.
        """
        data = mujoco.MjData(model)
        # by name: the robot model's <size nkey="1"> reserves an empty keyframe at index 0
        mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand"))
        for _ in range(round(0.4 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        out = qpos.copy()
        for nm in self.settle:
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
            for j in range(model.njnt):
                if int(model.jnt_bodyid[j]) == b and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    a = model.jnt_qposadr[j]
                    out[a:a + 7] = data.qpos[a:a + 7]
        return out

    def add_wall(self, spec, world, w, i):
        """A wall segment between two floor points, optionally with a doorway cut out."""
        name = w.get("name", f"wall{i}")
        x0, y0 = floats(w, "from", 2)
        x1, y1 = floats(w, "to", 2)
        h = float(w.get("height", 2.5))
        t = float(w.get("thickness", 0.1))
        rgba = self.colour(w) or [0.85, 0.83, 0.78, 1]
        mat = spec.add_material(name=f"{name}_mat", rgba=rgba)
        length = math.hypot(x1 - x0, y1 - y0)
        yaw = math.atan2(y1 - y0, x1 - x0)
        door = floats(w, "door", 2)               # [centre along the wall from `from`, clear width]
        segs = []                                 # (centre along wall, half length, z centre, half height)
        if door is None:
            segs.append((length / 2, length / 2, h / 2, h / 2))
        else:
            dc, dw = door
            dh = float(w.get("door_height", 2.05))
            if dc - dw / 2 > 1e-6:
                segs.append(((dc - dw / 2) / 2, (dc - dw / 2) / 2, h / 2, h / 2))
            if length - (dc + dw / 2) > 1e-6:
                a = dc + dw / 2
                segs.append(((a + length) / 2, (length - a) / 2, h / 2, h / 2))
            if h > dh:
                segs.append((dc, dw / 2, (dh + h) / 2, (h - dh) / 2))   # lintel over the doorway
        c, s = math.cos(yaw), math.sin(yaw)
        for k, (u, half, zc, zh) in enumerate(segs):
            px = x0 + c * u
            py = y0 + s * u
            world.add_geom(name=f"{name}_{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=[half, t / 2, zh], pos=[px, py, zc], quat=yaw_quat(yaw),
                           material=mat.name, condim=3, friction=[0.9, 0.01, 0.0002])

    def add_box(self, spec, world, b, i):
        """A static box: kerbs, ramps, pillars, low steps."""
        name = b.get("name", f"box{i}")
        rgba = self.colour(b) or [0.6, 0.6, 0.62, 1]
        mat = spec.add_material(name=f"{name}_mat", rgba=rgba)
        size = floats(b, "size", 3)
        world.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos3(b),
                       quat=yaw_quat(float(b.get("yaw", 0))), material=mat.name,
                       condim=3, friction=[0.9, 0.01, 0.0002])

    def add_terrain(self, spec, world, t, i):
        """A grid of slightly tilted, slightly raised tiles: uneven ground without a heightfield.

        Tiles are static boxes, so the cost is `ngeom` and nothing else; contacts are only
        generated where a foot is. `keepout="x y r"` leaves the robot's spawn patch flat.
        """
        name = t.get("name", f"terrain{i}")
        x0, y0, x1, y1 = floats(t, "area", 4)
        tile = float(t.get("tile", 0.5))
        hmax = float(t.get("height", 0.05))
        tilt = float(t.get("tilt", 0.0))          # max tile tilt [rad]
        rgba = self.colour(t) or [0.45, 0.43, 0.40, 1]
        mat = spec.add_material(name=f"{name}_mat", rgba=rgba)
        keep = floats(t, "keepout", 3)
        # deterministic even without --seed: terrain shape is part of the scenario, not of the
        # domain randomisation, unless the scenario asks for a seeded variant.
        rng = np.random.default_rng(int(t.get("seed", 0)) if self.rng is None else self.rng.integers(1 << 30))
        nx = max(1, round((x1 - x0) / tile))
        ny = max(1, round((y1 - y0) / tile))
        n = 0
        tops = np.zeros((nx, ny))
        for ix in range(nx):
            for iy in range(ny):
                cx = x0 + (ix + 0.5) * tile
                cy = y0 + (iy + 0.5) * tile
                flat = keep is not None and math.hypot(cx - keep[0], cy - keep[1]) < keep[2]
                hz = 0.01 if flat else float(rng.uniform(0.01, max(0.011, hmax)))
                q = [1.0, 0.0, 0.0, 0.0]
                if tilt and not flat:
                    rx, ry = rng.uniform(-tilt, tilt, 2)
                    q = mul_quat([math.cos(rx / 2), math.sin(rx / 2), 0, 0],
                                 [math.cos(ry / 2), 0, math.sin(ry / 2), 0])
                world.add_geom(name=f"{name}_{ix}_{iy}", type=mujoco.mjtGeom.mjGEOM_BOX,
                               size=[tile / 2 * 0.98, tile / 2 * 0.98, hz], pos=[cx, cy, hz],
                               quat=list(q), material=mat.name, condim=3, friction=[0.9, 0.01, 0.0002])
                # true top: the highest corner of the (possibly tilted) tile, not its centre top
                rm = np.zeros(9)
                mujoco.mju_quat2Mat(rm, np.asarray(q, dtype=float))
                corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
                corners = corners * [tile / 2 * 0.98, tile / 2 * 0.98, hz]
                tops[ix, iy] = hz + float((corners @ rm.reshape(3, 3).T)[:, 2].max())
                n += 1
        self.terrain[name] = (x0, y0, tile, tops)
        self.notes.append(f"terrain '{name}': {n} tiles")

    def attach_part(self, spec, world, node, free):
        kind = node.get("type")
        name = node.get("name") or kind
        if not self.present(node):
            self.notes.append(f"dropped optional {'object' if free else 'part'} '{name}'")
            return
        child = self.part_spec(kind)
        child_body = child.worldbody.bodies[0].name
        x, y, z = pos3(node)
        yaw = float(node.get("yaw", 0))
        nominal = (x, y)
        if free:
            # worst case over the whole jitter range, so a nominal build vouches for every seed
            j = float(node.get("jitter", 0.0))
            ax, ay = self.part_extent(kind, yaw)
            if float(node.get("yaw_jitter", 0.0)) > 0.05:
                ax = ay = math.hypot(*self.radii[kind])     # any yaw is possible: use the diagonal
            self.footprints.append((name, nominal[0], nominal[1], ax + j, ay + j))
        x, y, yaw = self.jitter(node, x, y, yaw)
        on = node.get("on")
        if on is not None:       # z is measured from the surface of that terrain, not from the floor
            if on not in self.terrain:
                raise SystemExit(f"{self.path}: on='{on}' but no <terrain name='{on}'> before it")
            x0, y0, tile, tops = self.terrain[on]
            # An object wider than a tile straddles several of them, and a tilted tile is highest
            # at a corner: rest it on the highest tile it covers, or it starts inside the next one.
            rx, ry = self.part_extent(kind, yaw)
            i0 = int(np.clip((x - rx - x0) / tile, 0, tops.shape[0] - 1))
            i1 = int(np.clip((x + rx - x0) / tile, 0, tops.shape[0] - 1))
            j0 = int(np.clip((y - ry - y0) / tile, 0, tops.shape[1] - 1))
            j1 = int(np.clip((y + ry - y0) / tile, 0, tops.shape[1] - 1))
            z += float(tops[i0:i1 + 1, j0:j1 + 1].max()) + 0.003
            if node.tag == "object":
                self.settle.append(name)   # rough ground: let physics find the real resting pose
        frame = world.add_frame(pos=[x, y, z], quat=yaw_quat(yaw))
        prefix = f"{name}_"
        spec.attach(child, prefix=prefix, frame=frame)
        body = spec.body(prefix + child_body)
        body.name = name                          # the VLA prompt vocabulary is the body name
        has_free = bool(body.joints)
        if free and not has_free:
            raise SystemExit(f"{self.path}: <object name='{name}' type='{kind}'> is a static part "
                             f"(no free joint in {kind}.xml) - declare it with <part>")
        if not free and has_free:
            # <part> means scenery: strip the free joint. Static bodies are invisible to
            # SimSource's object list and to sim_randomise, which is usually what you want for
            # a crate that is an obstacle rather than a target.
            for j in list(body.joints):
                spec.delete(j)
        rgba = self.colour(node)
        if rgba is not None:
            mat = next((m for m in spec.materials if m.name == prefix + "mat"), None)
            if mat is None:
                raise SystemExit(f"{kind}.xml has no material named 'mat', so rgba= cannot recolour it")
            mat.rgba = rgba

    def check(self, model):
        """Fail loudly on anything the running system needs and the scenario forgot."""
        need_cam = ["d435i_rgb", "d455_rgb", "third_person", "mid360_cam0", "mid360_cam1",
                    "mid360_cam2", "mid360_cam3"]
        for c in need_cam:
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c) < 0:
                raise SystemExit(f"{self.name}: camera '{c}' is missing")
        for b in ("pelvis", "base_mocap"):
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) < 0:
                raise SystemExit(f"{self.name}: body '{b}' is missing")
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PICK_SURFACE_GEOM)
        if g < 0:
            raise SystemExit(f"{self.name}: no geom '{PICK_SURFACE_GEOM}'. sim_randomise(scope="
                             f"'surface') has nothing to select. Give the scenario a table-like "
                             f"<part name=\"pick_table\" .../> whose surface geom is 'top'.")
        self.check_objects_are_supported(model)
        self.check_jitter_budget()
        # World-axis footprint of the pick surface (the surface can be turned: rubble_yard's bench
        # is at yaw=pi/2). SimSource measures it the same way.
        R = model.geom_quat[g]
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, R)
        hx, hy = (np.abs(mat.reshape(3, 3)[:2, :2]) @ model.geom_size[g][:2])
        nfree = sum(1 for j in range(model.njnt)
                    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                    and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[j])) != "pelvis")
        area = max(0.0, 2 * hx - 0.10) * max(0.0, 2 * hy - 0.10)   # 5 cm edge margin, both sides
        if nfree and area / nfree < 0.08 ** 2:
            self.notes.append(f"NOTE: {nfree} free bodies on a {2*hx:.2f} x {2*hy:.2f} m surface; "
                              f"sim_randomise wants 8 cm spacing and may fail to place them")

    def check_jitter_budget(self):
        """No two free objects may overlap anywhere in their jitter range.

        Checked on the nominal layout, so one build vouches for every --seed: otherwise a variant
        wedges two objects into each other and the solver throws both off the table, which only
        shows up when somebody happens to build that seed.
        """
        bad = []
        for i, (n1, x1, y1, rx1, ry1) in enumerate(self.footprints):
            for n2, x2, y2, rx2, ry2 in self.footprints[i + 1:]:
                gap_x = abs(x1 - x2) - (rx1 + rx2)
                gap_y = abs(y1 - y2) - (ry1 + ry2)
                if max(gap_x, gap_y) < CLEARANCE:
                    bad.append(f"{n1} and {n2}: only {max(gap_x, gap_y) * 100:.1f} cm apart at the "
                               f"worst jitter (need {CLEARANCE * 100:.0f} cm on one axis)")
        if bad:
            raise SystemExit(f"{self.name}: objects can collide once jittered:\n    "
                             + "\n    ".join(bad))

    @staticmethod
    def _body_extent(model, data, b):
        """Horizontal half-extents of a body's geoms in world axes [m]."""
        rx = ry = 0.0
        org = data.xpos[b]
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) != b:
                continue
            c, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
            rot = data.geom_xmat[g].reshape(3, 3)
            centre = data.geom_xpos[g] + rot @ c
            reach = np.abs(rot) @ half
            rx = max(rx, abs(centre[0] - org[0]) + reach[0])
            ry = max(ry, abs(centre[1] - org[1]) + reach[1])
        return rx, ry

    def check_objects_are_supported(self, model):
        """Every free object must start resting on something solid, with nothing intersecting it.

        Two authoring mistakes this catches, both of which put an object on the floor before the
        episode starts and are invisible in the XML: a `pos` outside the footprint of the surface
        it was meant to sit on (it simply falls), and a `pos` overlapping another object or a tote
        wall (the solver ejects it on the first step). Checked on the `stand` keyframe, which is
        exactly the state SimSource loads.
        """
        data = mujoco.MjData(model)
        k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        mujoco.mj_resetDataKeyframe(model, data, k)
        mujoco.mj_forward(model, data)              # populates poses and the contact list
        objects = {}
        for j in range(model.njnt):
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            b = int(model.jnt_bodyid[j])
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
            if nm != "pelvis":
                objects[nm] = b
        objgeoms = {nm: [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == b]
                    for nm, b in objects.items()}
        mine = {g for gs in objgeoms.values() for g in gs}
        scene = [g for g in range(model.ngeom) if g not in mine]
        bad = []
        for nm, gs in objgeoms.items():
            # Smallest signed distance to anything that is not this object. Orientation-independent
            # (an object that settled on its side has no meaningful "bottom face"), and it catches
            # both failure modes at once: a positive gap everywhere means the object is in mid-air
            # and falls on load; a negative one means it starts inside something and gets flung.
            best, best_geom = math.inf, -1
            for g in gs:
                for h in scene:
                    d = mujoco.mj_geomDistance(model, data, g, h, SUPPORT_PROBE, None)
                    if d < best:
                        best, best_geom = float(d), h
            under = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, best_geom) or "?"
            if best >= SUPPORT_PROBE:
                bad.append(f"{nm}: nothing within {SUPPORT_PROBE * 100:.0f} cm of it - it falls on "
                           f"load. Check its pos against the footprint of the surface it belongs on")
            elif best > SUPPORT_GAP:
                bad.append(f"{nm}: floating {best * 100:.1f} cm above '{under}' - it falls on load. "
                           f"Check its pos against the footprint of the surface it belongs on")
            elif best < -PENETRATION:
                bad.append(f"{nm}: starts {-best * 1000:.0f} mm inside '{under}' - the solver "
                           f"will fling it away. Check it against its neighbours' footprints")
        if bad:
            raise SystemExit(f"{self.name}: objects are not resting on their support:\n    "
                             + "\n    ".join(sorted(set(bad))))


def default_materials(model_path):
    """Material names referenced from <default> classes (spec.geoms does not see those)."""
    root = ET.parse(model_path).getroot()
    out = set()
    for d in root.iter("default"):
        for g in d.findall("geom"):
            if g.get("material"):
                out.add(g.get("material"))
    return out


def look_at(pos, target):
    """xyaxes for a camera at `pos` pointing at `target` (MuJoCo cameras look along -z, +y up)."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    y = up - f * float(f @ up)
    if np.linalg.norm(y) < 1e-6:            # looking straight down: use +x as the image up vector
        y = np.array([1.0, 0.0, 0.0]) - f * float(f @ np.array([1.0, 0.0, 0.0]))
    y /= np.linalg.norm(y)
    x = np.cross(y, -f)
    return x.tolist() + y.tolist()


def mul_quat(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return out


def write(spec, model, builder, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stem = builder.name if builder.seed is None else f"{builder.name}_s{builder.seed}"
    out = os.path.join(out_dir, f"{stem}.xml")
    root = ET.fromstring(spec.to_xml())
    # meshes live in models/assets/; the scenario is one directory deeper
    rel = os.path.relpath(os.path.join(ROOT, "models", "assets"), out_dir)
    for comp in root.findall("compiler"):
        comp.set("meshdir", rel)
    # <size nkey="1"> in the robot model reserves an *extra*, empty keyframe. Drop it: the written
    # scenario then has exactly one keyframe, so mj_resetDataKeyframe(m, d, 0) is 'stand' too.
    for size in root.findall("size"):
        size.attrib.pop("nkey", None)
        if not size.attrib:
            root.remove(size)
    for mesh in root.iter("mesh"):
        if mesh.get("file"):
            mesh.set("file", os.path.basename(mesh.get("file")))
    seed_txt = "nominal" if builder.seed is None else f"seed {builder.seed}"
    head = (f" GENERATED by scripts/build_scenarios.py from "
            f"{os.path.relpath(builder.path, ROOT)} ({seed_txt}) - do not edit by hand. "
            f"{builder.doc} ")
    root.insert(0, ET.Comment(head))
    with open(out, "w") as f:
        f.write(ET.tostring(root, encoding="unicode"))
    mujoco.MjModel.from_xml_path(out)     # the written file must load, not just the in-memory spec
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scenarios", nargs="*", help="scenario files (default: all of sim/scenarios/*.xml)")
    ap.add_argument("--seed", type=int, default=None, help="build a randomised variant -> <name>_s<seed>.xml")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    files = args.scenarios or sorted(
        os.path.join(SCENARIO_DIR, f) for f in os.listdir(SCENARIO_DIR) if f.endswith(".xml"))
    if not files:
        raise SystemExit(f"no scenarios in {SCENARIO_DIR}")
    rc = 0
    for f in files:
        b = Builder(f, seed=args.seed)
        try:
            spec, model = b.build()
            out = write(spec, model, b, args.out)
        except SystemExit as e:
            print(f"FAIL {os.path.basename(f)}: {e}")
            rc = 1
            continue
        print(f"{os.path.relpath(out, ROOT):42s} nq={model.nq} nv={model.nv} nu={model.nu} "
              f"nbody={model.nbody} ngeom={model.ngeom}")
        for n in b.notes:
            print(f"    {n}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
