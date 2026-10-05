"""Build the G1 pick-and-place scene with MuJoCo's MjSpec API.

Needs the upstream sources, which are NOT vendored here (they are large and this repo
ships the compiled result instead). To regenerate models/g1_pick_place.xml, clone them at
the commits pinned in models/ASSETS.md and point the environment variables below at them:

    MENAGERIE=...      google-deepmind/mujoco_menagerie  @ 8161bba
    REVO2=...          BrainCoTech/revo2_description 1.0.0 @ 0ea3db7
    UNITREE_EXTRA=...  unitreerobotics/unitree_ros robots/g1_description/meshes @ 7d6075f
    pip install trimesh fast-simplification      # mesh decimation, not in this project's deps

Robot:   Menagerie Unitree G1 (29 DoF, rev 1.0). Waist joints are free.
Hands:   HAND=revo2 (default): BrainCo Revo 2 five-finger hands compiled from the
         Apache-2.0 URDF in BrainCoTech/revo2_description (pinned in models/ASSETS.md),
         11 joints / 6 actuators per hand, distal joints coupled to proximal joints
         with the URDF's <mimic> ratios. Actuators, couplings and contact settings are
         authored here. (BrainCo's brainco-description MJCF is NOT used: it has no
         licence.)
         HAND=dex3: keep the Menagerie Unitree Dex3 three-finger hands.
Head:    RGB + depth camera at Unitree's official D435 pose on the torso
         (from unitree_ros g1_description), plus the Menagerie D435i visual body.
Body:    Unitree backpack mesh (visual only) on the back of the torso.
Scene:   pick table with 4 objects in front of the robot, packing table with a
         tote 2.5 m away, third-person and overhead cameras.
Base:    FIXED_BASE=1 (default) welds the pelvis to the world, so the robot
         cannot fall while only arms/hands/camera are used. FIXED_BASE=0 frees it.

Output:  models/g1_pick_place.xml + models/assets/ (meshdir-relative mesh paths).
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MENAGERIE = os.environ.get("MENAGERIE", "")
REVO2 = os.environ.get("REVO2", "")   # BrainCoTech/revo2_description (Apache-2.0)
EXTRA = os.environ.get("UNITREE_EXTRA", os.path.join(ROOT, "assets", "unitree_extra"))
for _name, _path in (("MENAGERIE", MENAGERIE), ("REVO2", REVO2), ("UNITREE_EXTRA", EXTRA)):
    if not os.path.isdir(_path):
        raise SystemExit(f"${_name}={_path!r} is not a directory - see this script's docstring: "
                         "the upstream sources are not vendored, only the model they produce "
                         "(models/g1_pick_place.xml) is.")
G1_XML = os.path.join(MENAGERIE, "unitree_g1", "g1_with_hands.xml")
D435_XML = os.path.join(MENAGERIE, "realsense_d435i", "d435i.xml")
HAND = os.environ.get("HAND", "revo2")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models", "g1_pick_place.xml")

TABLE_H = 0.75            # tabletop height [m]
PICK_TABLE = (0.55, 0.0)  # centre x,y; robot stands at origin facing +x
PACK_TABLE = (2.5, 1.2)
TABLE_SIZE = (0.35, 0.6, 0.02)  # half sizes

# Sensors on the head (all poses in torso_link frame, URDF camera x-axis = optical axis):
# - Internal Intel RealSense D435i behind the visor: Unitree g1_29dof_mode_15_with_dex1_1.urdf
#   "d435_joint" xyz 0.0576 0.0175 0.4299, pitch 0.8308 rad (47.6 deg down).
# - Livox Mid-360 on the crown: same URDF "mid360_joint" xyz 0.00028 0.00003 0.4284, rpy pi 0.051 0
#   (mounted upside down, so its +52..-7 deg vertical FOV becomes 7 deg up .. 52 deg down).
# - Unitree "G1 head dual-DoF module" with a RealSense D455 on top: pose from g1_comp.urdf chain
#   torso -> head_servo -> xl330 (tilt servo) -> d455: xyz 0.064 0 0.491. Pan/tilt joints
#   are modelled as position-controlled hinges (ranges are guesses until Unitree confirms).
D435I_POS = (0.0576235, 0.01753, 0.42987)
D435I_PITCH = 0.8307767239493009
D435I_FOVY_RGB, D435I_FOVY_DEPTH = 42, 58          # D435i: RGB 69x42, depth 87x58 deg
MID360_POS = (0.0002835, 0.00003, 0.428434)
MID360_PITCH = 0.05112069379091391
MID360_V_FOV = (-52.0, 7.0)                          # deg, after the upside-down mount
MID360_RANGE = 40.0                                  # m (10% reflectivity spec)
HEAD_MODULE_POS = (0.0039635 + 0.030518, 0.0, -0.047 + 0.52486)   # xl330 tilt-servo pivot
D455_OFFSET = (0.0295, 0.0, 0.013)                   # d455 link from the tilt pivot
D455_FOVY_RGB, D455_FOVY_DEPTH = 65, 58              # D455: RGB 90x65, depth 87x58 deg
HEAD_TILT_DEG = float(os.environ.get("HEAD_CAM_PITCH_DEG", "35"))  # initial tilt (servo), down positive
HEAD_PAN_RANGE, HEAD_TILT_RANGE = (-1.2, 1.2), (-0.5, 1.2)          # rad, guesses

# Revo 2 mount on the wrist: the G1 wrist_yaw_link mesh ends at x=0.026, the Revo 2
# base mesh starts 0.046 below its origin; add a 5 mm adapter plate.
REVO2_MOUNT_X = 0.045   # measured visually: the Revo 2 base mesh has a short flange stub, not 46 mm

# Backpack (visual only). No public URDF places this mesh, so this is eyeballed
# against photos of the G1 EDU: flush against the back of the torso shell.
BACKPACK_POS = (-0.115, 0.0, 0.135)


def quat_from_mat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).flatten())
    return q.tolist()


# The Revo 2 hand contract. Joint names, their order and their limits are what recorded datasets
# (the 31-slot observation.state / action vector) is indexed by. The
# build fails if the URDF ever yields anything else, e.g. after bumping the revo2_description pin
# (its 1.1.x releases move the thumb zero and change limits and mimic ratios).
REVO2_JOINTS = ("thumb_metacarpal", "thumb_proximal", "thumb_distal", "index_proximal", "index_distal",
                "middle_proximal", "middle_distal", "ring_proximal", "ring_distal",
                "pinky_proximal", "pinky_distal")
REVO2_ACTUATED = ("thumb_metacarpal", "thumb_proximal", "index_proximal", "middle_proximal",
                  "ring_proximal", "pinky_proximal")
REVO2_UPPER = {j: 1.57 if j == "thumb_metacarpal" else 1.03 if j.startswith("thumb")
               else 1.41 if j.endswith("proximal") else 1.63 for j in REVO2_JOINTS}   # lower = 0
REVO2_COUPLING = {"thumb": 1.0, "index": 1.155, "middle": 1.155, "ring": 1.155, "pinky": 1.155}


def load_revo2(side):
    """One Revo 2 hand as an MjSpec, compiled by MuJoCo from revo2_description's URDF.

    Bodies, joints, limits, inertials and meshes come from the URDF unchanged. Two renames: the
    root link <side>_base_link becomes <side>_hand_base_link (it hangs off the G1 wrist), and each
    geom gets its own mesh asset named <body>_<visual|collision>_<i>_<mesh file stem>, so the
    exported asset files have stable, self-describing names.

    Returns (spec, couplings, geom_kind): couplings = {distal joint: (proximal joint, multiplier,
    offset)} read from the URDF <mimic> tags, which MuJoCo's URDF importer ignores;
    geom_kind = {geom name: "visual" | "collision"}.
    """
    urdf = os.path.join(REVO2, "urdf", f"revo2_{side}_hand.urdf")
    mesh_dir = os.path.join(REVO2, "meshes", f"revo2_{side}_hand")
    root = ET.parse(urdf).getroot()
    couplings = {}
    for j in root.iter("joint"):
        mimic = j.find("mimic")
        if mimic is not None:
            couplings[j.get("name")] = (mimic.get("joint"), float(mimic.get("multiplier", "1")),
                                        float(mimic.get("offset", "0")))
    for m in root.iter("mesh"):   # package://revo2_description/meshes/<dir>/<file> -> absolute path
        m.set("filename", os.path.join(mesh_dir, os.path.basename(m.get("filename"))))
    # Keep the visual geoms and the massless tip/touch links (fingertip reference frames).
    ET.SubElement(ET.SubElement(root, "mujoco"), "compiler",
                  discardvisual="false", fusestatic="false", strippath="false")
    hand = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    hand.body(f"{side}_base_link").name = f"{side}_hand_base_link"
    old_meshes = list(hand.meshes)
    files = {m.name: m.file for m in old_meshes}
    geom_kind = {}
    for body in hand.bodies:
        count = {"visual": 0, "collision": 0}
        for g in body.geoms:
            kind = "visual" if g.contype == 0 else "collision"   # the importer makes visuals contype 0
            g.name = f"{body.name}_{kind}_{count[kind]}"
            mesh_name = f"{g.name}_{os.path.splitext(os.path.basename(files[g.meshname]))[0]}"
            hand.add_mesh(name=mesh_name, file=files[g.meshname])
            g.meshname = mesh_name
            geom_kind[g.name] = kind
            count[kind] += 1
    for m in old_meshes:
        hand.delete(m)
    return hand, couplings, geom_kind


# ---------------------------------------------------------------------------
# Remember the 'stand' keyframe per joint name so it survives hand swapping.
m0 = mujoco.MjModel.from_xml_path(G1_XML)
k0 = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_KEY, "stand")
stand = {}
for j in range(m0.njnt):
    n = m0.jnt_qposadr[j]
    size = 7 if m0.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
    stand[mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, j)] = m0.key_qpos[k0, n:n + size].copy()

spec = mujoco.MjSpec.from_file(G1_XML)
spec.modelname = f"g1_pick_place_{HAND}"
spec.meshdir = os.path.join(MENAGERIE, "unitree_g1", "assets")
spec.compiler.meshdir = spec.meshdir
spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC   # elliptic cones + impratio 10: MuJoCo's advice
spec.option.impratio = 10                           # for grasping (less slip at the fingertips)
spec.visual.global_.offwidth = 1920
spec.visual.global_.offheight = 1080
# Drop the original keyframe now: it is validated on every delete and its size changes below.
for key in list(spec.keys):
    spec.delete(key)

# --- hands -------------------------------------------------------------------
if HAND == "revo2":
    # Delete the Dex3 hands (bodies under the wrist, palm geoms, hand actuators).
    for a in list(spec.actuators):
        if "_hand_" in a.name:
            spec.delete(a)
    for side in ("left", "right"):
        wrist = spec.body(f"{side}_wrist_yaw_link")
        for b in list(wrist.bodies):
            spec.delete(b)
        for g in list(wrist.geoms):
            if g.meshname == f"{side}_hand_palm_link":
                spec.delete(g)

        revo, couplings, geom_kind = load_revo2(side)
        # Revo 2 base frame: fingers +z, palm normal +x, thumb +y (right) / -y (left).
        # G1 wrist_yaw frame: fingers +x, palm normal +y (right) / -y (left), forward +z.
        # Columns = images of Revo (x, y, z) axes in wrist coordinates.
        if side == "right":
            R = np.array([[0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]], dtype=float)
        else:
            R = np.array([[0, 0, 1],
                          [-1, 0, 0],
                          [0, -1, 0]], dtype=float)
        frame = wrist.add_frame(pos=[REVO2_MOUNT_X, 0, 0], quat=quat_from_mat(R))
        frame.attach_body(revo.body(f"{side}_hand_base_link"), "", "")

        # Our simulation settings on top of the URDF geometry:
        # - hand geoms join the G1's own 'visual' / 'collision' default classes; collision geoms
        #   get condim 4 (torsional friction) and a rubber-ish sliding friction for grasping;
        # - hand joints get the G1 joint defaults (armature, frictionloss) - the URDF has no
        #   <dynamics>, and these keep the fingers' position servos well damped.
        visual_cls, collision_cls = spec.find_default("visual"), spec.find_default("collision")
        for g in spec.geoms:
            kind = geom_kind.get(g.name)
            if kind == "visual":
                g.classname = visual_cls
                g.contype, g.conaffinity, g.group = 0, 0, visual_cls.geom.group
                g.density, g.material = visual_cls.geom.density, visual_cls.geom.material
            elif kind == "collision":
                g.classname = collision_cls
                g.group = collision_cls.geom.group
                g.rgba = [0.5, 0.5, 0.5, 1]   # MuJoCo's default; the URDF colour is for visuals
                g.condim = 4
                g.friction = [1.0, 0.005, 0.0001]
        g1_joint = spec.find_default("g1").joint
        for jn in REVO2_JOINTS:
            j = spec.joint(f"{side}_{jn}_joint")
            j.armature, j.frictionloss = g1_joint.armature, g1_joint.frictionloss

        # Real Revo 2: 6 motors per hand; each distal joint follows its proximal joint with the
        # URDF's <mimic> ratio (equality polycoef = offset + multiplier * parent).
        assert {c.replace(f"{side}_", "").replace("_distal_joint", ""): k for c, (_, k, _) in couplings.items()} \
            == REVO2_COUPLING, f"{side} Revo 2 couplings changed: {couplings}"
        for child, (parent, k, offset) in couplings.items():
            spec.add_equality(name=child.replace("_joint", "_couple"), type=mujoco.mjtEq.mjEQ_JOINT,
                              objtype=mujoco.mjtObj.mjOBJ_JOINT, name1=child, name2=parent,
                              data=[offset, k] + [0] * 9)
        for jn in REVO2_ACTUATED:
            jname = f"{side}_{jn}_joint"
            jr = spec.joint(jname).range
            act = spec.add_actuator(name=jname, target=jname, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                                    ctrlrange=list(jr), forcerange=[-2, 2])
            act.set_to_position(kp=3.0, kv=0.15)
    print("hands: BrainCo Revo 2 (12 actuators)")
else:
    print("hands: Unitree Dex3 (Menagerie)")

# --- backpack (visual) -------------------------------------------------------
torso = spec.body("torso_link")
spec.add_mesh(name="backpack_link", file=os.path.join(EXTRA, "backpack_link.STL"))
torso.add_geom(name="backpack", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="backpack_link",
               pos=list(BACKPACK_POS), rgba=[0.15, 0.15, 0.15, 1], contype=0, conaffinity=0, group=2)

# --- world: floor, lights, skybox ------------------------------------------
SHADOWS = os.environ.get("SHADOWS", "0") == "1"   # shadows + floor reflection cost ~4x in software GL
spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                 rgb1=[0.4, 0.6, 0.8], rgb2=[0.0, 0.0, 0.0], width=512, height=3072)
spec.add_texture(name="groundplane", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                 mark=mujoco.mjtMark.mjMARK_EDGE, rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], markrgb=[0.8, 0.8, 0.8],
                 width=300, height=300)
spec.add_material(name="groundplane", textures=["", "groundplane"], texuniform=True, texrepeat=[5, 5],
                  reflectance=0.2 if SHADOWS else 0.0)
spec.add_material(name="table_mat", rgba=[0.75, 0.6, 0.45, 1])
spec.add_material(name="tote_mat", rgba=[0.2, 0.3, 0.7, 1])

world = spec.worldbody
world.add_light(pos=[0, 0, 3.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                diffuse=[0.7, 0.7, 0.7], specular=[0.3, 0.3, 0.3], castshadow=SHADOWS)
world.add_light(pos=[2, -2, 3], dir=[-0.5, 0.5, -1], diffuse=[0.4, 0.4, 0.4], castshadow=False)
world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="groundplane",
               contype=1, conaffinity=1)


def add_table(name, cx, cy):
    b = world.add_body(name=name, pos=[cx, cy, 0])
    b.add_geom(name=f"{name}_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(TABLE_SIZE),
               pos=[0, 0, TABLE_H - TABLE_SIZE[2]], material="table_mat", contype=1, conaffinity=1,
               friction=[0.8, 0.005, 0.0001])
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        b.add_geom(name=f"{name}_leg{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[0.025, (TABLE_H - 2 * TABLE_SIZE[2]) / 2, 0],
                   pos=[sx * (TABLE_SIZE[0] - 0.05), sy * (TABLE_SIZE[1] - 0.05), (TABLE_H - 2 * TABLE_SIZE[2]) / 2],
                   material="table_mat", contype=0, conaffinity=0)
    return b


add_table("pick_table", *PICK_TABLE)
add_table("pack_table", *PACK_TABLE)

# Tote on the packing table: open box from 5 thin slabs.
tote = world.add_body(name="tote", pos=[PACK_TABLE[0], PACK_TABLE[1], TABLE_H])
tw, tl, th, t = 0.20, 0.30, 0.12, 0.006
tote.add_geom(name="tote_bottom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[tw, tl, t], pos=[0, 0, t], material="tote_mat")
for nm, size, pos in [("tote_wall_px", [t, tl, th / 2], [tw - t, 0, th / 2]),
                      ("tote_wall_nx", [t, tl, th / 2], [-(tw - t), 0, th / 2]),
                      ("tote_wall_py", [tw, t, th / 2], [0, tl - t, th / 2]),
                      ("tote_wall_ny", [tw, t, th / 2], [0, -(tl - t), th / 2])]:
    tote.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, material="tote_mat")
tote.add_site(name="tote_drop", pos=[0, 0, th + 0.1], size=[0.01, 0, 0], rgba=[0, 1, 0, 0.3])

# Pickable objects on the pick table (free bodies). Names are what the VLA prompt refers to.
OBJECTS = [
    ("red_box",     "box",      [0.03, 0.03, 0.03],  [0.9, 0.1, 0.1, 1],   (0.45, 0.15)),
    ("blue_can",    "cylinder", [0.03, 0.05, 0],     [0.1, 0.3, 0.9, 1],   (0.50, -0.15)),
    ("green_box",   "box",      [0.04, 0.025, 0.04], [0.1, 0.7, 0.2, 1],   (0.65, 0.30)),
    ("yellow_ball", "sphere",   [0.035, 0, 0],       [0.95, 0.85, 0.1, 1], (0.60, -0.35)),
]
gtype = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}
for name, typ, size, rgba, (x, y) in OBJECTS:
    half_h = size[2] if typ == "box" else (size[1] if typ == "cylinder" else size[0])
    z = TABLE_H + half_h + 0.001
    b = world.add_body(name=name, pos=[x, y, z])
    b.add_freejoint(name=f"{name}_free")
    b.add_geom(name=f"{name}_geom", type=gtype[typ], size=size, rgba=rgba, mass=0.1,
               friction=[1.0, 0.005, 0.0001], condim=4, contype=1, conaffinity=1)

# --- head sensors ------------------------------------------------------------
def cam_xyaxes(pitch):
    """Camera looking along +x tilted down by pitch (image up ~ world up)."""
    return [0, -1, 0, math.sin(pitch), 0, math.cos(pitch)]

# Internal D435i (fixed, behind the visor)
fwd = np.array([math.cos(D435I_PITCH), 0, -math.sin(D435I_PITCH)])
lens = np.array(D435I_POS) + 0.012 * fwd
torso.add_camera(name="d435i_rgb", pos=lens.tolist(), xyaxes=cam_xyaxes(D435I_PITCH), fovy=D435I_FOVY_RGB, resolution=[848, 480])
torso.add_camera(name="d435i_depth", pos=(lens + [0, -0.015, 0]).tolist(), xyaxes=cam_xyaxes(D435I_PITCH), fovy=D435I_FOVY_DEPTH, resolution=[848, 480])
torso.add_site(name="d435i_site", pos=list(D435I_POS), size=[0.004, 0, 0], rgba=[1, 0, 0, 0.5])

# Head dual-DoF module + D455 (pan about z, tilt about y; position actuators)
head_pan = torso.add_body(name="head_module_pan", pos=list(HEAD_MODULE_POS))
head_pan.add_joint(name="head_pan_joint", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], range=list(HEAD_PAN_RANGE), damping=0.5)
head_pan.add_geom(name="head_module_base", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.018, 0.012, 0], pos=[0, 0, -0.012],
                  rgba=[0.15, 0.15, 0.15, 1], contype=0, conaffinity=0, group=2, mass=0.05)
head_tilt = head_pan.add_body(name="head_module_tilt", pos=[0, 0, 0])
head_tilt.add_joint(name="head_tilt_joint", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=list(HEAD_TILT_RANGE), damping=0.5)
spec.add_mesh(name="d455_link", file=os.path.join(EXTRA, "d455_link.STL"))
head_tilt.add_geom(name="d455_body", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="d455_link", pos=list(D455_OFFSET),
                   rgba=[0.2, 0.2, 0.2, 1], contype=0, conaffinity=0, group=2, mass=0.1)
d455_lens = np.array(D455_OFFSET) + [0.015, 0, 0]
head_tilt.add_camera(name="d455_rgb", pos=d455_lens.tolist(), xyaxes=cam_xyaxes(0.0), fovy=D455_FOVY_RGB, resolution=[848, 480])
head_tilt.add_camera(name="d455_depth", pos=(d455_lens + [0, -0.03, 0]).tolist(), xyaxes=cam_xyaxes(0.0), fovy=D455_FOVY_DEPTH, resolution=[848, 480])
for jn, rng in (("head_pan_joint", HEAD_PAN_RANGE), ("head_tilt_joint", HEAD_TILT_RANGE)):
    a = spec.add_actuator(name=jn, target=jn, trntype=mujoco.mjtTrn.mjTRN_JOINT, ctrlrange=list(rng), forcerange=[-3, 3])
    a.set_to_position(kp=5.0, kv=0.3)
# Backward-compatible aliases used by older scripts
torso.add_camera(name="head_cam", pos=lens.tolist(), xyaxes=cam_xyaxes(D435I_PITCH), fovy=D435I_FOVY_RGB, resolution=[848, 480])
torso.add_camera(name="head_cam_depth", pos=(lens + [0, -0.015, 0]).tolist(), xyaxes=cam_xyaxes(D435I_PITCH), fovy=D435I_FOVY_DEPTH, resolution=[848, 480])

# Livox Mid-360: a puck on the crown plus a 4-camera depth rig that covers 360 deg
# horizontally and the -52..+7 deg vertical band; lidar.py unprojects it to a point cloud.
lidar = torso.add_body(name="mid360", pos=list(MID360_POS), euler=[math.pi, MID360_PITCH, 0])
lidar.add_geom(name="mid360_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.032, 0.03, 0], pos=[0, 0, 0.03],
               rgba=[0.1, 0.1, 0.1, 1], contype=0, conaffinity=0, group=2, mass=0.265)
lidar.add_site(name="mid360_site", pos=[0, 0, 0], size=[0.004, 0, 0], rgba=[0, 1, 0, 0.5])
band_center = math.radians(-(MID360_V_FOV[0] + MID360_V_FOV[1]) / 2)   # 22.5 deg down in world terms
for i, yaw in enumerate((0, 90, 180, 270)):
    yr = math.radians(yaw)
    # camera frame (mujoco): looks along -z, +y up. Build from forward f, up u: x = f x u, y = u, z = -f
    f = np.array([math.cos(yr) * math.cos(band_center), math.sin(yr) * math.cos(band_center), -math.sin(band_center)])
    # The lidar body is upside down (euler pi about x): express in body frame by flipping y and z.
    f_body = np.array([f[0], -f[1], -f[2]])
    u_body = np.array([0, 0, -1.0])
    u_body = u_body - f_body * (u_body @ f_body); u_body /= np.linalg.norm(u_body)
    x_body = np.cross(f_body, u_body)
    lidar.add_camera(name=f"mid360_cam{i}", pos=[0, 0, 0.03], xyaxes=x_body.tolist() + u_body.tolist(), fovy=100, resolution=[256, 256])

# --- fixed base ----------------------------------------------------------------
# FIXED_BASE=1: weld the pelvis to a mocap body 'base_mocap' instead of the world.
# The mocap body is placed at the pelvis 'stand' pose, so the robot cannot fall,
# and SimSource moves it kinematically (set_base_velocity) as a stand-in for
# locomotion until a whole-body controller exists.
FIXED_BASE = os.environ.get("FIXED_BASE", "1") != "0"
base_pose = next((v for v in stand.values() if len(v) == 7), None)   # free joint: pos(3) + quat(4)
if FIXED_BASE:
    base_pos = list(base_pose[0:3]) if base_pose is not None else [0, 0, 0.793]
    base_quat = list(base_pose[3:7]) if base_pose is not None else [1, 0, 0, 0]
    world.add_body(name="base_mocap", mocap=True, pos=base_pos, quat=base_quat)
    # data = anchor(3) + relpose(pos 3, quat 4) + torquescale(1): identity relpose so the pelvis
    # frame coincides with the mocap frame (base_mocap sits exactly at the pelvis keyframe pose).
    # Stiff weld (time constant 2.5x the 2 ms timestep; MuJoCo's stability floor is 2x) so the
    # pelvis tracks the kinematically driven mocap body with little lag.
    spec.add_equality(name="fixed_base", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                      name1="base_mocap", name2="pelvis", data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                      solref=[0.005, 1.0])
    print("pelvis welded to mocap body 'base_mocap' (FIXED_BASE=1)")

# The Menagerie G1 file brings its own shadow-casting light; align it with SHADOWS.
for light in spec.lights:
    light.castshadow = SHADOWS
if not SHADOWS:
    spec.visual.quality.shadowsize = 1024

# --- world cameras -------------------------------------------------------------
world.add_camera(name="third_person", pos=[1.8, -2.2, 1.7], xyaxes=[0.77, 0.64, 0, -0.25, 0.30, 0.92], fovy=45)
world.add_camera(name="overhead_pick", pos=[PICK_TABLE[0], PICK_TABLE[1], 2.2], xyaxes=[0, 1, 0, -1, 0, 0], fovy=50)
world.add_camera(name="scene_wide", pos=[1.3, -3.5, 2.0], xyaxes=[1, 0.35, 0, -0.1, 0.3, 0.95], fovy=55)

# --- keyframe: rebuild 'stand' for the new joint layout --------------------------
model = spec.compile()
qpos = model.qpos0.copy()
for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    if name in stand:
        n = model.jnt_qposadr[j]
        qpos[n:n + len(stand[name])] = stand[name]
jt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_tilt_joint")
if jt >= 0:
    qpos[model.jnt_qposadr[jt]] = math.radians(HEAD_TILT_DEG)
ctrl = np.zeros(model.nu)
for i in range(model.nu):
    ctrl[i] = qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
key_kw = {}
if FIXED_BASE:
    key_kw = dict(mpos=list(base_pos), mquat=list(base_quat))   # mocap coincident with the pelvis
spec.add_key(name="stand", qpos=qpos.tolist(), ctrl=ctrl.tolist(), **key_kw)
model = spec.compile()
if HAND == "revo2":   # the hand contract (see REVO2_JOINTS): names, order, limits, actuators
    jnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    anames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(model.nu)]
    for side in ("left", "right"):
        want = [f"{side}_{j}_joint" for j in REVO2_JOINTS]
        assert [n for n in jnames if n in want] == want, f"{side} hand joint order changed"
        assert [n for n in anames if n in want] == [f"{side}_{j}_joint" for j in REVO2_ACTUATED], \
            f"{side} hand actuators changed"
        for j in REVO2_JOINTS:
            rng = model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{j}_joint")]
            assert np.allclose(rng, [0.0, REVO2_UPPER[j]]), f"{side}_{j}_joint range changed: {rng}"
if FIXED_BASE:
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_mocap") >= 0, "base_mocap missing"

os.makedirs(os.path.dirname(OUT), exist_ok=True)
# --- decimate meshes and export a portable asset bundle -------------------------
# Software rendering is vertex-bound (1.1M verts -> <1 fps). Decimate visual meshes
# to <= VISUAL_MAX_VERTS and collision meshes to <= COLLISION_MAX_VERTS (MuJoCo uses
# the convex hull of collision meshes anyway). Meshes are written next to the model
# with a relative meshdir, so the XML also loads natively on the Mac.
import shutil
import trimesh
import fast_simplification
VISUAL_MAX_VERTS = int(os.environ.get("VISUAL_MAX_VERTS", "2500"))
COLLISION_MAX_VERTS = int(os.environ.get("COLLISION_MAX_VERTS", "800"))
asset_dir = os.path.join(os.path.dirname(OUT), "assets")
os.makedirs(asset_dir, exist_ok=True)
collision_meshes = {g.meshname for g in spec.geoms if g.meshname and g.classname.name == "collision"}
visual_meshes = {g.meshname for g in spec.geoms if g.meshname and g.classname.name != "collision"}
stats = []
for mesh in spec.meshes:
    src = mesh.file if os.path.isabs(mesh.file) else os.path.join(spec.meshdir, mesh.file)
    dst_name = f"{mesh.name}.stl"
    dst = os.path.join(asset_dir, dst_name)
    tm = trimesh.load(src, force="mesh")
    nv0 = len(tm.vertices)
    limit = COLLISION_MAX_VERTS if (mesh.name in collision_meshes and mesh.name not in visual_meshes) else VISUAL_MAX_VERTS
    if nv0 > limit:
        v, f = fast_simplification.simplify(tm.vertices.astype(np.float64), tm.faces.astype(np.int64),
                                            target_reduction=1 - limit / nv0)
        tm = trimesh.Trimesh(v, f)
    tm.export(dst)
    stats.append((nv0, len(tm.vertices), mesh.name))
    mesh.file = dst   # absolute for compiling; made relative in the written XML below
print(f"meshes: {sum(a for a,_,_ in stats)} -> {sum(b for _,b,_ in stats)} vertices, {len(stats)} files in {asset_dir}")
model = spec.compile()

xml = spec.to_xml()
# Attaching the Revo 2 specs duplicates their unnamed top-level <default> (classes
# 'visual'/'collision' already exist in the G1 spec). Drop the unnamed copies.
root = ET.fromstring(xml)
top = root.find("default")
if top is not None:
    for child in list(top.findall("default")):
        if child.get("class") is None:
            top.remove(child)
# Portable paths: every mesh now lives in <model dir>/assets.
for comp in root.findall("compiler"):
    comp.set("meshdir", "assets")
for mesh in root.iter("mesh"):
    if mesh.get("file"):
        mesh.set("file", os.path.basename(mesh.get("file")))
xml = ET.tostring(root, encoding="unicode")
with open(OUT, "w") as f:
    f.write(xml)
mujoco.MjModel.from_xml_path(OUT)  # verify the written file loads
print(f"wrote {OUT}: nq={model.nq} nv={model.nv} nu={model.nu} ncam={model.ncam} nbody={model.nbody} neq={model.neq}")
print("actuators:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)])
