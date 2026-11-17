# Lab day: first code on the real G1

Three scripts that talk to the robot over Unitree's DDS from a Mac on the robot's Ethernet,
plus a fake robot to rehearse with. They need nothing from the simulator install: each script
declares its own dependencies (Python 3.10, Unitree's `unitree_sdk2py`, `cyclonedds==0.10.2`,
which is the DDS the G1 speaks), and `uv run` sets that up on first use.

| script | does | moves the robot? |
|---|---|---|
| `01_read_state.py en7` | prints all 29 motors next to the sim's joint names, IMU, message rate; records `rt/lowstate` to `lab/out/state_*.npz` | no |
| `02_arm_test.py en7` | moves **one** arm joint by ≤ 0.5 rad and back through `rt/arm_sdk`, records commanded vs measured to `lab/out/arm_*.npz` | yes, one arm joint |
| `fake_robot.py lo0` | a pretend G1 on loopback, so 01 and 02 can be rehearsed without hardware | – |

`en7` stands for the Mac's Ethernet adapter. Find yours under [Setup](#2-at-the-lab-network-5-min).

## 1. Before you go (on the lab Mac if you can, otherwise on any Mac; 10 min, needs internet)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh         # uv, no admin needed; open a new terminal after
curl -L https://github.com/Johannes4044/unitree-g1-mujoco-sim/archive/refs/heads/main.zip -o g1.zip
unzip g1.zip && cd unitree-g1-mujoco-sim-main            # or: git clone ... if git is installed

uv run lab/01_read_state.py lo0     # first run downloads Python 3.10 + the SDK (~1 min), then
                                    # says "no rt/lowstate ..." - expected, there is no robot
```

Rehearse the whole thing against the fake robot (two terminals):

```bash
uv run lab/fake_robot.py lo0                      # terminal 1, leave running
uv run lab/01_read_state.py lo0                   # terminal 2: table of 29 joints
uv run lab/02_arm_test.py lo0                     # type yes; elbow goes +0.3 and back
uv run lab/02_arm_test.py lo0 --seconds 5         # press Ctrl-C mid-move: it must say ABORT and fade out
```

Everything is cached after this, so at the lab it also works without internet.

**Check whether you have admin rights on the lab Mac.** Setting a manual IP on the Ethernet
adapter (next step) needs them on macOS. No admin → ask IT tonight, or bring your own Mac.

## 2. At the lab: network (5 min)

1. Ethernet cable from the robot's network port to the Mac's USB-Ethernet adapter.
2. Which interface is it? `networksetup -listallhardwareports` → the "USB 10/100/1000 LAN" (or
   similar) entry's `Device: en7`. Unplug and re-run if unsure; the one that disappears is it.
3. System Settings → Network → that adapter → Details → TCP/IP → Configure IPv4 *Manually*:
   IP `192.168.123.222`, subnet mask `255.255.255.0`, router empty. (Wi-Fi can stay on for
   internet; DDS is bound to the interface you name.)
4. `ping 192.168.123.161` (motion computer) and `ping 192.168.123.164` (Jetson) answer.
5. macOS firewall: off, or click *Allow* when Python asks for incoming connections. DDS is UDP
   multicast and a blocking firewall looks exactly like "robot not found".

## 3. Safety rules (non-negotiable, read them out loud once)

- Someone holds the **Unitree remote** the whole time and knows which buttons put the robot into
  damping. That person does nothing else.
- First runs with the robot **on the gantry / hoist**, feet just touching or off the ground.
- Only `rt/arm_sdk` (arms, Unitree keeps balancing) and Unitree's own high-level client
  (`LocoClient`). **Not** `rt/lowcmd` / debug mode on day 1: that turns Unitree's balance off
  and a bug drops the robot.
- Change one thing at a time; small deltas first (`--delta 0.2`), then bigger.
- Nobody inside arm reach while `02_arm_test.py` runs. Ctrl-C is a fade-out, not an e-stop;
  the remote is the e-stop.
- The lab's own rules and staff overrule this page.

## 4. The day

Times are for a ~4 h slot; scale to what you get. Every block has a "done when", so you know
when to move on, and a stop condition.

| # | time | what | done when |
|---|---|---|---|
| 0 | 0:00–0:20 | Arrive, briefing with lab staff: power-on, remote, gantry, damping mode, who is the e-stop person. Network setup (section 2). | `ping .161` works |
| 1 | 0:20–0:50 | **Read only.** `uv run lab/01_read_state.py en7 --seconds 10` with the robot in damping on the gantry, then again in standing mode. | ~500 msg/s; values plausible; two `state_*.npz` saved |
| 2 | 0:50–1:10 | **Joint order check.** Robot in damping, staff OK: one person gently moves the *left* elbow by hand while 01 runs; only row 18 (`left_elbow_joint`) changes. Repeat for one right-arm and one leg joint. This proves the sim's joint order = the robot's. | 3 joints confirmed, sign noted (bending the elbow = q up or down?) |
| 3 | 1:10–1:40 | **Unitree's controller from our Mac** (section 5): stand up, wave hand, `Move` 0.3 m/s for a moment. Confirms the Mac can command at all before our own code does. | robot stood and waved from the Mac |
| – | 1:40–1:50 | break, write down what you learned | |
| 4 | 1:50–2:40 | **Our first motion.** Robot standing (staff decides gantry or floor). `02_arm_test.py en7 --delta 0.2`, then 0.3, then `--joint left_shoulder_pitch_joint --delta -0.3` (arm forward), then the right arm. Test Ctrl-C once on purpose. | four `arm_*.npz` saved, tracking error printed small, abort tested |
| 5 | 2:40–3:30 | **Collect what RealSource needs** (section 6) – questions only the robot answers. | the list in section 6 filled in |
| 6 | 3:30–4:00 | Buffer, tidy up, copy `lab/out/` off the Mac (USB stick / AirDrop). | data on your own machine |

Stop and go back to damping whenever: a reading looks wrong, the robot makes a noise you don't
expect, a motor gets hot (the temperature column), or anyone is unsure.

## 5. Running Unitree's own examples from this Mac

Unitree ships tested examples (stand up, damp, walk, wave, arm demos). Same environment as ours:

```bash
SHA=814556d15970dd2ecf1c9984e845ca02ab07e206
curl -L https://github.com/unitreerobotics/unitree_sdk2_python/archive/$SHA.zip -o sdk.zip && unzip -q sdk.zip
uv run --no-project --python 3.10 --with cyclonedds==0.10.2 --with ./unitree_sdk2_python-$SHA \
  python unitree_sdk2_python-$SHA/example/g1/high_level/g1_loco_client_example.py en7
```

Type `list` for the options. Careful: **`8` / `zero torque` makes the robot limp** – only on the
gantry. `1` (Squat2StandUp) and `12` (Lie2StandUp) need the robot in the right starting pose;
ask the staff. The `g1_arm7_sdk_dds_example.py` next to it drives **both arms and the waist to
zero** – a big motion; run it only after our small `02_arm_test.py` worked.

## 6. What to find out on the robot (feeds `src/g1/robot/real_source.py`)

The TODOs in `RealSource` say "confirm on the robot, do not guess". Bring answers to these:

- [ ] Robot model / DoF: 29-DoF with waist roll+pitch unlocked? (rows 13, 14 in 01 change?)
- [ ] `motor_state[i].temperature` is two numbers – which is the winding, which the driver?
      What do they read cold, and after 10 min standing?
- [ ] Joint signs: elbow bend direction (block 2) matches the sim?
- [ ] Standing pose: `state_*.npz` from standing vs the sim's keyframe (`workspace/01`).
- [ ] Base velocity: is there an estimate on `rt/lowstate` or only from SLAM `/Odometry`?
- [ ] Jetson (`ssh unitree@192.168.123.164`, credentials from the lab): Ubuntu/JetPack version,
      `rs-enumerate-devices -s` (RealSense serials: which is the D435i, which the D455?),
      `ros2 topic list` (Livox / SLAM running? topic names?)
- [ ] Hands: which hands are fitted? Is a BrainCo hand service running, what topic names?
- [ ] Head module: how is it commanded?
- [ ] Battery: which field, and how long does a charge last standing?

## 7. After the lab

- `lab/out/arm_*.npz` has `t, q_cmd, q_meas, weight, phase` for the moved joint: run the same
  move in the sim (`workspace/02_move_the_arm.py` with the same start pose and target) and plot
  both – the first sim-vs-real number for this project.
- Fill in `RealSource` from section 6.

## Troubleshooting

| symptom | fix |
|---|---|
| `no rt/lowstate within 5 s` | the script prints a checklist; firewall and wrong interface are the usual two |
| `uv: command not found` | open a new terminal after installing uv, or `~/.local/bin/uv` |
| download fails at the lab | you skipped section 1 with internet; use Wi-Fi/phone hotspot once |
| arm doesn't move, no error | robot not in a mode where arm_sdk is active – stand it up with the remote first |
| `ABORT (tracking error ...)` | the arm was blocked or the gains are too soft for that load; stop and look before retrying |
