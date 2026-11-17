# /// script
# requires-python = "==3.10.*"   # cyclonedds 0.10.x (what the G1 speaks) ships wheels up to 3.10 only
# dependencies = [
#   "unitree_sdk2py @ https://github.com/unitreerobotics/unitree_sdk2_python/archive/814556d15970dd2ecf1c9984e845ca02ab07e206.zip",
#   "cyclonedds==0.10.2",
#   "numpy",
# ]
# ///
"""A pretend G1 for rehearsing without the robot: publishes rt/lowstate, obeys rt/arm_sdk.

    uv run lab/fake_robot.py lo0          # terminal 1
    uv run lab/01_read_state.py lo0       # terminal 2, then 02_arm_test.py lo0

Each commanded joint moves a fraction of the way to weight*q_cmd + (1-weight)*q_now per tick;
no physics, no gravity. It exists to test the plumbing, not the robot.
"""
from __future__ import annotations

import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

ChannelFactoryInitialize(0, sys.argv[1] if len(sys.argv) > 1 else "lo0")
state = unitree_hg_msg_dds__LowState_()
for i, q in ((15, 0.3), (18, 1.0), (22, 0.3), (25, 1.0)):   # arms hanging, elbows bent
    state.motor_state[i].q = q
cmd = {"msg": None}
ChannelSubscriber("rt/arm_sdk", LowCmd_).Init(lambda m: cmd.update(msg=m), 10)
pub = ChannelPublisher("rt/lowstate", LowState_)
pub.Init()
print("fake G1 publishing rt/lowstate at 500 Hz; Ctrl-C to stop")
while True:
    m = cmd["msg"]
    if m is not None:
        w = m.motor_cmd[29].q
        for i in range(12, 29):
            s = state.motor_state[i]
            s.q += 0.02 * w * (m.motor_cmd[i].q - s.q)
    state.tick += 2
    pub.Write(state)
    time.sleep(0.002)
