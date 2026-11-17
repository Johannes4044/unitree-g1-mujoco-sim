"""Shared by the lab scripts: open DDS on the robot network and keep the latest rt/lowstate."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "lab" / "out"
sys.path.insert(0, str(REPO / "src"))   # the sim's joint names are the source of truth for both sides

from g1.robot.robot_source import BODY_JOINTS

HINTS = """no rt/lowstate within {timeout:.0f} s on interface '{iface}'. Check, in order:
  1. Is the robot powered on and booted (~1 min after power-on)?
  2. Is the Ethernet cable in the robot's port and the adapter on this Mac?
  3. `ifconfig {iface}` shows inet 192.168.123.222 netmask 0xffffff00? (lab/README.md, Setup)
  4. `ping 192.168.123.161` answers?
  5. macOS firewall: System Settings > Network > Firewall - off, or allow incoming for python.
  6. Wrong interface? `networksetup -listallhardwareports` lists them; the USB adapter is usually en5..en9."""


class LowState:
    """Subscribes rt/lowstate (500 Hz) and keeps the newest message plus its arrival time."""

    def __init__(self, iface: str, timeout: float = 5.0):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        ChannelFactoryInitialize(0, iface)
        self.msg = None
        self.t = 0.0
        self.count = 0
        self._lock = threading.Lock()
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_msg, 10)
        deadline = time.time() + timeout
        while self.msg is None and time.time() < deadline:
            time.sleep(0.05)
        if self.msg is None:
            sys.exit(HINTS.format(timeout=timeout, iface=iface))

    def _on_msg(self, msg) -> None:
        with self._lock:
            self.msg, self.t = msg, time.time()
            self.count += 1

    def latest(self):
        """(msg, age in seconds)."""
        with self._lock:
            return self.msg, time.time() - self.t

    def q(self) -> list[float]:
        msg, _ = self.latest()
        return [msg.motor_state[i].q for i in range(len(BODY_JOINTS))]
