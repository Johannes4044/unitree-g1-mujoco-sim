"""Unitree G1 MuJoCo simulator: robot sources and whole-body controllers.

Only two subpackages live here:

    g1.robot   - RobotSource and its implementations (sim, real, replay) plus the Mid-360 LiDAR
    g1.wbc     - whole-body controllers (kinematic, holosoma, pd_stand)

This is the simulator-only extract of the wider G1 project; the module files are kept
byte-identical to the main repository wherever possible so fixes can be synced both ways.
"""
__version__ = "0.1.0"
