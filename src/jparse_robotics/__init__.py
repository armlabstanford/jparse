"""
J-PARSE: Jacobian-based Projection Algorithm for Resolving Singularities Effectively.

A library for singularity-aware inverse kinematics control of serial manipulators.

This package provides two main interfaces:

1. **JParseCore** - Pure algorithm that takes a Jacobian matrix directly.
   Only requires numpy. Use this if you have your own kinematics implementation.

2. **Robot** - High-level interface with URDF support via Pinocchio.
   Handles kinematics internally and provides convenient methods.

Examples
--------
Using JParseCore with a custom Jacobian:

>>> from jparse_robotics import JParseCore
>>> import numpy as np
>>>
>>> jparse = JParseCore(gamma=0.1)
>>> J = np.array([[-0.707, -0.707], [0.707, 0.707]])
>>> J_parse = jparse.compute(J)
>>> dq = J_parse @ desired_velocity

Using Robot with a URDF file:

>>> from jparse_robotics import Robot
>>>
>>> robot = Robot.from_urdf("robot.urdf", "base_link", "ee_link")
>>> J_parse = robot.jparse(q)
>>> dq = J_parse @ desired_velocity

References
----------
Paper: "J-PARSE: Jacobian-based Projection Algorithm for Resolving
Singularities Effectively in Inverse Kinematic Control of Serial Manipulators"
Authors: S. Guptasarma, M. Strong, H. Zhen, M. Kennedy III
ArXiv: https://arxiv.org/abs/2505.00306
"""

from __future__ import annotations

from .core import (
    JParseCore,
    inverse_condition_number,
    manipulability_measure,
)

__version__ = "0.1.0"
__all__ = [
    # Core algorithm
    "JParseCore",
    # Utility functions
    "manipulability_measure",
    "inverse_condition_number",
    # Robot class (lazy import)
    "Robot",
]


def __getattr__(name: str):
    """Lazy import for Robot class to avoid requiring Pinocchio at import time."""
    if name == "Robot":
        from .robot import Robot

        return Robot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
