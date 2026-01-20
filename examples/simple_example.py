#!/usr/bin/env python3
"""
Simple J-PARSE Example - Quick Start Guide

This minimal example shows how to use jparse-robotics for a 2-link planar arm.
Run with: python simple_example.py
"""

import numpy as np
import jparse_robotics as jparse


def main():
    # Create J-PARSE solver
    solver = jparse.JParseCore(gamma=0.1)

    # Robot parameters: 2-link arm with 1m links
    L1, L2 = 1.0, 1.0

    # Current joint angles [rad]
    q = np.array([np.pi/4, np.pi/4])  # 45°, 45°

    # Forward kinematics
    x = L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1])
    y = L1 * np.sin(q[0]) + L2 * np.sin(q[0] + q[1])
    print(f"Current end-effector position: ({x:.3f}, {y:.3f})")

    # Jacobian for 2-link planar arm
    J = np.array([
        [-L1*np.sin(q[0]) - L2*np.sin(q[0]+q[1]), -L2*np.sin(q[0]+q[1])],
        [ L1*np.cos(q[0]) + L2*np.cos(q[0]+q[1]),  L2*np.cos(q[0]+q[1])]
    ])

    # Desired Cartesian velocity (move right and up)
    v_desired = np.array([0.1, 0.05])  # [m/s]

    # Compute joint velocities using J-PARSE
    J_parse = solver.compute(J)
    dq = J_parse @ v_desired

    print(f"Desired velocity: vx={v_desired[0]:.3f}, vy={v_desired[1]:.3f} m/s")
    print(f"Joint velocities: dq1={dq[0]:.3f}, dq2={dq[1]:.3f} rad/s")

    # Check singularity metrics
    m = jparse.manipulability_measure(J)
    icn = jparse.inverse_condition_number(J)
    print(f"Manipulability: {m:.4f}")
    print(f"Inverse condition number: {icn:.4f} (closer to 1 = better)")

    # Compare with standard pseudo-inverse
    J_pinv = solver.pinv(J)
    dq_pinv = J_pinv @ v_desired
    print(f"\nComparison - Standard pinv: dq1={dq_pinv[0]:.3f}, dq2={dq_pinv[1]:.3f} rad/s")


if __name__ == "__main__":
    main()
