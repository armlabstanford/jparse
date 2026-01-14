#!/usr/bin/env python3
"""
Basic J-PARSE usage without URDF.

This example demonstrates how to use the JParseCore class directly
with a manually computed Jacobian matrix. This is useful when you
have your own kinematics implementation or want to use J-PARSE
with a simulation environment.

No external dependencies required beyond numpy.
"""

import numpy as np

import jparse_robotics as jparse


def compute_planar_jacobian(theta1: float, theta2: float, l1: float = 1.0, l2: float = 1.0):
    """
    Compute the Jacobian for a 2-link planar arm.

    Parameters
    ----------
    theta1 : float
        First joint angle [rad]
    theta2 : float
        Second joint angle [rad]
    l1 : float
        Length of first link [m]
    l2 : float
        Length of second link [m]

    Returns
    -------
    J : ndarray
        2x2 Jacobian matrix
    """
    J = np.array(
        [
            [
                -l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2),
                -l2 * np.sin(theta1 + theta2),
            ],
            [
                l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2),
                l2 * np.cos(theta1 + theta2),
            ],
        ]
    )
    return J


def forward_kinematics(theta1: float, theta2: float, l1: float = 1.0, l2: float = 1.0):
    """Compute end-effector position for 2-link planar arm."""
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return np.array([x, y])


def main():
    # Robot parameters
    l1, l2 = 1.0, 1.0

    # Create J-PARSE solver with gamma=0.1 (singularity threshold)
    jparse_solver = jparse.JParseCore(gamma=0.1)

    # Initial joint configuration
    theta1 = np.pi / 4
    theta2 = np.pi / 4

    # Simulation parameters
    dt = 0.01
    num_steps = 500

    # Target position (circular trajectory that passes through singularity)
    center = np.array([1.0, 0.0])
    radius = 1.8  # Large radius to challenge the workspace
    omega = 0.5  # Angular velocity

    print("=" * 60)
    print("J-PARSE Demo: 2-Link Planar Arm Tracking Circular Trajectory")
    print("=" * 60)
    print(f"Link lengths: l1={l1}m, l2={l2}m")
    print(f"Workspace limit: {l1 + l2}m")
    print(f"Trajectory: Circle centered at {center}, radius={radius}m")
    print(f"J-PARSE gamma: {jparse_solver.gamma}")
    print()

    # Control gain
    k = 2.0

    for i in range(num_steps):
        t = i * dt

        # Desired position
        p_des = center + radius * np.array([np.cos(omega * t), np.sin(omega * t)])

        # Current position
        p_cur = forward_kinematics(theta1, theta2, l1, l2)

        # Position error
        error = p_des - p_cur

        # Compute Jacobian
        J = compute_planar_jacobian(theta1, theta2, l1, l2)

        # Compute J-PARSE pseudo-inverse
        J_parse = jparse_solver.compute(J)

        # Compute joint velocities
        v_des = k * error
        dq = J_parse @ v_des

        # Update joint angles
        dq = np.asarray(dq).flatten()
        theta1 += dq[0] * dt
        theta2 += dq[1] * dt

        # Print status every 100 steps
        if i % 100 == 0:
            m = jparse.manipulability_measure(J)
            icn = jparse.inverse_condition_number(J)
            print(f"Step {i:4d}: pos=[{p_cur[0]:6.3f}, {p_cur[1]:6.3f}], "
                  f"error={np.linalg.norm(error):.4f}, "
                  f"manipulability={m:.4f}, "
                  f"inv_cond={icn:.4f}")

    print()
    print("Simulation complete!")
    print()

    # Compare with standard pseudo-inverse at a near-singular configuration
    print("=" * 60)
    print("Comparison: J-PARSE vs Standard Pseudo-Inverse")
    print("=" * 60)

    # Configuration close to singularity (arm nearly stretched out)
    theta1_sing = 0.1
    theta2_sing = 0.1
    J_sing = compute_planar_jacobian(theta1_sing, theta2_sing, l1, l2)

    print(f"\nConfiguration: theta1={np.degrees(theta1_sing):.1f}deg, "
          f"theta2={np.degrees(theta2_sing):.1f}deg")
    print(f"Inverse condition number: {jparse.inverse_condition_number(J_sing):.4f}")
    print(f"Manipulability: {jparse.manipulability_measure(J_sing):.4f}")

    # Compute pseudo-inverses
    J_pinv = jparse_solver.pinv(J_sing)
    J_parse_sing = jparse_solver.compute(J_sing)
    J_dls = jparse_solver.damped_least_squares(J_sing, damping=0.1)

    print("\nMaximum joint velocity for unit Cartesian velocity:")
    print(f"  Standard pinv: {np.max(np.abs(J_pinv)):.2f}")
    print(f"  J-PARSE:       {np.max(np.abs(J_parse_sing)):.2f}")
    print(f"  DLS (λ=0.1):   {np.max(np.abs(J_dls)):.2f}")

    print("\nJ-PARSE provides bounded velocities even near singularities!")


if __name__ == "__main__":
    main()
