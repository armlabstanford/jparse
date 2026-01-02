"""
J-PARSE Example (No GUI): 2-Link Planar Manipulator

Demonstrates the J-PARSE algorithm without any graphical output.
"""

import numpy as np
from jparse import JParse

# Robot parameters
l1 = 1.0  # length of link 1 [m]
l2 = 1.0  # length of link 2 [m]

# Simulation parameters
dt = 0.05
T_total = 5.0
time = np.arange(0, T_total, dt)

# Trajectory parameters
center = np.array([1.0, 0.0])
radius = 2.0  # passes through singularity
omega = 0.25


def forward_kinematics(theta1, theta2):
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return x, y


def jacobian(theta1, theta2):
    J11 = -l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2)
    J12 = -l2 * np.sin(theta1 + theta2)
    J21 = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    J22 = l2 * np.cos(theta1 + theta2)
    return np.array([[J11, J12], [J21, J22]])


# Initialize
theta1, theta2 = np.pi / 4, np.pi / 4
jparse = JParse()

print("J-PARSE 2-Link Manipulator Example")
print("=" * 50)
print(f"Link lengths: l1={l1}m, l2={l2}m")
print(f"Tracking circle: center={center}, radius={radius}")
print("=" * 50)

errors = []

for i, t in enumerate(time):
    # Desired position
    x_des = center[0] + radius * np.cos(omega * t)
    y_des = center[1] + radius * np.sin(omega * t)

    # Current position
    x_cur, y_cur = forward_kinematics(theta1, theta2)

    # Error
    error = np.sqrt((x_des - x_cur)**2 + (y_des - y_cur)**2)
    errors.append(error)

    # Control
    J = jacobian(theta1, theta2)
    J_parse = jparse.JParse(J)

    v_des = 1.5 * np.array([x_des - x_cur, y_des - y_cur])
    dq = J_parse @ v_des

    # Update
    theta1 += dq.item(0) * dt
    theta2 += dq.item(1) * dt

    # Print every 1 second
    if i % 20 == 0:
        print(f"t={t:5.2f}s | pos=({x_cur:6.3f}, {y_cur:6.3f}) | "
              f"des=({x_des:6.3f}, {y_des:6.3f}) | error={error:.4f}m")

print("=" * 50)
print(f"Mean tracking error: {np.mean(errors):.4f}m")
print(f"Max tracking error:  {np.max(errors):.4f}m")
print(f"Final tracking error: {errors[-1]:.4f}m")
