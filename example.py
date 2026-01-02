"""
J-PARSE: Jacobian-based Projection Algorithm for Resolving Singularities Effectively

This example demonstrates J-PARSE on:
1. 2-Link Planar Manipulator
2. 3-Link Planar Manipulator
3. 7-DOF 3D Manipulator (requires pinocchio)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from jparse import JParse

# =============================================================================
# 2-LINK PLANAR MANIPULATOR
# =============================================================================

print("=" * 60)
print("2-LINK PLANAR MANIPULATOR")
print("=" * 60)

# Simulation parameters
dt = 0.05
T_total = 20
time_steps = int(T_total / dt)
time_2link = np.linspace(0, T_total, time_steps)

# Robot parameters
l1 = 1.0
l2 = 1.0

# Desired circular trajectory parameters
center = np.array([1.0, 0.0])
omega = 0.25
radius = 2.0  # passes through singularity

# Pre-allocate storage
theta1_hist = np.zeros(time_steps)
theta2_hist = np.zeros(time_steps)
x_hist = np.zeros(time_steps)
y_hist = np.zeros(time_steps)
xd_hist = np.zeros(time_steps)
yd_hist = np.zeros(time_steps)

# Initial conditions
theta1 = np.pi / 4
theta2 = np.pi / 4


def forward_kinematics_2link(theta1, theta2):
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return x, y


def jacobian_2link(theta1, theta2):
    J11 = -l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2)
    J12 = -l2 * np.sin(theta1 + theta2)
    J21 = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    J22 = l2 * np.cos(theta1 + theta2)
    return np.array([[J11, J12], [J21, J22]])


def control_law_2link(x_current, y_current, x_des, y_des, theta1, theta2, jparse_obj):
    k = 1.5
    error = np.array([x_des - x_current, y_des - y_current])
    v_des = k * error
    J = jacobian_2link(theta1, theta2)
    J_parse = jparse_obj.JParse(J)
    dq = J_parse @ v_des
    return dq


# Instantiate JParse
jparse = JParse()

# Simulation loop
for i, t in enumerate(time_2link):
    x_des = center[0] + radius * np.cos(omega * t)
    y_des = center[1] + radius * np.sin(omega * t)
    xd_hist[i] = x_des
    yd_hist[i] = y_des

    x_current, y_current = forward_kinematics_2link(theta1, theta2)
    x_hist[i] = x_current
    y_hist[i] = y_current

    dq = control_law_2link(x_current, y_current, x_des, y_des, theta1, theta2, jparse)
    theta1 += dq.item(0) * dt
    theta2 += dq.item(1) * dt

    theta1_hist[i] = theta1
    theta2_hist[i] = theta2

# Animation
fig_anim, ax_anim = plt.subplots()
ax_anim.set_xlim(-2, 2)
ax_anim.set_ylim(-2, 2)
ax_anim.set_aspect("equal")
ax_anim.set_title("2-Link Robot Arm Tracking a Circle")

link_line, = ax_anim.plot([], [], "o-", lw=4)
target_point, = ax_anim.plot([], [], "rx", markersize=10)


def init_2link():
    link_line.set_data([], [])
    target_point.set_data([], [])
    return link_line, target_point


def animate_2link(i):
    theta1_i = theta1_hist[i]
    theta2_i = theta2_hist[i]
    joint1 = np.array([l1 * np.cos(theta1_i), l1 * np.sin(theta1_i)])
    joint2 = joint1 + np.array([l2 * np.cos(theta1_i + theta2_i), l2 * np.sin(theta1_i + theta2_i)])
    x_data = [0, joint1[0], joint2[0]]
    y_data = [0, joint1[1], joint2[1]]
    link_line.set_data(x_data, y_data)
    target_point.set_data([xd_hist[i]], [yd_hist[i]])
    return link_line, target_point


ani_2link = animation.FuncAnimation(
    fig_anim, animate_2link, frames=len(time_2link), init_func=init_2link, interval=50, blit=True
)

# Plots
fig1, ax1 = plt.subplots()
ax1.plot(time_2link, x_hist, label="x(t) actual", linewidth=2)
ax1.plot(time_2link, xd_hist, "--", label="x(t) desired", linewidth=2)
ax1.plot(time_2link, y_hist, label="y(t) actual", linewidth=2)
ax1.plot(time_2link, yd_hist, "--", label="y(t) desired", linewidth=2)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Position [m]")
ax1.set_title("2-Link End-Effector Position vs. Time")
ax1.legend()
ax1.grid(True)

fig2, ax2 = plt.subplots()
ax2.plot(x_hist, y_hist, label="Actual Trajectory", linewidth=2)
ax2.plot(xd_hist, yd_hist, "--", label="Desired Circle", linewidth=2)
ax2.set_xlabel("x [m]")
ax2.set_ylabel("y [m]")
ax2.set_title("2-Link End-Effector Trajectory")
ax2.legend()
ax2.axis("equal")
ax2.grid(True)

fig3, ax3 = plt.subplots()
ax3.plot(time_2link, theta1_hist, label=r"$\theta_1(t)$", linewidth=2)
ax3.plot(time_2link, theta2_hist, label=r"$\theta_2(t)$", linewidth=2)
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("Joint Angles [rad]")
ax3.set_title("2-Link Joint Angles vs. Time")
ax3.legend()
ax3.grid(True)

# =============================================================================
# 3-LINK PLANAR MANIPULATOR
# =============================================================================

print("=" * 60)
print("3-LINK PLANAR MANIPULATOR")
print("=" * 60)

# Time parameters
dt = 0.05
T_total = 30
time_steps_3 = int(T_total / dt)
time_3link = np.linspace(0, T_total, time_steps_3)

# Robot Parameters (3-Link)
l1_3 = 1.0
l2_3 = 1.0
l3_3 = 1.0


def forward_kinematics_3link(q1, q2, q3):
    x = l1_3 * np.cos(q1) + l2_3 * np.cos(q1 + q2) + l3_3 * np.cos(q1 + q2 + q3)
    y = l1_3 * np.sin(q1) + l2_3 * np.sin(q1 + q2) + l3_3 * np.sin(q1 + q2 + q3)
    return x, y


def jacobian_3link(q1, q2, q3):
    J11 = -l1_3 * np.sin(q1) - l2_3 * np.sin(q1 + q2) - l3_3 * np.sin(q1 + q2 + q3)
    J12 = -l2_3 * np.sin(q1 + q2) - l3_3 * np.sin(q1 + q2 + q3)
    J13 = -l3_3 * np.sin(q1 + q2 + q3)
    J21 = l1_3 * np.cos(q1) + l2_3 * np.cos(q1 + q2) + l3_3 * np.cos(q1 + q2 + q3)
    J22 = l2_3 * np.cos(q1 + q2) + l3_3 * np.cos(q1 + q2 + q3)
    J23 = l3_3 * np.cos(q1 + q2 + q3)
    return np.array([[J11, J12, J13], [J21, J22, J23]])


def control_law_3link(x_current, y_current, x_des, y_des, q1, q2, q3, jparse_obj):
    k = 1.5
    error = np.array([x_des - x_current, y_des - y_current])
    v_des = k * error
    J = jacobian_3link(q1, q2, q3)
    J_parse = jparse_obj.JParse(J)
    dq = J_parse @ v_des
    return dq[0, 0], dq[0, 1], dq[0, 2]


# Part A: Fully reachable circle
circleA_center = np.array([1.5, 0.0])
circleA_radius = 0.5
omegaA = 0.5

# Part B: Partially reachable circle
circleB_center = np.array([1.0, 0.0])
circleB_radius = 2.5
omegaB = 0.5

# Simulation for Part A
q1_hist_A = np.zeros(time_steps_3)
q2_hist_A = np.zeros(time_steps_3)
q3_hist_A = np.zeros(time_steps_3)
x_hist_A = np.zeros(time_steps_3)
y_hist_A = np.zeros(time_steps_3)
xd_hist_A = np.zeros(time_steps_3)
yd_hist_A = np.zeros(time_steps_3)

q1_A, q2_A, q3_A = np.pi / 4, np.pi / 4, np.pi / 4

for i, t in enumerate(time_3link):
    x_des = circleA_center[0] + circleA_radius * np.cos(omegaA * t)
    y_des = circleA_center[1] + circleA_radius * np.sin(omegaA * t)
    xd_hist_A[i] = x_des
    yd_hist_A[i] = y_des

    x_current, y_current = forward_kinematics_3link(q1_A, q2_A, q3_A)
    x_hist_A[i] = x_current
    y_hist_A[i] = y_current

    dq1, dq2, dq3 = control_law_3link(x_current, y_current, x_des, y_des, q1_A, q2_A, q3_A, jparse)
    q1_A += dq1 * dt
    q2_A += dq2 * dt
    q3_A += dq3 * dt

    q1_hist_A[i] = q1_A
    q2_hist_A[i] = q2_A
    q3_hist_A[i] = q3_A

# Simulation for Part B
q1_hist_B = np.zeros(time_steps_3)
q2_hist_B = np.zeros(time_steps_3)
q3_hist_B = np.zeros(time_steps_3)
x_hist_B = np.zeros(time_steps_3)
y_hist_B = np.zeros(time_steps_3)
xd_hist_B = np.zeros(time_steps_3)
yd_hist_B = np.zeros(time_steps_3)

q1_B, q2_B, q3_B = np.pi / 4, np.pi / 4, np.pi / 4

for i, t in enumerate(time_3link):
    x_des = circleB_center[0] + circleB_radius * np.cos(omegaB * t)
    y_des = circleB_center[1] + circleB_radius * np.sin(omegaB * t)
    xd_hist_B[i] = x_des
    yd_hist_B[i] = y_des

    x_current, y_current = forward_kinematics_3link(q1_B, q2_B, q3_B)
    x_hist_B[i] = x_current
    y_hist_B[i] = y_current

    dq1, dq2, dq3 = control_law_3link(x_current, y_current, x_des, y_des, q1_B, q2_B, q3_B, jparse)
    q1_B += dq1 * dt
    q2_B += dq2 * dt
    q3_B += dq3 * dt

    q1_hist_B[i] = q1_B
    q2_hist_B[i] = q2_B
    q3_hist_B[i] = q3_B

# Animation Part A
fig_anim_A, ax_anim_A = plt.subplots()
ax_anim_A.set_xlim(-3, 3)
ax_anim_A.set_ylim(-3, 3)
ax_anim_A.set_aspect("equal")
ax_anim_A.set_title("3-Link Manipulator Tracking a Reachable Circle (Part A)")

link_line_A, = ax_anim_A.plot([], [], "o-", lw=4)
target_point_A, = ax_anim_A.plot([], [], "rx", markersize=10)


def init_A():
    link_line_A.set_data([], [])
    target_point_A.set_data([], [])
    return link_line_A, target_point_A


def animate_A(i):
    joint1 = np.array([l1_3 * np.cos(q1_hist_A[i]), l1_3 * np.sin(q1_hist_A[i])])
    joint2 = joint1 + np.array([l2_3 * np.cos(q1_hist_A[i] + q2_hist_A[i]), l2_3 * np.sin(q1_hist_A[i] + q2_hist_A[i])])
    joint3 = joint2 + np.array([l3_3 * np.cos(q1_hist_A[i] + q2_hist_A[i] + q3_hist_A[i]), l3_3 * np.sin(q1_hist_A[i] + q2_hist_A[i] + q3_hist_A[i])])
    link_line_A.set_data([0, joint1[0], joint2[0], joint3[0]], [0, joint1[1], joint2[1], joint3[1]])
    target_point_A.set_data([xd_hist_A[i]], [yd_hist_A[i]])
    return link_line_A, target_point_A


ani_A = animation.FuncAnimation(fig_anim_A, animate_A, frames=len(time_3link), init_func=init_A, interval=50, blit=True)

# Animation Part B
fig_anim_B, ax_anim_B = plt.subplots()
ax_anim_B.set_xlim(-3, 3)
ax_anim_B.set_ylim(-3, 3)
ax_anim_B.set_aspect("equal")
ax_anim_B.set_title("3-Link Manipulator Tracking a Partially Reachable Circle (Part B)")

link_line_B, = ax_anim_B.plot([], [], "o-", lw=4)
target_point_B, = ax_anim_B.plot([], [], "rx", markersize=10)


def init_B():
    link_line_B.set_data([], [])
    target_point_B.set_data([], [])
    return link_line_B, target_point_B


def animate_B(i):
    joint1 = np.array([l1_3 * np.cos(q1_hist_B[i]), l1_3 * np.sin(q1_hist_B[i])])
    joint2 = joint1 + np.array([l2_3 * np.cos(q1_hist_B[i] + q2_hist_B[i]), l2_3 * np.sin(q1_hist_B[i] + q2_hist_B[i])])
    joint3 = joint2 + np.array([l3_3 * np.cos(q1_hist_B[i] + q2_hist_B[i] + q3_hist_B[i]), l3_3 * np.sin(q1_hist_B[i] + q2_hist_B[i] + q3_hist_B[i])])
    link_line_B.set_data([0, joint1[0], joint2[0], joint3[0]], [0, joint1[1], joint2[1], joint3[1]])
    target_point_B.set_data([xd_hist_B[i]], [yd_hist_B[i]])
    return link_line_B, target_point_B


ani_B = animation.FuncAnimation(fig_anim_B, animate_B, frames=len(time_3link), init_func=init_B, interval=50, blit=True)

# Plots for Part A
fig1_A, ax1_A = plt.subplots()
ax1_A.plot(time_3link, x_hist_A, label="x(t) actual", linewidth=2)
ax1_A.plot(time_3link, xd_hist_A, "--", label="x(t) desired", linewidth=2)
ax1_A.plot(time_3link, y_hist_A, label="y(t) actual", linewidth=2)
ax1_A.plot(time_3link, yd_hist_A, "--", label="y(t) desired", linewidth=2)
ax1_A.set_xlabel("Time [s]")
ax1_A.set_ylabel("Position [m]")
ax1_A.set_title("3-Link End-Effector Position vs. Time (Part A)")
ax1_A.legend()
ax1_A.grid(True)

fig2_A, ax2_A = plt.subplots()
ax2_A.plot(x_hist_A, y_hist_A, label="Actual Trajectory", linewidth=2)
ax2_A.plot(xd_hist_A, yd_hist_A, "--", label="Desired Trajectory", linewidth=2)
ax2_A.set_xlabel("x [m]")
ax2_A.set_ylabel("y [m]")
ax2_A.set_title("3-Link End-Effector Trajectory (Part A)")
ax2_A.legend()
ax2_A.axis("equal")
ax2_A.grid(True)

# Plots for Part B
fig1_B, ax1_B = plt.subplots()
ax1_B.plot(time_3link, x_hist_B, label="x(t) actual", linewidth=2)
ax1_B.plot(time_3link, xd_hist_B, "--", label="x(t) desired", linewidth=2)
ax1_B.plot(time_3link, y_hist_B, label="y(t) actual", linewidth=2)
ax1_B.plot(time_3link, yd_hist_B, "--", label="y(t) desired", linewidth=2)
ax1_B.set_xlabel("Time [s]")
ax1_B.set_ylabel("Position [m]")
ax1_B.set_title("3-Link End-Effector Position vs. Time (Part B)")
ax1_B.legend()
ax1_B.grid(True)

fig2_B, ax2_B = plt.subplots()
ax2_B.plot(x_hist_B, y_hist_B, label="Actual Trajectory", linewidth=2)
ax2_B.plot(xd_hist_B, yd_hist_B, "--", label="Desired Trajectory", linewidth=2)
ax2_B.set_xlabel("x [m]")
ax2_B.set_ylabel("y [m]")
ax2_B.set_title("3-Link End-Effector Trajectory (Part B)")
ax2_B.legend()
ax2_B.axis("equal")
ax2_B.grid(True)

# =============================================================================
# 7-DOF 3D MANIPULATOR (requires pinocchio)
# =============================================================================

try:
    import pinocchio as pin

    print("=" * 60)
    print("7-DOF 3D MANIPULATOR")
    print("=" * 60)

    # Build a 7-DOF Chain
    model = pin.Model()

    link_lengths = [0.3] * 7
    parent = 0

    for i, L in enumerate(link_lengths):
        axis = i % 3
        if axis == 0:
            jmodel = pin.JointModelRY()
        elif axis == 1:
            jmodel = pin.JointModelRX()
        else:
            jmodel = pin.JointModelRZ()

        Xtree = pin.SE3(np.eye(3), np.array([L, 0, 0]))
        joint_id = model.addJoint(parent, jmodel, Xtree, f"joint{i+1}")
        inertia = pin.Inertia(1.0, np.zeros(3), np.eye(3))
        model.appendBodyToJoint(joint_id, inertia, pin.SE3.Identity())
        parent = joint_id

    ee_joint_id = parent
    data = model.createData()

    print(f"Built model with {model.nq} DOFs and {model.njoints} joints")

    def rotation_matrix_to_axis_angle(R):
        cos_theta = (np.trace(R) - 1) / 2.0
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        if np.isclose(theta, 0.0):
            return np.zeros(3)
        if np.isclose(theta, np.pi):
            axis = np.sqrt((np.diag(R) + 1) / 2.0)
            axis *= np.sign([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
            return axis * theta
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(theta))
        return axis * theta

    # Time parameters
    dt = 0.05
    T_total = 30.0
    time_7dof = np.arange(0.0, T_total, dt)

    # Desired 3D helix
    radius_3d = 0.5
    z_speed = 0.05
    omega_3d = 0.1
    pos_des_3d = np.array([[radius_3d * np.cos(omega_3d * t), radius_3d * np.sin(omega_3d * t), z_speed * t] for t in time_7dof])

    # Storage
    q = np.zeros(model.nq)
    q_hist_7dof = np.zeros((len(time_7dof), model.nq))
    x_hist_7dof = np.zeros((len(time_7dof), 3))

    # Gains
    k_pos = 1.5
    k_rot = 1.5

    for i, t in enumerate(time_7dof):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, q)

        oMi = data.oMi[ee_joint_id]
        p_cur = oMi.translation
        R_cur = oMi.rotation

        p_des = pos_des_3d[i]
        R_des = np.eye(3)

        e_pos = p_des - p_cur
        R_err = R_des @ R_cur.T
        e_rot = rotation_matrix_to_axis_angle(R_err)

        J6 = data.J

        J_parse_7dof = jparse.JParse(
            J=J6,
            singular_direction_gain_position=1,
            singular_direction_gain_orientation=2,
            position_dimensions=3,
            angular_dimensions=3,
        )

        v_spatial = np.hstack((k_pos * e_pos, k_rot * e_rot))
        dq = J_parse_7dof @ v_spatial
        dq = dq.A1

        q += dq * dt
        q_wrapped = (q + np.pi) % (2 * np.pi) - np.pi

        q_hist_7dof[i] = q_wrapped
        x_hist_7dof[i] = p_cur

    # Compute errors for plotting
    pos_err = pos_des_3d - x_hist_7dof
    rot_err = np.zeros_like(pos_err)

    for i in range(len(time_7dof)):
        pin.forwardKinematics(model, data, q_hist_7dof[i])
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, q_hist_7dof[i])
        oMi = data.oMi[ee_joint_id]
        R_cur = oMi.rotation
        R_err = np.eye(3) @ R_cur.T
        rot_err[i] = rotation_matrix_to_axis_angle(R_err)

    # Plot errors
    fig_7dof, (ax_pos, ax_rot, ax_joints) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax_pos.plot(time_7dof, pos_err[:, 0], label="Error X", linewidth=2)
    ax_pos.plot(time_7dof, pos_err[:, 1], label="Error Y", linewidth=2)
    ax_pos.plot(time_7dof, pos_err[:, 2], label="Error Z", linewidth=2)
    ax_pos.set_ylabel("Position Error [m]")
    ax_pos.set_title("7-DOF End-Effector Position Error vs. Time")
    ax_pos.legend(loc="upper right")
    ax_pos.grid(True)

    ax_rot.plot(time_7dof, rot_err[:, 0], label="Error Roll (X)", linewidth=2)
    ax_rot.plot(time_7dof, rot_err[:, 1], label="Error Pitch (Y)", linewidth=2)
    ax_rot.plot(time_7dof, rot_err[:, 2], label="Error Yaw (Z)", linewidth=2)
    ax_rot.set_ylabel("Orientation Error [rad]")
    ax_rot.set_title("7-DOF End-Effector Orientation Error vs. Time")
    ax_rot.legend(loc="upper right")
    ax_rot.grid(True)

    for j in range(q_hist_7dof.shape[1]):
        ax_joints.plot(time_7dof, q_hist_7dof[:, j], label=f"Joint {j+1}", linewidth=1.5)
    ax_joints.set_xlabel("Time [s]")
    ax_joints.set_ylabel("Joint Angle [rad]")
    ax_joints.set_title("7-DOF Joint Angles vs. Time")
    ax_joints.legend(loc="upper right", ncol=2)
    ax_joints.grid(True)

    plt.tight_layout()

except ImportError:
    print("=" * 60)
    print("7-DOF 3D MANIPULATOR - SKIPPED (pinocchio not installed)")
    print("To install: pip install pin")
    print("=" * 60)

plt.show()
