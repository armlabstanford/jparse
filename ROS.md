# ROS Integration for J-PARSE

This document covers running J-PARSE with ROS for real robot control.

## Dependencies

*Note: These are handled in the Docker image directly and are already installed!*

1. [Catkin Simple](https://github.com/catkin/catkin_simple)
2. [HRL KDL](https://github.com/armlabstanford/hrl-kdl)

---

## XArm Velocity Control

### Simulation

To run the XArm in simulation:

```bash
roslaunch manipulator_control xarm_launch.launch
```

### Trajectories

**Ellipse trajectory** (poses within and on workspace boundary):
```bash
roslaunch manipulator_control full_pose_trajectory.launch robot:=xarm
```

**Type-2 singularity trajectory** (passing directly above base link):
```bash
roslaunch manipulator_control se3_type_2_singular_traj.launch robot:=xarm
```

**Keypoint control** (stop at major/minor axis of ellipse):
```bash
roslaunch manipulator_control se3_type_2_singular_traj.launch robot:=xarm key_points_only_bool:=true frequency:=0.1 use_rotation:=false
```

```bash
roslaunch manipulator_control full_pose_trajectory.launch robot:=xarm key_points_only_bool:=true frequency:=0.1 use_rotation:=false
```

**Line trajectory** (from over robot to out of reach):
```bash
roslaunch manipulator_control line_extended_singular_traj.launch robot:=xarm key_points_only_bool:=true frequency:=0.2 use_rotation:=false
```

### Control Method

```bash
roslaunch manipulator_control xarm_main_vel.launch is_sim:=true show_jparse_ellipses:=true phi_gain_position:=2.0 phi_gain_angular:=2.0 jparse_gamma:=0.2 method:=JParse
```

| Parameter | Description |
|-----------|-------------|
| `is_sim` | Boolean for sim or real |
| `show_jparse_ellipses` | Show position JParse ellipsoids in rviz |
| `phi_gain_position` | Kp gain for JParse singular direction position |
| `phi_gain_angular` | Kp gain for JParse singular direction orientation |
| `jparse_gamma` | JParse threshold value gamma |
| `method` | "JParse", "JacobianPseudoInverse", "JacobianDampedLeastSquares", "JacobianProjection", "JacobianDynamicallyConsistentInverse" |

### Real Robot

```bash
roslaunch manipulator_control xarm_main_vel.launch is_sim:=false method:=JParse
```

Recommended methods for physical system: **JParse**, **JacobianDampedLeastSquares**

---

## Kinova Gen 3 Velocity Control

### Setup

```bash
roslaunch manipulator_control kinova_gen3.launch
```

### Trajectories

**Line Extended keypoints:**
```bash
roslaunch manipulator_control line_extended_singular_traj.launch robot:=kinova key_points_only_bool:=true frequency:=0.1 use_rotation:=false
```

**Elliptical keypoints:**
```bash
roslaunch manipulator_control full_pose_trajectory.launch robot:=kinova key_points_only_bool:=true frequency:=0.06 use_rotation:=false
```

### Control

```bash
roslaunch manipulator_control kinova_vel_control.launch is_sim:=true show_jparse_ellipses:=true phi_gain_position:=2.0 phi_gain_angular:=2.0 jparse_gamma:=0.2 method:=JParse
```

---

## SpaceMouse Teleoperation

Run JParse with a SpaceMouse controller for interactive teleoperation:

```sh
# Run the real robot
roslaunch manipulator_control xarm_real_launch.launch using_spacemouse:=true

# Run jparse method with or without jparse control
roslaunch xarm_main_vel.launch use_space_mouse:=true use_space_mouse_jparse:={true|false}

# Run spacemouse (use_native_xarm_spacemouse should be OPPOSITE of use_space_mouse_jparse)
roslaunch xarm_spacemouse_teleop.launch use_native_xarm_spacemouse:={true|false}
```

---

## C++ JParse Publisher and Service

Publish JParse components and visualize using C++:

```bash
roslaunch manipulator_control jparse_cpp.launch jparse_gamma:=0.2 singular_direction_gain_position:=2.0 singular_direction_gain_angular:=2.0
```

| Parameter | Description |
|-----------|-------------|
| `namespace` | Namespace of the robot (e.g. xarm) |
| `base_link_name` | Baselink frame |
| `end_link_name` | End-effector frame |
| `jparse_gamma` | JParse gamma value (0,1) |
| `singular_direction_gain_position` | Gains in singular direction for position |
| `singular_direction_gain_angular` | Gains in singular direction for orientation |
| `run_as_service` | Boolean true/false |

### Running as a Service

```bash
roslaunch manipulator_control jparse_cpp.launch run_as_service:=true
```

Call the service (XArm example):

```bash
rosservice call /jparse_srv "gamma: 0.2
singular_direction_gain_position: 2.0
singular_direction_gain_angular: 2.0
base_link_name: 'link_base'
end_link_name: 'link_eef'"
```

View kinematic tree options:
```bash
rosrun rqt_tf_tree rqt_tf_tree
```

### Python Node with C++ JParse

```bash
roslaunch manipulator_control xarm_python_using_cpp.launch is_sim:=true phi_gain_position:=2.0 phi_gain_angular:=2.0 jparse_gamma:=0.2 use_service_bool:=true
```

| Parameter | Description |
|-----------|-------------|
| `use_service_bool` | True: use service, False: use message |
| `jparse_gamma` | JParse gain (0,1) |
| `phi_gain_position` | Gain on position component |
| `phi_gain_angular` | Gain on angular component |
| `is_sim` | Sim versus real (boolean) |
