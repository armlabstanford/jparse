# 🦾 J-PARSE: Jacobian-based Projection Algorithm for Resolving Singularities Effectively in Inverse Kinematic Control of Serial Manipulators


### [Shivani Guptasarma](https://www.linkedin.com/in/shivani-guptasarma/), [Matthew Strong](https://peasant98.github.io/), [Honghao Zhen](https://www.linkedin.com/in/honghao-zhen/), and [Monroe Kennedy III](https://monroekennedy3.com/)


_In Submission, Transactions on Robotics_

<img
  src="images/splash.png"
  alt="JPARSE splash"
  style="width:100%;"
/>

<!-- ![JPARSE Concept diagram](images/jparse_concept_fig.png)
 -->

[![Project](https://img.shields.io/badge/Project_Page-J_PARSE-blue)](https://jparse-manip.github.io)
[![ArXiv](https://img.shields.io/badge/Arxiv-J_PARSE-red)](https://arxiv.org/abs/2505.00306)

---

## Installation

### Install from PyPI

```bash
pip install jparse-robotics
```

### Install from Source (Development)

```bash
git clone https://github.com/armlabstanford/jparse.git
cd jparse
pip install -e .
```

### With URDF/Pinocchio Support (Optional)

For loading robots from URDF files:

```bash
# Using conda (recommended)
conda install -c conda-forge pinocchio
pip install jparse-robotics

# Or pip only
pip install jparse-robotics pin
```

---

## Quick Start

### Basic Usage (Pure Algorithm)

```python
import numpy as np
import jparse_robotics as jparse

# Create solver
solver = jparse.JParseCore(gamma=0.1)

# Your Jacobian matrix
J = np.array([[-0.707, -0.707],
              [ 0.707,  0.707]])

# Compute J-PARSE pseudo-inverse
J_parse = solver.compute(J)

# Velocity control
desired_velocity = np.array([0.1, 0.1])
joint_velocities = J_parse @ desired_velocity
```

### With URDF Robot Model

```python
import numpy as np
import jparse_robotics as jparse

# Load robot from URDF
robot = jparse.Robot.from_urdf("robot.urdf", "base_link", "ee_link", gamma=0.1)

# Get J-PARSE at a configuration
q = np.zeros(robot.num_joints)
J_parse = robot.jparse(q)

# Velocity control
desired_velocity = np.array([0.1, 0, 0, 0, 0, 0])  # [vx, vy, vz, wx, wy, wz]
joint_velocities = J_parse @ desired_velocity
```

See `examples/` for more detailed examples.

### Interactive 2D Visualization

Try the interactive demo to see J-PARSE in action:

```bash
pip install matplotlib
python examples/visualization_2d.py
```

- **Click** anywhere to set a target position
- **Press 'j'** to toggle between J-PARSE and standard pseudo-inverse
- Watch how J-PARSE keeps joint velocities bounded near singularities!

---

## Pyroki Integration (3D Interactive Visualization)

We have integrated J-PARSE with [pyroki](https://github.com/chungmin99/pyroki) for interactive 3D visualization with real robot models!

<img
  src="examples/singularity_comparison.png"
  alt="J-PARSE vs Pseudo-inverse comparison"
  style="width:80%;"
/>

### Installation

```bash
pip install jparse-robotics pyroki viser robot-descriptions yourdfpy
```

### Run the 3D Visualization

```bash
# Clone the repo (if you haven't already)
git clone https://github.com/armlabstanford/jparse.git
cd jparse

# Run with xarm7 (default)
python examples/pyroki_3d_visualization.py

# Or choose another robot
python examples/pyroki_3d_visualization.py --robot panda
python examples/pyroki_3d_visualization.py --robot iiwa7
python examples/pyroki_3d_visualization.py --robot ur5
```

Then open **http://localhost:8080** in your browser.

### Features

- **Multiple robots**: xarm7, panda, iiwa7, iiwa14, ur5, ur10
- **Method switching**: Toggle between J-PARSE / Pseudo-inverse / Damped LS
- **Nullspace control**: Pulls joints toward home pose without affecting end-effector
- **Singular direction gain**: Helps exit singularities with controlled motion
- **Real-time metrics**: Manipulability, inverse condition number, joint velocities
- **Manipulability ellipsoid**: Visualize the arm's dexterity in real-time

### Controls

- **Drag the gizmo** to set target position
- **Adjust sliders** to tune J-PARSE gamma, nullspace gain, singular direction gain
- **Watch "Max Joint Vel"** explode with pseudo-inverse near singularities, stay bounded with J-PARSE!

---

## Quick Start with Docker

To build the Docker image for the our environment, we use VNC docker, which allows for a graphical user interface displayable in the browser.

### Use the Public Docker Image (Recommended)

We have created a public Docker image that you can pull! 
Steps:

```sh
docker pull peasant98/jparse
docker run --privileged -p 6080:80 --shm-size=512m -v <path to jparse repo>:/home/ubuntu/Desktop/jparse_ws/src peasant98/jparse
```

### Build the Image Yourself

You can build the docker image yourself! To do so, follow the below steps:
```sh
cd Docker
docker build -t jparse .
docker run --privileged -p 6080:80 --shm-size=512m -v <path to jparse repo>:/home/ubuntu/Desktop/jparse_ws/src jparse

```

## API Reference

### `jparse.JParseCore`

Pure J-PARSE algorithm. Only requires numpy.

```python
solver = jparse.JParseCore(gamma=0.1)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gamma` | float | 0.1 | Singularity threshold (0 < gamma < 1). Directions with σᵢ/σₘₐₓ < gamma are treated as singular. |

#### `solver.compute(jacobian, ...)`

Compute the J-PARSE pseudo-inverse of a Jacobian matrix.

```python
J_parse = solver.compute(J)
J_parse, nullspace = solver.compute(J, return_nullspace=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `jacobian` | ndarray | required | m × n Jacobian matrix |
| `singular_direction_gain_position` | float | 1.0 | Gain for position singular directions |
| `singular_direction_gain_angular` | float | 1.0 | Gain for angular singular directions |
| `position_dimensions` | int | None | Number of position rows (e.g., 3 for 3D) |
| `angular_dimensions` | int | None | Number of angular rows (e.g., 3 for 3D) |
| `return_nullspace` | bool | False | Also return nullspace projection matrix |

**Returns:**
- `J_parse` (ndarray): n × m J-PARSE pseudo-inverse matrix
- `nullspace` (ndarray, optional): n × n nullspace projection matrix

#### `solver.pinv(jacobian)`

Standard Moore-Penrose pseudo-inverse (for comparison).

**Returns:** n × m pseudo-inverse matrix

#### `solver.damped_least_squares(jacobian, damping=0.01)`

Damped least squares pseudo-inverse (for comparison).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `jacobian` | ndarray | required | m × n Jacobian matrix |
| `damping` | float | 0.01 | Damping factor λ |

**Returns:** n × m DLS pseudo-inverse matrix

---

### `jparse.Robot`

High-level robot interface with URDF support (requires Pinocchio).

```python
robot = jparse.Robot.from_urdf("robot.urdf", "base_link", "ee_link", gamma=0.1)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `urdf` | str | required | Path to URDF file or XML string |
| `base_link` | str | required | Name of base link |
| `end_link` | str | required | Name of end-effector link |
| `gamma` | float | 0.1 | J-PARSE singularity threshold |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_joints` | int | Number of actuated joints |
| `gamma` | float | Current singularity threshold (settable) |

#### `robot.jacobian(q)`

Compute the 6 × n geometric Jacobian.

**Returns:** 6 × n Jacobian matrix (rows 0-2: linear, rows 3-5: angular)

#### `robot.jparse(q, ...)`

Compute J-PARSE pseudo-inverse at configuration q.

```python
J_parse = robot.jparse(q)
J_parse = robot.jparse(q, position_only=True)  # 3D position only
J_parse, nullspace = robot.jparse(q, return_nullspace=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | ndarray | required | Joint configuration |
| `position_only` | bool | False | Use only position rows (3×n) |
| `return_nullspace` | bool | False | Also return nullspace matrix |
| `singular_direction_gain_position` | float | 1.0 | Position gain |
| `singular_direction_gain_angular` | float | 1.0 | Angular gain |

**Returns:** J-PARSE pseudo-inverse (and optionally nullspace)

#### `robot.forward_kinematics(q)`

Compute end-effector pose.

**Returns:** `(position, rotation)` - 3D position and 3×3 rotation matrix

#### `robot.manipulability(q)`

Compute Yoshikawa's manipulability measure: √det(JJᵀ)

**Returns:** float (higher = better conditioned)

#### `robot.inverse_condition_number(q)`

Compute σₘᵢₙ/σₘₐₓ of the Jacobian.

**Returns:** float in [0, 1] (0 = singular, 1 = isotropic)

---

### Utility Functions

```python
# Manipulability measure
m = jparse.manipulability_measure(J)  # √det(JJᵀ)

# Inverse condition number
icn = jparse.inverse_condition_number(J)  # σₘᵢₙ/σₘₐₓ
```

---

### ROS Integration

```python
from jparse_robotics.ros import ROSRobot

robot = ROSRobot.from_parameter_server("base_link", "ee_link", gamma=0.1)
robot.publish_ellipsoids(q, end_effector_pose)  # Visualize in RViz
```

---

### Dependencies (ROS)
*Note: these are handled in the Docker image directly, and are already installed!*

1. [Catkin Simple](https://github.com/catkin/catkin_simple): https://github.com/catkin/catkin_simple
2. [HRL KDL](https://github.com/armlabstanford/hrl-kdl): https://github.com/armlabstanford/hrl-kdl 


## Running Velocity Control (XArm) Example

### Simulation
To run the XArm in simulation, first run
```bash
roslaunch manipulator_control xarm_launch.launch
```

#### Run Desired Trajectory
Next, run one of the trajectory generation scripts. This can either be the ellipse that has poses within and on the boundary of the reachable space of the arm (to test stability):
```bash
roslaunch manipulator_control full_pose_trajectory.launch robot:=xarm
```
or for passing through the type-2 singularity (passing directly above the base link): 
```bash
roslaunch manipulator_control se3_type_2_singular_traj.launch robot:=xarm
```
To have more control over keypoints (stop at major and minor axis of ellipse), run
```bash
roslaunch manipulator_control se3_type_2_singular_traj.launch robot:=xarm key_points_only_bool:=true frequency:=0.1 use_rotation:=false
```
or 
```bash
roslaunch manipulator_control full_pose_trajectory.launch robot:=xarm key_points_only_bool:=true frequency:=0.1 use_rotation:=false
```
(here frequency specifies how much time is spent at each keypoint).
or 
```bash
roslaunch manipulator_control line_extended_singular_traj.launch robot:=xarm key_points_only_bool:=true frequency:=0.2 use_rotation:=false
```
(line trajectory that goes from over the robot, to out of reach in front of the robot.)

#### Run Control Method
```bash
roslaunch manipulator_control xarm_main_vel.launch is_sim:=true show_jparse_ellipses:=true phi_gain_position:=2.0 phi_gain_angular:=2.0  jparse_gamma:=0.2 method:=JParse 
```

The arguments are 
| Parameter   | Attribute Description |
|------------|----------------------|
| `is_sim`   | Boolean for sim or real |
| `show_jparse_ellipses`   | Boolean for showing position JParse ellipsoids (for that method only) in rviz |
| `phi_gain_position`   | Kp gain for JParse singular direction position |
| `phi_gain_angular`   | Kp gain for JParse singular direction orientation |
| `jparse_gamma`   | JParse threshold value gamma |
| `method`   |  "JParse", "JacobianPseudoInverse" (basic); "JacobianDampedLeastSquares"; "JacobianProjection"; "JacobianDynamicallyConsistentInverse" |


## Real Robot Velocity Control 
### XArm Velocity Control Example
To run on the physical Xarm, the update is to use
```bash
roslaunch manipulator_control xarm_main_vel.launch is_sim:=false method:=JParse 
```
Recommended methods for physical system (to avoid unsafe motion) is: "JParse", "JacobianDampedLeastSquares"


### Kinova Gen 3 Velocity Control Example
Run the Kinova environment
```bash
roslaunch manipulator_control kinova_gen3.launch
```

#### Select Trajectory
Run desired Line Extended keypoints trajectory:
```bash
roslaunch manipulator_control line_extended_singular_traj.launch robot:=kinova key_points_only_bool:=true frequency:=0.1 use_rotation:=false
```

Run elliptical keypoints trajectory
```bash
roslaunch manipulator_control full_pose_trajectory.launch robot:=kinova key_points_only_bool:=true frequency:=0.06 use_rotation:=false
```

#### Select Control
Run the Method: 
```bash
roslaunch manipulator_control kinova_vel_control.launch is_sim:=true show_jparse_ellipses:=true phi_gain_position:=2.0 phi_gain_angular:=2.0  jparse_gamma:=0.2 method:=JParse 
```

## Running JParse with the SpaceMouse controller

You can also run JParse with a human teleoperator using a SpaceMouse controller. This will allow for a fun sandbox to verify JParse. 

We plan to extend this to a simple learning policy as well. The code for that (collecting data, training a policy, and running inference) will be published soon!

To run, you can run

```sh
# run the real robot 
roslaunch manipulator_control xarm_real_launch.launch using_spacemouse:=true

# run the jparse method with or without jparse control
roslaunch xarm_main_vel.launch use_space_mouse:=true use_space_mouse_jparse:={true|false}

# run the spacemouse example!! Make sure the use_native_xarm_spacemouse argument is OPPOSITE of use_space_mouse_jparse.
roslaunch xarm_spacemouse_teleop.launch use_native_xarm_spacemouse:={true|false}
```
   
## Run C++ JParse publisher and service
This allows for publishing JParse components and visualizing using a C++ script
```bash
roslaunch manipulator_control jparse_cpp.launch jparse_gamma:=0.2  singular_direction_gain_position:=2.0 singular_direction_gain_angular:=2.0
```

The arguments are 
| Parameter   | Attribute Description |
|------------|----------------------|
| `namespace`   | namespace of the robot (e.g. xarm) |
| `base_link_name`   | Baselink frame |
| `end_link_name`   | end-effector frame |
| `jparse_gamma`   | JParse gamma value (0,1) |
| `singular_direction_gain_position`   | gains in singular direction for position |
| `singular_direction_gain_angular`   |  gains in singular direction for orientation |
| `run_as_service` | (boolean) true/false | 

For running as a service: 
```bash
roslaunch manipulator_control jparse_cpp.launch run_as_service:=true
```
Then to run service from a terminal (Xarm example): 

```bash
rosservice call /jparse_srv "gamma: 0.2
singular_direction_gain_position: 2.0
singular_direction_gain_angular: 2.0
base_link_name: 'link_base'
end_link_name: 'link_eef'" 
```
To see versatility, simply change the kinematic chain for the JParse solution for that segment. To view options for your kinematic tree:
```bash
rosrun rqt_tf_tree rqt_tf_tree
```

To test with the robot (using a python node to control the arm, with JParse coming from C++), first run script above, then:
```bash
roslaunch manipulator_control xarm_python_using_cpp.launch is_sim:=true phi_gain_position:=2.0 phi_gain_angular:=2.0  jparse_gamma:=0.2 use_service_bool:=true 
```
This has same parameters as the python version, but with the service versus message option. Message is faster/cleaner, but service is very versatile: 
| Parameter   | Attribute Description |
|------------|----------------------|
| `use_service_bool`   | True: use service, False: use message|
| `jparse_gamma`   | JParse gain (0,1)|
| `phi_gain_position`   | gain on position component|
| `phi_gain_angular`   | gain on angular component|
| `is_sim`   | use of sim versus real (boolean)|
