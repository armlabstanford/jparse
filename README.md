# J-PARSE: Jacobian-based Projection Algorithm for Resolving Singularities Effectively in Inverse Kinematic Control of Serial Manipulators

<img
  src="images/jparse_splash.png"
  alt="JPARSE splash"
  style="width:100%;"
/>

![JPARSE Concept diagram](images/jparse_concept_fig.png)



## Quick Start with Docker

To build the Docker image for the task:

```sh
cd Docker
docker build -t jparsee .
docker run --privileged -p 6080:80 --shm-size=512m -v <path to jparse workspace>:/home/ubuntu/Desktop/jparse peasant98/vnc-gazebo-ros

```

### Dependences
*Note: these are handled in Docker image directly.*

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

We plan to extend this to a simple learning policy as well.

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
To see versitility, simply change the kinematic chain for the JParse solution for that segment. To view options for your kinematic tree:
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



## Simulated Torque Control (under development - future work) 
*We are actively working on torque control for J-PARSE that provides all of the rigorous performance of velocity control. We do not recommend running this on a real robot yet. But this provides a preview of performance in torque control scenarios.*

For torque, we use the Panda robot. To run the Panda: 
```bash
roslaunch manipulator_control franka_launch.launch 
```

#### Trajectory Selection
Next run the desired trajectory generator, ellipse which tests the boundaries of motion: 
```bash
roslaunch manipulator_control full_pose_trajectory.launch robot:=panda
```
or type-2 singularity (passing above the base):
```bash
roslaunch manipulator_control se3_type_2_singular_traj.launch robot:=panda
```
or line-extended keypoints (from above the base, to reaching out in front beyond reach)
```bash
roslaunch manipulator_control line_extended_singular_traj.launch robot:=panda key_points_only_bool:=true frequency:=0.05 use_rotation:=false
```

#### Select Control
Finally run the control with the desired method:
```bash
roslaunch manipulator_control panda_main_torque.launch is_sim:=true phi_gain_position:=3.0 phi_gain_angular:=3.0 jparse_gamma:=0.2 method:=JParse
```

The method options are: "JParse", "JacobianPseudoInverse" (basic); "JacobianDampedLeastSquares"; "JacobianTranspose"; "JacobianNullspaceDecoupled"

