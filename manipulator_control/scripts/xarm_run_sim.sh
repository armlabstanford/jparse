#!/bin/bash

# list methods to use
# methods=("JParse" "JacobianProjection" "JacobianSafetyProjection" "JacobianDampedLeastSquares_0_1" "JacobianDampedLeastSquares_0_055" "JacobianSafety" "JacobianPseudoInverse")
# methods=("JacobianPseudoInverse")
# methods=("JacobianDampedLeastSquares_0_1" "JacobianSafetyProjection" "JacobianProjection") # "JacobianDampedLeastSquares_0_05" "JacobianDampedLeastSquares_0_03" "JacobianPseudoInverse")
methods=("JacobianProjection")
# methods=("JParse" "JacobianProjection" "JacobianSafetyProjection" "JacobianSafety")

declare -A experiments 
experiments["LineExtendedKeypoints_trajectory"]="roslaunch manipulator_control line_extended_singular_traj.launch robot:=xarm key_points_only_bool:=true frequency:=0.05 use_rotation:=false"
experiments["LineExtendedKeypoints_bag"]="roslaunch manipulator_control full_pose_bag_file.launch bag_name:=xarm_sim_gamma_0_2_\$METHOD\_line_extended_keypoints record_all:=false"

experiments["EllipseKeypoints_trajectory"]="roslaunch manipulator_control full_pose_trajectory.launch robot:=xarm key_points_only_bool:=true frequency:=0.05 use_rotation:=false"
experiments["EllipseKeypoints_bag"]="roslaunch manipulator_control full_pose_bag_file.launch bag_name:=xarm_sim_gamma_0_2_\$METHOD\_ellipse_keypoints record_all:=false"

run_experiment() {
    local experiment_name=$1
    local method=$2

    # Get trajectory and bag commands with method substitution
    local trajectory_command=${experiments["${experiment_name}_trajectory"]}
    local bag_command=${experiments["${experiment_name}_bag"]}
    trajectory_command=${trajectory_command//\$METHOD/$method}
    bag_command=${bag_command//\$METHOD/$method}

    echo "Starting experiment: $experiment_name with method: $method"

    # Start xarm_launch.launch if not already running
    if ! pgrep -f "xarm_launch.launch" > /dev/null; then
        echo "Starting xarm_launch.launch..."
        xterm -hold -e "roslaunch manipulator_control xarm_launch.launch" &
        sleep 5  # Ensure Franka's ROS nodes are fully initialized
    fi

    # Open xterm for trajectory execution
    xterm -hold -e "$trajectory_command" &
    pid1=$!

    sleep 2  # Allow initialization

    local damping=0.1
    local jparse_gamma=0.07

    # set different lambda values for DLS1 and DLS2
    if [ "$method" == "JacobianDampedLeastSquares_0_1" ]; then
        damping=0.1
        method="JacobianDampedLeastSquares"
    elif [ "$method" == "JacobianDampedLeastSquares_0_06" ]; then
        damping=0.06
        method="JacobianDampedLeastSquares"
    elif [ "$method" == "JacobianDampedLeastSquares_0_055" ]; then
        damping=0.055
        method="JacobianDampedLeastSquares"
    elif [ "$method" == "JacobianDampedLeastSquares_0_03" ]; then
        damping=0.03
        method="JacobianDampedLeastSquares"
    elif [ "$method" == "JacobianDampedLeastSquares_0_08" ]; then
        damping=0.08
        method="JacobianDampedLeastSquares"
    fi

    # set different gamma values for JParse ablations
    if [ "$method" == "JParse" ]; then
        # jparse_gamma=0.07
        # jparse_gamma=0.1
        jparse_gamma=0.2
    elif [ "$method" == "JacobianProjection" ]; then
        # jparse_gamma=0.07
        # jparse_gamma=0.1
        jparse_gamma=0.2
    elif [ "$method" == "JacobianSafety" ]; then
        # jparse_gamma=0.07
        # jparse_gamma=0.1
        jparse_gamma=0.2
    elif [ "$method" == "JacobianSafetyProjection" ]; then
        # jparse_gamma=0.07
        # jparse_gamma=0.1
        jparse_gamma=0.2
    fi

    sleep 10

    xterm -hold -e "roslaunch manipulator_control xarm_main_vel.launch jparse_gamma:=$jparse_gamma lambda:=$damping show_jparse_ellipses:=true  is_sim:=true method:=$method" &
    pid2=$!

    sleep 3  # Wait before starting bag file recording

    # Open xterm for bag recording
    xterm -hold -e "$bag_command" &
    pid3=$!

    # Wait 5 minutes (300 seconds) for the experiment to run
    sleep 200  

    echo "Stopping current experiment..."

    # Kill only the specific processes of this experiment
    kill $pid1 $pid2 $pid3 2>/dev/null

    sleep 5  # Small pause before the next experiment
}

for experiment in "LineExtendedKeypoints" "EllipseKeypoints"; do # "LineExtendedKeypoints" EllipseKeypoints
    for method in "${methods[@]}"; do
        run_experiment "$experiment" "$method"
    done
done

echo "All experiments completed!"