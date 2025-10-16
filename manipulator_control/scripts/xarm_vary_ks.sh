#!/bin/bash

# Define methods and multiple phi gains
methods=("JParse")
ks_gains=("1" "2" "3" "4" "5")  # Multiple gains

# Define experiments
declare -A experiments

experiments["LineExtendedKeypoints_trajectory"]="roslaunch manipulator_control line_extended_singular_traj.launch robot:=xarm key_points_only_bool:=true frequency:=0.05 use_rotation:=false"
experiments["LineExtendedKeypoints_bag"]="roslaunch manipulator_control full_pose_bag_file.launch bag_name:=xarm_sim_vary_ks_\$METHOD\_gamma_0_1_line_extended_keypoints record_all:=false"

experiments["EllipseKeypoints_trajectory"]="roslaunch manipulator_control full_pose_trajectory.launch robot:=xarm key_points_only_bool:=true frequency:=0.05 use_rotation:=false"
experiments["EllipseKeypoints_bag"]="roslaunch manipulator_control full_pose_bag_file.launch bag_name:=xarm_sim_vary_ks_\$METHOD\_gamma_0_1_ellipse_keypoints record_all:=false"

# Function to start an experiment
run_experiment() {
    local experiment_name=$1
    local method=$2
    local ks=$3  # Capture ks

    # Generate modified method name with ks
    local method_with_gain="${method}_ks${ks}"

    # Get trajectory and bag commands with method substitution
    local trajectory_command=${experiments["${experiment_name}_trajectory"]}
    local bag_command=${experiments["${experiment_name}_bag"]}
    trajectory_command=${trajectory_command//\$METHOD/$method}
    bag_command=${bag_command//\$METHOD/$method_with_gain}  # Include ks in bag name

    echo "Starting experiment: $experiment_name with method: $method and ks: $ks"

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

    xterm -hold -e "roslaunch manipulator_control xarm_main_vel.launch is_sim:=true show_jparse_ellipses:=true jparse_gamma:=0.1 ks:=$ks method:=$method" &
    pid2=$!

    sleep 3  # Wait before starting bag file recording

    # Open xterm for bag recording (bag name includes phi_gain)
    xterm -hold -e "$bag_command" &
    pid3=$!

    # Wait 5 minutes (300 seconds) for the experiment to run
    # EDIT: we don't need that long
    sleep 200

    echo "Stopping current experiment (keeping xarm_launch.launch running)..."

    # Kill only the specific processes of this experiment
    kill $pid1 $pid2 $pid3 2>/dev/null

    sleep 5  # Small pause before the next experiment
}

for ks in "${ks_gains[@]}"; do
    for experiment in  "LineExtendedKeypoints" "EllipseKeypoints"; do
        for method in "${methods[@]}"; do
            run_experiment "$experiment" "$method" "$ks"
        done
    done
done

echo "All experiments completed!"