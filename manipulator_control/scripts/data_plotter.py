#!/usr/bin/env python3

import rospy
import rosbag
import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True
plt.rcParams['axes.labelsize'] = 16  # Font size for axis labels
plt.rcParams['xtick.labelsize'] = 14 # Font size for x-axis tick labels
plt.rcParams['ytick.labelsize'] = 14 # Font size for y-axis tick labels
plt.rcParams['font.weight'] = 'bold'

import mpld3 #for saving html plots
import plotly.graph_objects as go
from plotly import express as px  # Import color sets


from mpl_toolkits.mplot3d import Axes3D
from geometry_msgs.msg import PoseStamped, Vector3
from std_msgs.msg import Float64, Header
from sensor_msgs.msg import JointState

import os
import re
import rospkg


shading_colors = {
    'light_red': '#FFCCCC',     # Your existing light red
    'light_green': '#CCFFCC',   # Your existing light green
    'light_blue': '#CCE5FF',    # Light blue
    'light_yellow': '#FFFFCC',  # Light yellow
    'light_purple': '#E6CCFF',  # Light purple
    'light_orange': '#FFE5CC',  # Light orange
    'light_teal': '#CCFFFF',    # Light teal
    'light_pink': '#FFCCFF',    # Light pink
    'light_gray': '#EEEEEE'     # Light gray
}

CB_colors = {
    'Blue': '#377EB8',
    'LightBlue': '#5FA2D4',  # Lighter version of blue for plots
    'Orange': '#FF7F00',
    'Green': '#4DAF4A',
    'DarkGreen': '#2E7D2E',  # Darker green for plots
    'Pink': '#F781BF',
    'Brown': '#A65628',
    'Purple': '#984EA3',
    'Gray': '#999999',
    'Red': '#E41A1C',
    'Yellow': '#DEDE00'
}

METHOD_TO_PLT_NAME = {
    'JacobianDampedLeastSquares_03_kp': "DLS λ = 0.03",
    'JacobianDampedLeastSquares_05_kp': "DLS λ = 0.05",
    'JacobianDampedLeastSquares': "DLS",
    'JacobianDampedLeastSquares_0_1': "DLS λ = 0.1",
    'JacobianDampedLeastSquares_0_08': "DLS λ = 0.08",
    'JacobianDampedLeastSquares_0_06': "DLS λ = 0.06",
    'JacobianDampedLeastSquares_0_05': "DLS λ = 0.05",
    'JParse': "JParse",
    "JParse_xarm_real_gamma_0_07_freq_0_05": "JParse (γ=0.07, f=0.05)",
    "JParse_xarm_real_gamma_0_07_freq_0_06": "JParse (γ=0.07, f=0.06)",
    "JParse_Kp10_gamma_01": "JParse Kp=10 γ=1",
    "JParse_Kp20_gamma_01": "JParse Kp=20 γ=1",
    "JParse_Kp10_gamma_02": "JParse Kp=10 γ=2",
    "JParse_Kp20_gamma_02": "JParse Kp=2 γ=0.2",
    "JParse_Kp30_gamma_01": "JParse Kp=30 γ=1",
    "JParse_Kp30_gamma_02": "JParse Kp=30 γ=2",
    "JParse_Kp40_gamma_01": "JParse Kp=40 γ=1",
    "JParse_Kp40_gamma_02": "JParse Kp=40 γ=2",
    "JParse_Kp50_gamma_01": "JParse Kp=50 γ=1",
    "JParse_Kp50_gamma_02": "JParse Kp=50 γ=2",
    "JParse_Kp1_gamma_01": "JParse Kp=1",
    "JParse_Kp2_gamma_01": "JParse Kp=2",
    "JParse_Kp3_gamma_01": "JParse Kp=3",
    "JParse_Kp4_gamma_01": "JParse Kp=4",
    "JParse_Kp5_gamma_01": "JParse Kp=5",

    # Vary methods for parameter sweeps
    "vary_ks_JParse_ks1_gamma_0_1": "JParse ks=1",
    "vary_ks_JParse_ks2_gamma_0_1": "JParse ks=2",
    "vary_ks_JParse_ks3_gamma_0_1": "JParse ks=3",
    "vary_ks_JParse_ks4_gamma_0_1": "JParse ks=4",
    "vary_ks_JParse_ks5_gamma_0_1": "JParse ks=5",

    # Puma methods
    "adls_gimbalcont_lambda_0_1_w_0_02": "ADLS λ=0.1 w=0.02",
    "DLS_gimbalcont_lambda_0_1": "DLS λ=0.1",
    "JParse_gimbalcont_gamma_0_1": "JParse γ=0.1",

    "JacobianPseudoInverse": "PseudoInverse",
    "JacobianProjection": "Jacobian Projection",
    "JParseKp2": "JParse",
    "JParseKp1": "JParse Kp=1",
    "JacobianDampedLeastSquaresPoint1": "DLS λ = 0.1",
    "JacobianDampedLeastSquaresPoint05": "DLS λ = 0.05",
    "DampedLeastSquares_Point03": "DLS λ = 0.03",
    "DampedLeastSquares_Point05": "DLS λ = 0.05",
    "JacobianDampedLeastSquares_Point03": "DLS λ = 0.03",
    "JacobianDampedLeastSquares_Point05": "DLS λ = 0.05",
    "JacobianProjection": "Jacobian Projection",
    "JacobianSafetyProjection": "Jacobian Safety Projection",
    "JacobianSafety": "Jacobian Safety",
    "JacobianDampedLeastSquaresPoint03": "DLS λ = 0.03",
    "JacobianDampedLeastSquaresPoint05": "DLS λ = 0.05",
    "JacobianDampedLeastSquaresPoint1": "DLS λ = 0.1",
}

class BagFileProcessor:
    def __init__(self):
        rospy.init_node('bag_file_processor', anonymous=True)

        rospack = rospkg.RosPack()
        # Get the bag directory
        package_path = rospack.get_path('manipulator_control')

        #Access parameter to read xarm (Velocity) or panda (Torque) bagfiles
        self.robot_type = rospy.get_param('~robot_type', 'xarm') #Options are xarm or panda
        self.save_figures_bool = rospy.get_param('~save_figures', True)
        self.modified_bag_bool = rospy.get_param('~modified_bag', False)
        self.plot_all_methods = False


        self.highlight_regions = True
        self.xlim = 95
        # is many methods is true, we will only show two legends
        self.many_methods = False
        self.interval = 20

        # Time range for plotting (in seconds, relative to aligned time)
        self.plot_start_time = 0  # Start time for plotting
        self.plot_end_time = 1000  # End time for plotting (set to large value to use all data)

        # Max y-axis value for error plots (set to None to use automatic scaling)
        self.max_position_error_y = None  # Maximum y-axis value for position error (None = auto)
        self.max_orient_error_y = None  # Maximum y-axis value for orientation error

        if self.robot_type == "xarm":
            self.short_method_list = ["JParse_xarm_real_line_extended_gamma_0_07_freq_0_06"] #PAPER FIG

        elif self.robot_type == "panda":
            self.short_method_list = ["JParse_Kp15_gamma_01", "JParse_Kp15_gamma_02" , "JacobianDampedLeastSquares"]

        if self.robot_type == "xarm":
            self.data_figure_folder = package_path + "/data_figures/xarm_figures"
            local_directory = "/xarm/"

            self.methods = ["JParse_xarm_real_gamma_0_07_freq_0_06"]

            self.trajectories = ["line_extended"]
            self.traj_name_plot = dict(zip(self.trajectories, [
                 "Line Extended Keypoint"
            ]))

        elif self.robot_type == "xarm_sim":
            self.data_figure_folder = package_path + "/data_figures/xarm_sim_figures"
            local_directory = "/xarm_sim/"

            self.methods = ["JacobianDampedLeastSquares_0_06","JacobianDampedLeastSquares_0_08",
                            "JacobianDampedLeastSquares_0_1",
                            "JParse"
                            ]

            # self.methods = ["JacobianPseudoInverse", "JParse", ]

            self.trajectories = ["line_extended_keypoints"]
            self.traj_name_plot = dict(zip(self.trajectories, [
                "Line Extended Keypoints"
            ]))


        elif self.robot_type == "panda":
            #panda figures
            self.data_figure_folder = package_path + "/data_figures/panda_figures"
            #Panda Torque bag
            # local_directory = "/new_panda_sim_bags/kpx_15_kpth_15"
            local_directory = "/kpx_15_kpth_15_variable_retraction_gain/kpx_15_kpth_15_variable_retraction_gain"
            gain_select_variable = "kpx15kpth15" #"kpx15kpth5"; 

            # local_directory = "/new_panda_sim_bags/kpx_15_kpth_15_original"

            self.methods = [
                "JParse_Kp15_gamma_01", "JParse_Kp15_gamma_02", "JacobianDampedLeastSquares"]#, "JacobianNullspaceDecoupled"]
            self.methods = self.short_method_list

            self.trajectories = [
                "ellipse", "ellipse_keypoints", "line_extended_keypoints"
            ]   

            self.trajectories = [
                "ellipse_keypoints", "line_extended_keypoints"
            ]
            self.trajectories = [
                "line_extended_keypoints"
            ]     
            self.traj_name_plot = dict(zip(self.trajectories, [
                 "Line Extended Keypoints"
            ]))


            #torque control with new intermediate control law for joint4

        elif self.robot_type == "kinova":
            self.data_figure_folder = package_path + "/data_figures/kinova_figures"
            local_directory = "/kinova/"

            self.trajectories = [
                "goal_reaching"
            ]   

            self.traj_name_plot = dict(zip(self.trajectories, [
                 "Straight goal_reaching"
            ]))

            self.methods = [
                "dls_lambda_0_01", "adls_lambda_0_1_w0_0_02", "jparse_gamma_0_1"]
            
            self.short_method_list = self.methods

            self.time_adjustments = {}

        elif self.robot_type == "puma":
            self.data_figure_folder = package_path + "/data_figures/puma_figures"
            local_directory = "/puma/"

            # Disable shading for puma plots
            self.highlight_regions = False

            self.trajectories = [
                "cont"
            ]

            self.traj_name_plot = dict(zip(self.trajectories, [
                 "Continuous"
            ]))

            self.methods = [
                "adls_gimbalcont_lambda_0_1_w_0_02",
                "DLS_gimbalcont_lambda_0_1",
                "JParse_gimbalcont_gamma_0_1",]#, "JacobianNullspaceDecoupled"]


            self.short_method_list = self.methods

        elif self.robot_type == "xarm_real":
            #executing on real xarm
            self.data_figure_folder = package_path + "/data_figures/xarm_real_figures"
            local_directory = "/xarm_real/" #for ellipse keypoint and line keypoint
            # local_directory = "/demonstration" #for space-mouse control
            self.methods = [
                # "JParse_Kp10_gamma_02", 
                "JParse"
            ]
            self.trajectories = [
                "cartesian"
            ]
            self.traj_name_plot = dict(zip(self.trajectories, [
                "Teleoperation"
            ]))


        elif self.robot_type == "xarm_real_spacemouse":
            self.data_figure_folder = package_path + "/data_figures/xarm_real_figures"
            local_directory = "/xarm_real_bags/demonstration" #for space-mouse control

            self.methods = ["default", "jparse"]
            self.short_method_list = ["default", "jparse"]
            self.trajectories = ["cartesian"]#, "cartesian_2"]
            self.traj_name_plot = dict(zip(self.trajectories, [
                "Cartesian"
            ]))
            self.time_adjustments = {
                "xarm_spacemouse_default_cartesian": {"start_time": 0.0, "end_time": 200.0},
                # "xarm_spacemouse_default_cartesian_2": {"start_time": 0.0, "end_time": 200.0},
                "xarm_spacemouse_jparse_cartesian": {"start_time": 0.0, "end_time": 200.0},
                # "xarm_spacemouse_jparse_cartesian_1": {"start_time": 0.0, "end_time": 200.0},
            }

        else:
            rospy.logerr("Invalid robot type specified. Please use 'xarm' or 'panda'.")
            return

        # Get the path to the local directory
        self.bag_dir = rospy.get_param('~bag_directory', package_path + "/bag" + local_directory)
        self.process_bag_files()
    
    def extract_topic_data(self, bag, topic, msg_type):
        print(f"\n{'='*60}")
        print(f"EXTRACTING TOPIC DATA: {topic}")
        print(f"Message Type: {msg_type.__name__}")
        print(f"{'='*60}")

        timestamps = []
        data = []
        for _, msg, t in bag.read_messages(topics=[topic]):
            timestamps.append(t.to_sec())
            if msg_type == PoseStamped:
                data.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

            elif msg_type == Vector3:
                data.append([msg.x, msg.y, msg.z])
            elif msg_type == Float64:
                data.append(msg.data)
            elif msg_type == Header:
                data.append(msg.stamp.to_sec())
            elif msg_type == JointState:
                # For JointState, extract the second joint's data (index 1)
                # Check if there are enough joints in the message
                if len(msg.position) > 1:
                    data.append(msg.position[1])
                else:
                    # If there's no second joint, use a default value (e.g., 0.0)
                    # or handle the error as needed
                    data.append(0.0)
                    rospy.logwarn(f"JointState on topic {topic} doesn't have a second joint position")

        print(f"✓ Extracted {len(timestamps)} messages from {topic}")
        if len(timestamps) > 0:
            print(f"  Time range: {timestamps[0]:.2f}s to {timestamps[-1]:.2f}s")
        print(f"{'='*60}\n")

        return np.array(timestamps), np.array(data) if data else (np.array([]), np.array([]))

    def align_time(self, times, bag_name=None):
        if times.size == 0:
            return times, 0    
        aligned_times = times - times[0]
        stop_time = aligned_times[-1]
        return aligned_times, stop_time

    def format_method_name(self, method):
        '''
        Format the method name for plotting. Get the name from the dictionary
        '''
        return METHOD_TO_PLT_NAME.get(method, method)

    def process_bag_files(self):

        print(f"\n{'█'*60}")
        print(f"█ STARTING BAG FILE PROCESSING")
        print(f"█ Robot type: {self.robot_type}")
        print(f"█ Bag directory: {self.bag_dir}")
        print(f"█ Trajectories to process: {self.trajectories}")
        print(f"█ Methods to process: {self.methods}")
        print(f"█ Save figures: {self.save_figures_bool}")
        print(f"{'█'*60}\n")

        for traj_idx, traj in enumerate(self.trajectories, 1):

            print(f"\n{'█'*60}")
            print(f"█ PROCESSING TRAJECTORY {traj_idx}/{len(self.trajectories)}: {traj}")
            print(f"{'█'*60}\n")

            extracted_data = {}
            for method_idx, method in enumerate(self.methods, 1):
                print(f"\n[Method {method_idx}/{len(self.methods)}] Processing method: {method}")
                print(f"-" * 60)
                if self.robot_type == "xarm":
                    if self.modified_bag_bool:
                        bag_path = os.path.join(self.bag_dir, f"{method}_{traj}_modified.bag")
                        bag_name = f"{method}_{traj}_modified"
                    else:
                        bag_path = os.path.join(self.bag_dir, f"{method}_{traj}.bag")
                        bag_name = f"{method}_{traj}"
                elif self.robot_type == "xarm_sim":
                    bag_path = os.path.join(self.bag_dir, f"xarm_sim_{method}_{traj}.bag")
                    bag_name = f"xarm_sim_{method}_{traj}"
                elif self.robot_type == "panda":
                    if self.modified_bag_bool:
                        bag_path = os.path.join(self.bag_dir, f"panda_sim_{method}_{traj}_modified.bag")
                        bag_name = f"panda_sim_{method}_{traj}_modified"
                    else:
                        bag_path = os.path.join(self.bag_dir, f"panda_sim_{method}_{traj}.bag")
                        bag_name = f"panda_sim_{method}_{traj}"


                elif self.robot_type == "kinova":
                    bag_path = os.path.join(self.bag_dir, f"gen3_sim_{method}_{traj}.bag")
                    bag_name = f"gen3_sim_{method}_{traj}"

                elif self.robot_type == "puma":
                    bag_path = os.path.join(self.bag_dir, f"puma_sim_{method}_{traj}.bag")
                    bag_name = f"puma_sim_{method}_{traj}"

                elif self.robot_type == "xarm_real":
                    if self.modified_bag_bool:
                        bag_path = os.path.join(self.bag_dir, f"xarm_real_{method}_{traj}.bag")
                        bag_name = f"xarm_real_{method}_{traj}"
                    else:
                        bag_path = os.path.join(self.bag_dir, f"xarm_real_{method}_{traj}.bag")
                        bag_name = f"xarm_real_{method}_{traj}"
                elif self.robot_type == "xarm_real_spacemouse":
                    bag_path = os.path.join(self.bag_dir, f"xarm_spacemouse_{method}_{traj}.bag")
                    bag_name = f"xarm_spacemouse_{method}_{traj}"
                else:
                    rospy.logerr("Invalid robot type specified. Please use 'xarm' or 'panda'.")
                    return
                if not os.path.exists(bag_path):
                    rospy.logwarn(f"Bag file {bag_path} not found!")
                    continue
                
                # opening bag
                print(f"\n{'#'*60}")
                print(f"# OPENING BAG FILE: {bag_name}")
                print(f"# Path: {bag_path}")
                print(f"{'#'*60}\n")

                with rosbag.Bag(bag_path, 'r') as bag:
                    print("[1/6] Extracting /start_here topic...")
                    start_times, start_here_data = self.extract_topic_data(bag, "/start_here", Header)
                    print(start_times)
                    if len(start_times) == 0:
                        take = 0
                        print("⚠ No /start_here data found. Setting take=0")
                    else:
                        take = start_times[0]
                        print(f"✓ Start time: {take:.2f}s")

                    print(f"\n[2/6] Extracting /current_target_pose topic...")
                    target_times, target_data = self.extract_topic_data(bag, "/current_target_pose", PoseStamped)

                    # just take target_times of take or more
                    if len(target_times) > 0 and len(target_data) > 0:
                        old_target_times = target_times
                        mask = old_target_times >= take
                        target_times = target_times[mask]
                        target_data = target_data[mask]
                        print(f"  → Filtered to {len(target_times)} messages after start time")
                    else:
                        print(f"  → No target pose data found, skipping filtering")

                    if self.modified_bag_bool:
                        print(f"\n[3/6] Extracting /real_pose_error topic (modified bag)...")
                        pose_times, pose_error = self.extract_topic_data(bag, "/pose_error", PoseStamped)

                    else:
                        print(f"\n[3/6] Extracting /pose_error topic...")
                        pose_times, pose_error = self.extract_topic_data(bag, "/pose_error", PoseStamped)

                        if len(pose_times) > 0 and len(pose_error) > 0:
                            old_pose_times = pose_times
                            mask = old_pose_times >= take
                            pose_times = pose_times[mask]
                            pose_error = pose_error[mask]
                            print(f"  → Filtered to {len(pose_times)} messages after start time")
                        else:
                            print(f"  → No pose error data found, skipping filtering")

                    pose_error_norm = np.linalg.norm(pose_error, axis=1)
                    print(f"  → Computed pose error norm (shape: {pose_error_norm.shape})")

                    print(f"\n[4/6] Extracting /orientation_error topic...")
                    orient_times, orient_error = self.extract_topic_data(bag, "/orientation_error", Vector3)
                    if len(orient_times) > 0 and len(orient_error) > 0:
                        old_orient_times = orient_times
                        mask = old_orient_times >= take
                        orient_times = orient_times[mask]
                        orient_error = orient_error[mask]
                        print(f"  → Filtered to {len(orient_times)} messages after start time")
                    else:
                        print(f"  → No orientation error data found, skipping filtering")

                    orient_error_norm = np.linalg.norm(orient_error, axis=1)
                    print(f"  → Computed orientation error norm (shape: {orient_error_norm.shape})")

                    print(f"\n[5/6] Extracting /current_end_effector_pose topic...")
                    end_effector_times, end_effector_data = self.extract_topic_data(bag, "/current_end_effector_pose", PoseStamped)
                    if len(end_effector_times) > 0 and len(end_effector_data) > 0:
                        old_end_effector_times = end_effector_times
                        mask = old_end_effector_times >= take
                        end_effector_times = end_effector_times[mask]
                        end_effector_data = end_effector_data[mask]
                        print(f"  → Filtered to {len(end_effector_times)} messages after start time")
                    else:
                        print(f"  → No end effector data found, skipping filtering")

                    print(f"\n[6/6] Extracting /manip_measure topic...")
                    manip_measure_times, manip_data = self.extract_topic_data(bag, "/manip_measure", Float64)
                    if len(manip_measure_times) > 0 and len(manip_data) > 0:
                        old_manip_measure_times = manip_measure_times
                        mask = old_manip_measure_times >= take
                        manip_measure_times = manip_measure_times[mask]
                        manip_data = manip_data[mask]
                        print(f"  → Filtered to {len(manip_measure_times)} messages after start time")
                    else:
                        print(f"  → No manipulability measure data found, skipping filtering")

                    print(f"\n{'*'*60}")
                    print(f"✓ COMPLETED extraction for {bag_name}")
                    print(f"{'*'*60}\n")

                    # For kinova, only require error data since we skip position plots
                    if self.robot_type == "kinova":
                        if pose_times.size == 0 or pose_error.size == 0:
                            rospy.logwarn(f"Missing essential error data in {bag_path}, skipping method {method} for {traj}.")
                            continue
                    else:
                        if (target_times.size == 0 or end_effector_times.size == 0 or
                            target_data.size == 0 or end_effector_data.size == 0):
                            rospy.logwarn(f"Missing or empty data in {bag_path}, skipping method {method} for {traj}.")
                            continue

                    print(f"→ Storing extracted data for method '{method}'...\n")
                    extracted_data[method] = (
                        self.align_time(target_times,bag_name=bag_name), target_data,
                        self.align_time(pose_times, bag_name=bag_name), pose_error,
                        self.align_time(pose_times, bag_name=bag_name), pose_error_norm,
                        self.align_time(orient_times, bag_name=bag_name), orient_error,
                        self.align_time(orient_times, bag_name=bag_name), orient_error_norm,
                        self.align_time(end_effector_times, bag_name=bag_name), end_effector_data,
                        self.align_time(manip_measure_times, bag_name=bag_name), manip_data
                    )
            
            if not extracted_data:
                rospy.logwarn(f"No valid data available for trajectory {traj}")
                continue

            print(f"\n{'>'*60}")
            print(f"> GENERATING PLOTS FOR TRAJECTORY: {traj}")
            print(f"> Methods with data: {list(extracted_data.keys())}")
            print(f"{'>'*60}\n")

            # Plot the target pose for the end effector
            print("→ Creating Figure 1: Target Position plots...")
            fig, axs = plt.subplots(3, 1, figsize=(12, 14))
            for method in extracted_data:
                # print("extracted_data: ", extracted_data)
                target_times, target_pose, _, _, _, _, _, _, _, _, _, _, _, _ = extracted_data[method]
                new_target_times = [target_times[0][idx] for idx in range(len(target_times[0])) if target_times[0][idx]>0.0 and target_times[0][idx]<target_times[1]]
                new_target_pose_x = [target_pose[idx,0] for idx in range(len(target_times[0])) if target_times[0][idx]>0.0 and target_times[0][idx]<target_times[1]] 
                new_target_pose_y = [target_pose[idx,1] for idx in range(len(target_times[0])) if target_times[0][idx]>0.0 and target_times[0][idx]<target_times[1]] 
                new_target_pose_z = [target_pose[idx,2] for idx in range(len(target_times[0])) if target_times[0][idx]>0.0 and target_times[0][idx]<target_times[1]] 
                # Properly formatted LaTeX labels
                # latex_label = r"${}$".format(method.replace("_", r"\_"))
                latex_label = self.format_method_name(method)
                axs[0].plot(new_target_times, new_target_pose_x, label=latex_label)
                axs[1].plot(new_target_times, new_target_pose_y, label=latex_label)
                axs[2].plot(new_target_times, new_target_pose_z, label=latex_label)
            axs[0].set_ylabel('X Position')
            axs[1].set_ylabel('Y Position')
            axs[2].set_ylabel('Z Position')
            axs[2].set_xlabel('Time (s)')
            for ax in axs:
                if ax is not None:  # Skip None for kinova
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            if self.robot_type == "xarm":
                plt.suptitle(f'Velocity Control, Target Position for {self.traj_name_plot[traj]}')
            elif self.robot_type == "xarm_sim":
                plt.suptitle(f'xArm Simulation, Target Position for {self.traj_name_plot[traj]}')
            elif self.robot_type == "xarm_real_autonomous":
                plt.suptitle(f'Real Robot, Target Position for {self.traj_name_plot[traj]}')
            elif self.robot_type == "panda":
                plt.suptitle(f'Torque Control, Target Position for {self.traj_name_plot[traj]}')
            else:
                plt.suptitle(f'Target Position for {self.traj_name_plot[traj]}')
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.2, top=0.96, bottom=0.1)
            plt.show()

            # now save figure
            if self.save_figures_bool:
                figure_name = self.data_figure_folder + "/" + self.robot_type + "_all_methods_target_poses_" + traj + ".png"
                if not os.path.exists(self.data_figure_folder):
                    os.makedirs(self.data_figure_folder)
                    print(f"  → Created directory: {self.data_figure_folder}")
                fig.savefig(figure_name)  # Save as PNG
                fig.savefig(figure_name.replace(".png", ".pdf"), format="pdf")  # Save as PDF
                print(f"  ✓ Saved Figure 1: {figure_name}")
                print(f"  ✓ Saved Figure 1: {figure_name.replace('.png', '.pdf')}\n")

            print("→ Creating Figure 2: Error and Manipulability plots...")
            if self.robot_type == "kinova":
                # For kinova, only show error and manipulability plots (3 subplots)
                fig, axs = plt.subplots(3, 1, figsize=(12, 12),
                                        gridspec_kw={'height_ratios': [1.2, 1, 1]})
                # Insert None at beginning to maintain index mapping (axs[1]=pos_err, axs[2]=ori_err, axs[3]=manip)
                axs = [None] + list(axs)
            else:
                fig, axs = plt.subplots(4, 1, figsize=(12, 16),
                                        gridspec_kw={'height_ratios': [1, 1.2, 1, 1]})
            plotted_pos = False

            method_linestyles = [':', '--', '-', '--', '-', '--', ':']
            method_colors = ['LightBlue', 'Orange', 'DarkGreen', 'Brown', 'Pink']  # Colorblind-safe palette
            idx = 0
            max_duration = 0  # Track maximum duration across all methods
            for method in extracted_data:

                target_times, target_pose, _, pose_error, pose_times, pose_error_norm, _, _, orient_times, orient_error_norm, _, _, manip_measure_times, manip_data = extracted_data[method]

                # Get method-specific time range if available, otherwise use global times
                if self.robot_type == "xarm_sim":
                    bag_key = f"xarm_sim_{method}_{traj}"
                elif self.robot_type == "panda":
                    bag_key = f"panda_sim_{method}_{traj}"
                elif self.robot_type == "xarm_real_spacemouse":
                    bag_key = f"xarm_spacemouse_{method}_{traj}"
                elif self.robot_type == "puma":
                    bag_key = f"puma_sim_{method}_{traj}"
                else:
                    bag_key = None

                if bag_key and hasattr(self, 'time_adjustments') and bag_key in self.time_adjustments:
                    method_start_time = self.time_adjustments[bag_key]["start_time"]
                    method_end_time = self.time_adjustments[bag_key]["end_time"]
                else:
                    method_start_time = self.plot_start_time
                    method_end_time = self.plot_end_time

                # Track maximum duration for x-axis limits
                duration = method_end_time - method_start_time
                if duration > max_duration:
                    max_duration = duration

                # Shift times to start from 0 for display
                new_pose_times = [pose_times[0][idx] - method_start_time for idx in range(len(pose_times[0])) if pose_times[0][idx]>=method_start_time and pose_times[0][idx]<method_end_time]
                new_pose_error_x = [pose_error[idx,0] for idx in range(len(pose_times[0])) if pose_times[0][idx]>=method_start_time and pose_times[0][idx]<method_end_time]
                new_pose_error_y = [pose_error[idx,1] for idx in range(len(pose_times[0])) if pose_times[0][idx]>=method_start_time and pose_times[0][idx]<method_end_time]
                new_pose_error_z = [pose_error[idx,2] for idx in range(len(pose_times[0])) if pose_times[0][idx]>=method_start_time and pose_times[0][idx]<method_end_time]

                # make the pose error norm from the pose_error components
                new_pose_error_norm = np.linalg.norm(pose_error, axis=1)
                new_pose_error_norm = [new_pose_error_norm[idx] for idx in range(len(pose_times[0])) if pose_times[0][idx]>=method_start_time and pose_times[0][idx]<method_end_time]
                if method == "JacobianTranspose":
                    # print out max and min of the pose error norm
                    print(f"Max pose error norm for {method}: {max(new_pose_error_norm)}")
                    print(f"Min pose error norm for {method}: {min(new_pose_error_norm)}")

                # Process X and Y position data - shift times to start from 0
                new_target_times = [target_times[0][idx] - method_start_time for idx in range(len(target_times[0])) if target_times[0][idx]>=method_start_time and target_times[0][idx]<method_end_time]
                new_target_pose_x = [target_pose[idx,0] for idx in range(len(target_times[0])) if target_times[0][idx]>=method_start_time and target_times[0][idx]<method_end_time]
                new_target_pose_y = [target_pose[idx,1] for idx in range(len(target_times[0])) if target_times[0][idx]>=method_start_time and target_times[0][idx]<method_end_time]
                new_target_pose_z = [target_pose[idx,2] for idx in range(len(target_times[0])) if target_times[0][idx]>=method_start_time and target_times[0][idx]<method_end_time]

                new_orient_times = [orient_times[0][idx] - method_start_time for idx in range(len(orient_times[0])) if orient_times[0][idx] >= method_start_time and orient_times[0][idx] < method_end_time]
                new_orient_error_norm = [orient_error_norm[idx] for idx in range(len(orient_times[0])) if orient_times[0][idx] >= method_start_time and orient_times[0][idx] < method_end_time]

                new_manip_measure_times = [manip_measure_times[0][idx] - method_start_time for idx in range(len(manip_measure_times[0])) if manip_measure_times[0][idx]>=method_start_time and manip_measure_times[0][idx]<method_end_time]
                new_manip_measure = [manip_data[idx] for idx in range(len(manip_measure_times[0])) if manip_measure_times[0][idx]>=method_start_time and manip_measure_times[0][idx]<method_end_time] 
                
                latex_label = self.format_method_name(method)
                
                if self.plot_all_methods:
                    line_width = 2.5
                else:
                    line_width =  2.5
                
                # Check if we're dealing with vary methods
                is_vary_case = any(m.startswith("vary") for m in extracted_data.keys())

                # Plot position data only once - for vary case use first method, otherwise use JParse
                # Skip position plotting for kinova
                if (self.robot_type != "kinova" and not plotted_pos and
                    ((is_vary_case and idx == 0) or (not is_vary_case and "JParse" in method))):
                    plot_target_times = []
                    for t in new_target_times:
                        plot_target_times.append(t)
                    #     idx+=1

                    # filter out the first 5 seconds from the data
                    target_times = [t for t in plot_target_times if t > 0]

                    new_target_pose_x = [x for x, t in zip(new_target_pose_x, plot_target_times) if t > 0]
                    new_target_pose_y = [y for y, t in zip(new_target_pose_y, plot_target_times) if t > 0]
                    new_target_pose_z = [z for z, t in zip(new_target_pose_z, plot_target_times) if t > 0]
                    axs[0].plot(target_times, new_target_pose_x, label=f"Target Pose X", linewidth=line_width, color=CB_colors['Red'])
                    axs[0].plot(target_times, new_target_pose_y, label=f"Target Pose Y", linewidth=line_width, color=CB_colors['Blue'])
                    axs[0].plot(target_times, new_target_pose_z, label=f"Target Pose Z", linewidth=line_width, color=CB_colors['Green'])
                    plotted_pos = True

                # Special handling for different method types
                if method.startswith("vary"):
                    # For vary methods, use different colors but all solid lines
                    plot_color = CB_colors[method_colors[idx]]
                    plot_linestyle = '-'
                elif "JParse" in method:
                    # For other JParse methods, use purple solid line
                    plot_color = CB_colors['Purple']
                    plot_linestyle = '-'
                else:
                    # For non-JParse methods, use standard colors/styles
                    plot_color = CB_colors[method_colors[idx]]
                    plot_linestyle = method_linestyles[idx]

                axs[1].plot(new_pose_times, new_pose_error_norm, label=latex_label, linewidth=line_width, linestyle=plot_linestyle, color=plot_color)
                axs[2].plot(new_orient_times, new_orient_error_norm, label=latex_label, linewidth=line_width, linestyle=plot_linestyle, color=plot_color)
                axs[3].plot(new_manip_measure_times, new_manip_measure, label=latex_label, linewidth=line_width, linestyle=plot_linestyle, color=plot_color)

                idx += 1

            # Add horizontal line at y=0 for manipulability measure plot
            axs[3].axhline(y=0, color='black', linewidth=2.5, linestyle='-', alpha=0.8, zorder=5)

            for ax in axs:
                if ax is not None:  # Skip None for kinova
                    for spine in ax.spines.values():
                        spine.set_linewidth(6)  # thickness of border
                        spine.set_edgecolor('black')  # border color

                    # set x limit from 0 to the maximum duration across all methods
                    ax.set_xlim(0, max_duration)

            idx = 0
            for ax in axs:
                if ax is None:  # Skip None for kinova
                    idx += 1
                    continue
                # Add grid for better readability
                ax.grid(True, linestyle='--', alpha=0.7)

                # Calculate shading times starting from 0 (shifted for display)
                times = [
                    1 * self.interval,
                    2 * self.interval,
                    3 * self.interval,
                    4 * self.interval,
                    5 * self.interval,
                    6 * self.interval
                ]

                if self.highlight_regions:
                    ax.axvspan(0, times[0], color=shading_colors['light_gray'], alpha=0.75)
                    ax.axvspan(times[0], times[1], color=shading_colors['light_blue'], alpha=0.75)
                    ax.axvspan(times[1], times[2], color=shading_colors['light_yellow'], alpha=0.75)
                    ax.axvspan(times[2], times[3], color=shading_colors['light_blue'], alpha=0.75)
                    ax.axvspan(times[3], times[4], color=shading_colors['light_gray'], alpha=0.75)


                    # ax.axvspan(0, 40, color='#FFCCCC', alpha=0.75)    
                    # ax.axvspan(80, 137, color='#CCFFCC', alpha=0.75)
                
                # Make tick labels larger
                ax.tick_params(axis='y', which='major', labelsize=10)
                ax.tick_params(axis='x', labelsize=11, width=1.5)  # increase tick font and tick line width
                    
                # Improve legend
                if self.many_methods:
                    if idx == 2 or idx == 0:
                        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11, framealpha=0.9, edgecolor='gray')
                else:
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11, framealpha=0.9, edgecolor='gray')
                idx += 1

                for label in ax.get_xticklabels():
                    label.set_fontweight('bold')
                
                for label in ax.get_yticklabels():
                    label.set_fontweight('bold')

                # Add spines with slight thickness
                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)

            # Set y-axis labels with larger font
            if self.robot_type == "kinova":
                # For kinova, only set labels for error plots (skip position)
                axs[1].set_ylabel('Position Error (m)', fontsize=14, fontweight='bold')
                axs[2].set_ylabel('Ori. Error (rad)', fontsize=14, fontweight='bold')
                axs[3].set_ylabel('Manip. Measure', fontsize=14, fontweight='bold')
                axs[3].set_xlabel('Time (s)', fontsize=14, fontweight='bold')
            else:
                axs[0].set_ylabel('Position (m)', fontsize=14, fontweight='bold')
                axs[1].set_ylabel('Position Error (m)', fontsize=14, fontweight='bold')
                axs[2].set_ylabel('Ori. Error (rad)', fontsize=14, fontweight='bold')
                axs[3].set_ylabel('Manip. Measure', fontsize=14, fontweight='bold')
                axs[3].set_xlabel('Time (s)', fontsize=14, fontweight='bold')

            # Adjust y-limits for the plots (excluding manip measure plot)
            for i, ax in enumerate(axs):
                # Skip None axes for kinova or manip measure plot
                if ax is None or i == 3:
                    continue
                
                # For the first plot (position), we don't need to adjust y-limits

                if self.plot_all_methods:
                    continue
                    
                # Get current y-limits
                y_min, y_max = ax.get_ylim()
                
                # If data starts near zero, keep zero as the minimum
                if i != 0:  # Position error norm
                    if y_min > -0.05 * y_max:
                        y_min = 0
                else:
                    continue
                
                # Reduce the upper limit to focus on the fine differences
                # TODO: SHIVANI
                if i == 1:  # Position error norm
                    if self.max_position_error_y is not None:
                        y_max_new = self.max_position_error_y
                    else:
                        y_max_new = y_min + (y_max - y_min) * 1  # Use auto scaling
                elif i==2:  # Orientation error norm
                    y_max_new = self.max_orient_error_y  # Use settable parameter

                # Set new limits
                ax.set_ylim(y_min, y_max_new)

            plt.tight_layout()  # Increase padding around each subplot

            plt.subplots_adjust(hspace=0.3, top=0.96, bottom=0.1)
            plt.show()

            #now save figure
            if self.save_figures_bool:
                save_path = self.data_figure_folder + "/" + self.robot_type + "_all_methods_norm_error_" + traj + ".png"
                if not os.path.exists(self.data_figure_folder):
                    os.makedirs(self.data_figure_folder)
                fig.savefig(save_path)  # Save as PNG
                fig.savefig(save_path.replace(".png", ".pdf"), format="pdf")  # Save as PDF
                print(f"  ✓ Saved Figure 2: {save_path}")
                print(f"  ✓ Saved Figure 2: {save_path.replace('.png', '.pdf')}\n")

            print(f"\n{'✓'*60}")
            print(f"✓ COMPLETED PLOTTING FOR TRAJECTORY: {traj}")
            print(f"{'✓'*60}\n")

        print(f"\n{'█'*60}")
        print(f"█ ALL PROCESSING COMPLETE!")
        print(f"█ Total trajectories processed: {len(self.trajectories)}")
        print(f"{'█'*60}\n")
        rospy.loginfo("Processing complete.")

if __name__ == "__main__":
    try:
        BagFileProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass