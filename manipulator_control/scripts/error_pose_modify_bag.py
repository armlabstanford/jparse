#!/usr/bin/env python3

import rospy
import rosbag
import os
import numpy as np
import rospkg
import sys
from geometry_msgs.msg import PoseStamped

# Substrings to filter bag files
# TARGET_SUBSTRINGS = ["line_extended_keypoints.bag", "ellipse_keypoints.bag", "ellipse.bag"]
TARGET_SUBSTRINGS = ["line_extended_keypoints.bag", "ellipse_keypoint.bag"] #for xarm real autonomous

def get_bagfile_paths():
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('manipulator_control')
    # bag_dir = os.path.join(package_path, 'bag', 'new_panda_sim_bags', 'kpx_15_kpth_15')
    # bag_dir = os.path.join(package_path, 'bag', 'new_panda_sim_bags', 'kpx_15_kpth_15_variable_retraction_gain')
    # bag_dir = os.path.join(package_path, 'bag', 'new_xarm_sim_bags', 'line_extended_keypoints_freq_0_07_ellipse_keypoints_freq_0_1')

    bag_dir = os.path.join(package_path, 'bag', 'xarm_real_bags', 'autonomous')


    if not os.path.exists(bag_dir):
        raise FileNotFoundError(f"Bag directory '{bag_dir}' not found.")

    bagfiles = []
    for filename in os.listdir(bag_dir):
        if any(substr in filename for substr in TARGET_SUBSTRINGS):
            bagfiles.append(os.path.join(bag_dir, filename))
    
    if not bagfiles:
        raise FileNotFoundError("No matching bag files found in directory.")
    
    return bagfiles

def pose_subtract(target, current):
    error = PoseStamped()
    error.header = target.header
    error.pose.position.x = target.pose.position.x - current.pose.position.x
    error.pose.position.y = target.pose.position.y - current.pose.position.y
    error.pose.position.z = target.pose.position.z - current.pose.position.z
    error.pose.orientation.x = target.pose.orientation.x - current.pose.orientation.x
    error.pose.orientation.y = target.pose.orientation.y - current.pose.orientation.y
    error.pose.orientation.z = target.pose.orientation.z - current.pose.orientation.z
    error.pose.orientation.w = target.pose.orientation.w - current.pose.orientation.w
    return error

def process_bag(input_bag_path, output_bag_path):
    rospy.loginfo(f"Reading from bag: {input_bag_path}")
    rospy.loginfo(f"Writing to bag: {output_bag_path}")

    end_effector_poses = []
    target_poses = []

    # Read the input bag file
    with rosbag.Bag(input_bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages():
            if topic == "/current_end_effector_pose":
                end_effector_poses.append(msg)
            elif topic == "/current_target_pose":
                target_poses.append(msg)

    rospy.loginfo(f"Collected {len(end_effector_poses)} end-effector poses")
    rospy.loginfo(f"Collected {len(target_poses)} target poses")

    # Open the output bag file
    with rosbag.Bag(output_bag_path, 'w') as out_bag:
        with rosbag.Bag(input_bag_path, 'r') as in_bag:
            for topic, msg, t in in_bag.read_messages():
                # Write all original messages to the output bag
                out_bag.write(topic, msg, t)

                # If it's a target pose, compute the error
                if topic == "/current_target_pose":
                    target_time = msg.header.stamp.to_sec()

                    # Find closest end-effector pose within 0.2 seconds
                    closest_pose = None
                    closest_time_diff = float('inf')

                    for ee_pose in end_effector_poses:
                        ee_time = ee_pose.header.stamp.to_sec()
                        time_diff = abs(ee_time - target_time)

                        if time_diff < 0.2 and time_diff < closest_time_diff:
                            closest_time_diff = time_diff
                            closest_pose = ee_pose

                    if closest_pose:
                        error_msg = pose_subtract(msg, closest_pose)
                        out_bag.write("/real_pose_error", error_msg, rospy.Time.from_sec(target_time))
                        rospy.loginfo(f"Wrote error at {target_time} with closest diff {closest_time_diff:.4f}")

    rospy.loginfo(f"Processing complete for {input_bag_path}!")

def process_all_bags():
    bagfiles = get_bagfile_paths()
    rospy.loginfo(f"Found {len(bagfiles)} matching bag files")

    for input_bag_path in bagfiles:
        output_bag_path = input_bag_path.replace('.bag', '_modified.bag')
        rospy.loginfo(f"Processing bagfile: {input_bag_path}")
        rospy.loginfo(f"Saving modified bagfile to: {output_bag_path}")

        try:
            process_bag(input_bag_path, output_bag_path)
        except Exception as e:
            rospy.logerr(f"Failed to process {input_bag_path}: {e}")

if __name__ == '__main__':
    rospy.init_node('pose_error_computation')

    try:
        process_all_bags()
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        sys.exit(1)