#!/usr/bin/env python3

import rosbag
from geometry_msgs.msg import TwistStamped, PoseStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
import rospy
import rospkg
import os
import sys

def get_bagfile_path(bagname):
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('manipulator_control')
    bagfile_path = os.path.join(package_path, 'bag', bagname)
    
    if not os.path.exists(bagfile_path):
        raise FileNotFoundError(f"Bagfile '{bagfile_path}' not found.")
    
    return bagfile_path

def create_marker(end_effector_pose, twist):
    marker = Marker()
    marker.header = end_effector_pose.header
    marker.header.frame_id = end_effector_pose.header.frame_id
    marker.ns = "robot_action_vector"
    marker.id = 0
    marker.type = Marker.ARROW
    marker.action = Marker.ADD

    # Start point (position of end-effector)
    start = Point()
    start.x = end_effector_pose.pose.position.x
    start.y = end_effector_pose.pose.position.y
    start.z = end_effector_pose.pose.position.z

    # End point based on twist.linear components
    scale_factor = 5
    end = Point()
    end.x = start.x + twist.twist.linear.x * scale_factor
    end.y = start.y + twist.twist.linear.y * scale_factor
    end.z = start.z + twist.twist.linear.z * scale_factor
    
    marker.points = [start, end]

    # Arrow size and color
    marker.scale.x = 0.015  # Shaft diameter
    marker.scale.y = 0.04  # Head diameter
    marker.scale.z = 0.04  # Head length

    marker.color.a = 1.0
    marker.color.r = 0.5
    marker.color.g = 0.0
    marker.color.b = 0.8

    return marker

def process_bag(input_bagfile, output_bagfile):
    with rosbag.Bag(output_bagfile, 'w') as outbag:
        end_effector_pose = None
        
        with rosbag.Bag(input_bagfile, 'r') as inbag:
            for topic, msg, t in inbag.read_messages():
                # Save all original messages
                if topic != '/robot_action_vector':
                    outbag.write(topic, msg, t)

                # Save modified TwistStamped message
                if topic == '/robot_action':# and isinstance(msg, TwistStamped):
                    msg.header.frame_id = "base_link"
                    outbag.write('/robot_action', msg, t)
                    rospy.loginfo(f"Modified message: {msg}")

                    if end_effector_pose is not None:
                        marker = create_marker(end_effector_pose, msg)
                        outbag.write('/robot_action_vector', marker, t)

                # Store latest end-effector pose
                if topic == '/current_end_effector_pose':# and isinstance(msg, PoseStamped):
                    end_effector_pose = msg

if __name__ == '__main__':
    rospy.init_node('modify_bagfile')

    if len(sys.argv) < 2:
        print("Usage: modify_bagfile.py <bagname>")
        sys.exit(1)
        
    import pdb; pdb.set_trace()

    bagname = sys.argv[1]
    try:
        input_bagfile = get_bagfile_path(bagname)
        output_bagfile = input_bagfile.replace('.bag', '_modified.bag')

        rospy.loginfo(f"Processing bagfile: {input_bagfile}")
        rospy.loginfo(f"Saving modified bagfile to: {output_bagfile}")

        process_bag(input_bagfile, output_bagfile)

        rospy.loginfo("Bagfile modification complete.")

    except Exception as e:
        rospy.logerr(f"Error: {e}")
        sys.exit(1)