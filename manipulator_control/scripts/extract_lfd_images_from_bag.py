#!/usr/bin/env python3

import rospy
import rosbag
import os
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospkg
import sys

def get_bagfile_path(bagname):
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('manipulator_control')
    bagfile_path = os.path.join(package_path, 'bag', 'xarm_real_bags', 'LfD', bagname)
    
    if not os.path.exists(bagfile_path):
        raise FileNotFoundError(f"Bag file '{bagfile_path}' not found.")
    
    return bagfile_path

def save_image(image_msg, output_dir, index):
    bridge = CvBridge()
    try:
        # Convert ROS Image message to OpenCV BGR format
        cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        filename = os.path.join(output_dir, f"image_{index:05d}.png")
        cv2.imwrite(filename, cv_image)
        rospy.loginfo(f"Saved image to {filename}")
        return cv_image
    except Exception as e:
        rospy.logerr(f"Failed to convert and save image: {e}")
        return None

def process_bag(input_bag_path):
    if not os.path.exists(input_bag_path):
        rospy.logerr(f"Bag file not found: {input_bag_path}")
        return
    
    # Create an output folder at the same level as the bag file
    output_dir = os.path.join(os.path.dirname(input_bag_path), 'images')
    os.makedirs(output_dir, exist_ok=True)

    rospy.loginfo(f"Saving images to: {output_dir}")

    index = 0
    frames = []
    timestamps = []

    try:
        with rosbag.Bag(input_bag_path, 'r') as bag:
            for topic, msg, t in bag.read_messages():
                if topic == "/camera/rgb/image_raw":# and isinstance(msg, Image):
                    cv_image = save_image(msg, output_dir, index)
                    if cv_image is not None:
                        frames.append(cv_image)
                        timestamps.append(msg.header.stamp.to_sec())
                        index += 1

        rospy.loginfo(f"Extracted and saved {index} images")

        # If there are frames, create a video
        if len(frames) > 1:
            # Calculate average time difference to define FPS
            time_diffs = np.diff(timestamps)
            avg_time_diff = np.mean(time_diffs)
            fps = round(1.0 / avg_time_diff) if avg_time_diff > 0 else 30

            rospy.loginfo(f"Average time difference: {avg_time_diff:.4f} seconds => FPS: {fps}")

            # Get video dimensions from the first frame
            height, width, _ = frames[0].shape
            video_path = os.path.join(output_dir, 'output_video.avi')

            # Use the MJPG codec and .avi container, which tends to avoid color issues
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            # Write frames in BGR format directly to the AVI file
            for frame in frames:
                video_writer.write(frame)

            video_writer.release()
            rospy.loginfo(f"Saved MJPEG video to {video_path}")

    except Exception as e:
        rospy.logerr(f"Error reading bag file: {e}")

if __name__ == '__main__':
    rospy.init_node('image_extraction_node')

    if len(sys.argv) < 2:
        rospy.logerr("Usage: rosrun <your_package> image_extraction.py <bagname>")
        sys.exit(1)

    bagname = sys.argv[1]

    try:
        input_bag_path = get_bagfile_path(bagname)
        rospy.loginfo(f"Reading bag file: {input_bag_path}")
        process_bag(input_bag_path)
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        sys.exit(1)