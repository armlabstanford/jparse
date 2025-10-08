# Official Dockerfile for J-PARSE 

FROM tiryoh/ros-desktop-vnc:noetic

RUN set -x \
	&& apt-get update \
	&& apt-get install -y build-essential \
	&& apt-get install -y ros-noetic-moveit \
	&& apt-get install -y ros-noetic-moveit-servo \
	&& apt-get install -y ros-noetic-moveit-visual-tools \
	&& apt-get install -y ros-noetic-moveit-ros-visualization \
	&& apt-get install -y ros-noetic-graph-msgs \
	&& apt-get install -y ros-noetic-rosparam-shortcuts \
	&& apt-get install -y ros-noetic-control-toolbox \
	&& apt-get install -y gstreamer1.0-tools gstreamer1.0-libav libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev gstreamer1.0-plugins-good gstreamer1.0-plugins-base \
	&& rm -rf /var/lib/apt/lists/* 


RUN set -x \
    && pip3 install conan==1.59 \
    && conan config set general.revisions_enabled=1 \
    && conan profile new default --detect > /dev/null \
    && conan profile update settings.compiler.libcxx=libstdc++11 default


RUN rosdep update \
        && echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

RUN apt-get update && apt-get install -y ros-noetic-ros-control ros-noetic-ros-controllers ros-noetic-rgbd-launch ros-noetic-joy


RUN /bin/bash -c 'conan config set general.revisions_enabled=1'
RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; mkdir -p /ros_kortex_ws/src; cd ros_kortex_ws/src; git clone https://github.com/Kinovarobotics/ros_kortex; git clone https://github.com/Kinovarobotics/ros_kortex_vision; git clone https://github.com/ros-simulation/gazebo_ros_pkgs; cd ..; catkin_make'
RUN echo "source /ros_kortex_ws/devel/setup.bash" >> ~/.bashrc


# Install ros real sense
RUN apt-get update && apt-get install -y ros-noetic-realsense2-camera && apt-get install -y ros-noetic-cv-bridge && apt install -y ros-noetic-sensor-msgs

RUN apt-get install -y ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control

# install xarm code
RUN cd / && mkdir -p dev_ws/src && cd dev_ws/src

RUN source /opt/ros/noetic/setup.bash && cd /ros_kortex_ws/src && git clone https://github.com/xArm-Developer/xarm_ros.git --recursive \
	&& git clone -b noetic-devel https://github.com/armlabstanford/hrl-kdl.git \
	&& git clone https://github.com/catkin/catkin_simple.git \
    && cd xarm_ros && git pull && git submodule sync && git submodule update --init --remote \
    && cd /ros_kortex_ws/src && rosdep update && rosdep install --from-paths . --ignore-src --rosdistro $ROS_DISTRO -y

RUN source /opt/ros/noetic/setup.bash && cd /ros_kortex_ws && rm -rf build/ devel/ && catkin build && source devel/setup.bash
# install franka panda gazebo
# deps
RUN apt install -y ros-$ROS_DISTRO-gazebo-ros-control ros-${ROS_DISTRO}-rospy-message-converter ros-${ROS_DISTRO}-effort-controllers ros-${ROS_DISTRO}-joint-state-controller ros-${ROS_DISTRO}-moveit ros-${ROS_DISTRO}-moveit-commander ros-${ROS_DISTRO}-moveit-visual-tools
RUN apt install -y ros-${ROS_DISTRO}-libfranka

RUN cd / && mkdir -p panda_ws/src && cd panda_ws/src
RUN apt-get update && apt-get install -y ros-${ROS_DISTRO}-boost-sml
RUN source /opt/ros/noetic/setup.bash && cd /panda_ws/src && git clone -b noetic-devel https://github.com/justagist/panda_simulator

RUN cd /panda_ws/src/panda_simulator && ./build_ws.sh
RUN pip install "numpy<1.24" && pip install numpy-quaternion==2020.5.11.13.33.35 && pip install numba

RUN git clone https://github.com/xArm-Developer/xArm-Python-SDK.git && cd xArm-Python-SDK && python3 setup.py install

RUN pip install hidapi==0.14.0.post4
RUN apt install -y nano
RUN pip install mpld3 packaging
RUN apt install -y texlive texlive-latex-extra texlive-fonts-recommended dvipng

RUN apt install ros-noetic-apriltag-ros -y

COPY ./entrypoint.sh /
ENTRYPOINT [ "/bin/bash", "-c", "/entrypoint.sh" ]

RUN ln -sf /usr/bin/gcc /usr/bin/cc

ENV USER ubuntu
ENV PASSWD ubuntu
ENV CONAN_REVISIONS_ENABLED 1