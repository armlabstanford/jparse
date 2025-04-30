cmake_minimum_required(VERSION 3.0.2)
project(manipulator_control)

find_package(catkin_simple REQUIRED)

catkin_python_setup()

catkin_simple()


## Add support for C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(Eigen3 REQUIRED)

include_directories(include ${EIGEN3_INCLUDE_DIR} ${catkin_INCLUDE_DIRS} ${PCL_LIBRARIES})


cs_add_executable(jparse src/jparse.cpp)


cs_install()

cs_export()
#https://github.com/catkin/catkin_simple 