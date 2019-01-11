#! /bin/bash


# install in the current directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROS_WS_DIR=$DIR/ros_ws



# Prerequisits
sudo apt-get install git cmake



# ROS
cd $DIR
git clone https://github.com/jetsonhacks/installROSTX2.git
cd $DIR/installROSTX2
./installROS.sh ros-kinetic-ros-base
sudo apt-get -y install ros-kinetic-cv-bridge ros-kinetic-image-transport
./setupCatkinWorkspace.sh ../../$ROS_WS_DIR   # by default it install in the home dir ../../ bring it back to /


# Patch for Jetson inference (make error)
# https://devtalk.nvidia.com/default/topic/1007290/jetson-tx2/building-opencv-with-opengl-support-/post/5141945/#5141945
cd /usr/lib/aarch64-linux-gnu/
sudo ln -sf tegra/libGL.so libGL.so


# Jetson-inference
cd $DIR
mkdir libs
cd libs
git clone https://github.com/dusty-nv/jetson-inference
cd jetson-inference
git submodule update --init
mkdir distrib
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../distrib/ ../
make
sudo make install


# ROS DetectNet

cd $ROS_WS_DIR/src
git clone https://github.com/cesare-montresor/DetectNet_ROS.git
cd $ROS_WS_DIR



