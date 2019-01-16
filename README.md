# DetectNet_ROS
NVIDIA provides [deep-learning inference](https://github.com/dusty-nv/jetson-inference) networks and deep vision primitives with TensorRT and Jetson TX2. 'DetectNet' performs detecting objects, and finding where in the video those objects are located (i.e. extracting their bounding boxes). 

ROS topic can be used as image input via ROS image transport for DetectNet using DetectNet_ROS.

Tested and working in July 2018


In order to run it requires to compile and install jetson-inference on TX2 module

    cd jetson-inference
    mkdir build
    cd build
    cmake ../
    make
    sudo make install
    



  
# Pre-requisite
- Jetson TX2 with JetPack >3.2 (just use latest ) 
- TensorRT >3
- CUDA 9.0
- cuDNN >6.1
- OpenCV >3
- jetson-inference

# ROS deps

- ros-kinetic-ros-base  (see: https://github.com/jetsonhacks/installROSTX2 )
- ros-kinetic-cv-bridge
- ros-kinetic-image-transport
