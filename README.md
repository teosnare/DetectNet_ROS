# DetectNet_ROS
NVIDIA provides [deep-learning inference](https://github.com/dusty-nv/jetson-inference) networks and deep vision primitives with TensorRT and Jetson TX2. 'DetectNet' performs detecting objects, and finding where in the video those objects are located (i.e. extracting their bounding boxes). 
  
ROS topic can be used as image input via ros image transport for DetectNet using DetectNet_ROS.
  
# Pre-requisite
- Jetson TX2 with JetPack >3.2
- TensorRT >3
- CUDA 9.0
- cuDNN 6.1
- OpenCV >3
- jetson-inference
