Scaricato repo https://github.com/cesare-montresor/DetectNet_ROS

cd $HOME
mkdir archon
copia install.sh
chmod +x install.sh

modificato install.sh per installare jetson-inference in cartella locale:

mkdir libs
cd libs

cmake -DCMAKE_INSTALL_PREFIX=../distrib/ ../

./install.sh

Ho dovuto rieseguire "a mano" installROS.sh (non ho verificato il perchè, forse semplicemente bastava un chmod)
cd installROSTX2
./installROS.sh

Ho evitato setupCatkinWorkspace.sh

in /ros_ws/src/Detectnet_ROS

Modificato CMakeList.txt come nella mia versione di repo per caricare lib e header da cartella lib/.../distrib


in ros_ws/src
git clone https://github.com/ros-drivers/video_stream_opencv.git

catkin_make


