
## Resolution & Mouse DPI
xrandr --output DVI-I-1 --mode 1920x1080 --rate 60
xinput --set-prop "pointer:RAPOO RAPOO 5G Wireless Device" "Device Accel Constant Deceleration" 0.6

## Terminal1 - Local ROS
roscore


## Terminal2
roslaunch kinect2_bridge kinect2_bridge.launch


## Terminal3
rostopic list


## image viewer
rqt


## point cloud
# topic
/kinect2/hd/points
/kinect2/sd/points
# message type
sensor_msgs/PointCloud2
# viewer
rosrun kinect2_viewer kinect2_viewer kinect2 hd cloud
rosrun kinect2_viewer kinect2_viewer kinect2 sd cloud


## camera
# band width monitor
rostopic bw /kinect2/qhd/image_color
# hz monitor
rostopic hz /kinect2/qhd/image_color
# camera info
rostopic echo /kinect2/qhd/camera_info


## CLEAN old cache & RESET apt-get update
~/RE_apt-get.sh

## Check installed packages
dpkg --get-selections | grep $$$


## NuPic Installation
# UPDATE pip
curl https://bootstrap.pypa.io/get-pip.py | sudo python
# Install Nupic.bindings
# sudo pip install nupic.bindings # 0.4.4
sudo pip install https://s3-us-west-2.amazonaws.com/artifacts.numenta.org/numenta/nupic.core/releases/nupic.bindings/nupic.bindings-0.4.0-cp27-none-linux_x86_64.whl # 0.4.0
# Install Nupic
sudo pip install nupic # 0.4.13
# Validation
python
>>> import nupic, sys
>>> print 'nupic' in sys.modules
True # Success!


## Check OpenCV
# Show version
pkg-config --modversion opencv
# Main folder in usr/include/opencv
pkg-config --cflags opencv
# Show in usr/lib
pkg-config --libs opencv
# Install Nonfree
sudo add-apt-repository --yes ppa:xqms/opencv-nonfree
sudo apt-get update
sudo apt-get install libopencv-nonfree-dev libopencv-nonfree2.4 python-opencv

## SSH Remote
# Setup
sudo service ssh start
sudo /etc/init.d/ssh restart
sudo /etc/init.d/ssh start # optional
sudo /etc/init.d/ssh stop # optional
sudo gedit /etc/ssh/sshd_config # modify {Port} (default: 22), then RESTART
# Check
sudo ps -e | grep ssh
# Login
ssh -l {UserName} {ServerIp} # input {PassWord}
ssh {UserName}@{ServerIp} # input {PassWord}
ssh robot@172.18.29.191 # robot@bii
# File/Folder Transfer
scp -r {LocalFile} {UserName}@{ServerIp}:{RemoteFile} # from local to remote
scp -r {UserName}@{ServerIp}:{RemoteFile} {LocalFile} # from remote to local
# Exit
exit