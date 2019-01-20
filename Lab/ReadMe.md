[image1]: ./images/IMU.png
[image3]: ./images/rotaryencoder.png
[image4]: ./images/sensorfusion.png
[image5]: ./images/
[image6]: ./images/
[image7]: ./images/
[image8]: ./images/
[image9]: ./images/
[image10]: ./images/




# Introduction 

Implement an EKF package 

Control a simulated robot by collecting sensory information from the following onboard sensors: 

* IMU
* Rotary Encoder 
* Camera
* GPS

Using this collected data, we will apply an EKF package to implement sensor fusion to estimate the robot's pose.

In this lab we will make use of 5 ROS packages:

1. Turtlebot_gazebo - launches a mobile robot inside gazebo environment 
2. robot_pose_efk - estimates the position and orientation of the robot
3. odom_to_trajectory - append the odometry value generated over time into a trajectory path 
4. turtle_bot_teleop - allows operation of robot with keyboard commands
5. rviz - allows you to visualize the estimate position and orientation of the robot. 


# Sensor Fusion
Inside an enovironment a robot will localize itself by collecting data from its different on-board sensors. Each sensor has noise and error. 

IMU - measures linear accleration and angular velocity 

a double integral of accleration is calculated to estimate the robot's position. The IMU is already noisey, therefore a double integration will accumulate even more error over time. 

NOTE: Check drift for error paramaters

x = doubleintegral a dt

![alt text][image1]


Rotary encoder - are attached to the robots acuated wheel. It measures the velocity and position of the wheels.
![alt text][image2]

To estimate the robot's position the integral of the velocity is calculated

NOTE: Check resolution so that the robot can account for environmental factors that may cause noise

Vision x = f(camera_depth)
Is an RGB-D camera that captures images and calculate the depth towards the obstacle which can be translated into a position 

The light affects the perfrormance. The camera is usually unable to measure the depths at small distances. 

NOTE: Check smallest range of operation.

Cannot accurate estimate the robot's pose with only 1 of these sensors. They all are very noisey and untrustworthy

Sensor fusion of at least two of them is usually required
![alt text][image3]

The Extended Kalman Filter will take all the noisy measurements, compare them, filter the noise and provide a good estimate of the robot's pose.


# Catkin Workspace

Create a catkin_ws to hold various ROS packages

```` shell
$ mkdir -p /home/workspace/catkin_ws/src
$ cd /home/workspace/catkin_ws/src
$ catkin_init_workspace
$ cd ..
$ catkin_make

````
Perform system update/upgrade

```shell

$ apt-get update
$ apt-get upgrade - y

```



# Turtlebot Gazebo Package 

Used by roboticists to perform localization, mapping and path planning 

In this lab we will be using turtlebot 2 in a gazebo environment and estimate its pose. 

Go through documentation and identify nodes and topics 



Robot Pose EKF Package


Odometry to Trajectory Package 

TurtleBot Teleop Package 

Rviz Package

Main Launch

Rqt Multiplot 
