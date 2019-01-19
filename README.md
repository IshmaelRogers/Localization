# Localization

[image1]: ./images/filter_example


The position tracking problem is easier to solve than the global localization one.

The kidnapped robot problem, the robot is teleported to a different location. 

In global localization, the robot's initial pose is unknown. 



The challenge of determining the robot's *pose in a mapped environment implements a probablisitics algorithm to filter out noisey sensor measurements. 

Pose - is the robot's (x,y) coordinates + the orientation (theta) relative to a known start point.  

Starting with a known map of the environment, and a sensor susceptible to sensor noise like lidar or ultrasonics, the robot will start off with guesses (probablities) about where it is relative to the start point. Over time the robots should narrow down on a precise location. 


Localization algorithms 

* Extended Kalman Filter - The most commonly used estimates the state of non linear models

* Markov - a base filter localization, that maintains a probablitlity distribution over the set of possible poses (i.e position and orientation values of the robot.)

* Gird - histogram hilter estimates the pose uses grid https://www.udacity.com/course/artificial-intelligence-for-robotics--cs373 j

* Monte Carlo - particle filter

Localization Challenges
---

* Position tracking - initial pose of the robot is known, the challenge is to estimate the robot's pose as it moves around an envirornment.  
  - Uncertainity in the movements of the robot make this a challenging problem 

* Global - the robot's initial pose is unknown and it must determine the pose relative to the ground map . 
  - Uncertainity is higher making this problem extremely difficult.
  
* Kidnapped Robot - Similiar to Global but the robot can be kidnapped at any time and moved to a new location on the map. 
  - Helps prepare for the the worst possible case. The algorithm itself is not free from error. There will be instances where the robot will miscalulate where it is. The Kidnapped Robot problem teaches the robot to recover for such instances and correctly locate itself on the map. 
  
  
 # Kalman Filters
 
 - an estimation algorithm that is used widely in controls. estimables the value of a variable in real time as the variable is being collected. 
 
 It can take data with a lot of uncertainty or noise in the measurements and provide an accurate estimate of the real value; very quickly. 
 
 Example: Underwater Robotics
 ---
 
 Monitoring the pressure as the systems swims through the water.
 
 Problems: 
 
 * The pressure measurements are not perfectly accurate 
 * electical noise from the sensor 
 
 Solution: 
 
 When the pressure sensor starts collecting data, the Kalman filter begins to narrow in and estimates the actual pressure. In addition to the sensor readings, the Kalman filter accounts for the uncertainity of the sensor readings.
 
 
 What happens every time a measurement is recorded? 
 
 Kalman filter is an iteration of the following to steps: 
 
 * Measurement update 
 
 * State prediction 
 
 ![alt text][image1]
 
 
  # Background  
  
  - Used in the Apollo program 
  
  # Applications
  - used to estimate the state of the system when the measurements are noisey
    - Position tracking for a mobile robot
    - Feature tracking 
    
  # Variations 
  
  KF - applied to linear systems
  
  EKF - applied to nonlinear system 
  
  UKF - highly nonlinear - http://ais.informatik.uni-freiburg.de/teaching/ws12/mapping/pdf/slam05-ukf.pdf
  
  Robot Uncertainty 
 
  1D Gaussian 
   
  
  Measurement update 
  
  State Prediction 
  
  1-D Kalman Filter
  
  Multivariate Gaussian 
  
  Multidimensional KF
  
  Design of Multidimensional KF
  
  Extended Kalman Filter 
  
  EKF Example 
  
  Limitations 
  
  Extended Kalman Filter 
  
  Lab: Kalman Filter 
  
  Montel Carlo Localization 
  
  Build MCL in C++ 
  
  Project: Where am I? 
  
 







