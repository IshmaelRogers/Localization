# Localization

[image1]: ./images/filter_example.png
[image2]: .images/bellCurve.png
[image3]: ./images/probvdist.png



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
  
  Ideal vs Real world
  
  * both know starting position 
  
  
  In the real world the robot may encounter complexities that result in the robot's movement being imprecise.
  
  Terrain
  wheel sleep
  enviornment factors
  
  The robot will not reach the desired goal.
  
  Example:
  
  If we record the robot moving 10 meter forward a total of 100 times we would get a plot like this:
  ![alt text][image2] 
  
   This graph displays a probabilty distribution of the robot's fianl position after multiple iterations of the movement.
   
   The X-axis is the distance traveled by the robot
   The Y-acxis is how often the robot stopped at that distance
   
   The shape of the gaussian
   
   few enivornment factors movements are precise distribution is narrow
   
   Rescue missions have many environmental factors
   
   Sensors contain a bunch of sensor noise. 
   
   Advantage
   ---
   
   Movement and Sensory measurements are uncertain, the kalman filter take in account the uncertainity of each sensor's measurement to help the robot better sense its own state. This estimatation happens only after a few sensor measurements. 
   
   To do this:
   
   Use an intitial guess and take in account of expected uncertainity 
   
   Sensor Fusion - uses the kalman filter to calculate an accurate estimate using data from multiple sensor. 
   
 
  # 1D Gaussian
   At the basis of the Kalman Filter is the Gaussian distribution also known as a bell curve. Imagine Hexpod Zero was commanded to execute 1 motion, the rover's location can be representedd as a Gaussian. 
   
   The exact location is not certain but the level of uncertainity was bound. 
   
   The role of a Kalman Filter
   ---
   
   after a movement or a measurement update, it output a unimodal Gaussian distribution. 
   NOTE: This is its best guess at the true value of a parameter
   
   
   A Gaussian distribution is a probability distribution which is a continous function. 
   
   
   Claim:
   
   The probability that a random variable, x, will take a value between x1 and x2 is given by the integral of the function from x1 to x2. 
   
   (p x1 < x < x2) = x2 integral x1 fx(x)dx
   
   In the image below, the probability of the rover being located between 8.7m and 9m is 7%
   
   ![alt text][image3]
   
   
   Mean and variance
   ---
   
   A gaussian is charactterized by two parameters 
   
   mean (mue) - the most probable occurrence. 
   variance (sigma^2) - the width of the curve, 
   
   *Unimodal - implies a single peak present in the distribution.
   
   Gaussian distributions are frequently abbreviated as:
   
   (Nx: mue, sigma^2)
   
   Formula
   ---
   
   <a href="https://www.codecogs.com/eqnedit.php?latex=p(x)&space;=&space;\frac{e^{\frac{-(x-\mu&space;)^{2}}{2\sigma&space;^{2}}}}{\sigma&space;\sqrt{2\pi&space;}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(x)&space;=&space;\frac{e^{\frac{-(x-\mu&space;)^{2}}{2\sigma&space;^{2}}}}{\sigma&space;\sqrt{2\pi&space;}}" title="p(x) = \frac{e^{\frac{-(x-\mu )^{2}}{2\sigma ^{2}}}}{\sigma \sqrt{2\pi }}" /></a>
   
   NOTE: Exponential of a quadratic function. The quadratice compares the value of x to (mue). In the case that x=mue the exponential is equal to 1 (e^0 = 1). 
   
   NOTE:The constant in front of the exponential is a necessary normalizing factor. 
   
   In discrete proababiity, the probabilities of all the options must sum to one. 
   
   The area underneath the function always suns to one 
   
   integral p(x)dx = 1
   
   Coding the 1D Gaussian in C++
---

```cpp

#include <iostream>
#include <math.h>

using namespace std;

double f(double mu, double sigma2, double x)
{

  prob = 1.0 / sqrt(2.0 * M_PI * sigma2) * exp(-0.5 * pow((x - mu), 2.0) / sigma2);
  return prob;
  
}

int main()
{
cout<< f(10.0, 4.0, 8.9) << endl;
return 0; 
}

```
   
   
  
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
  
 







