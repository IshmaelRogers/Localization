# Localization

[image1]: ./images/filter_example.png
[image2]: ./images/bellCurve.png
[image3]: ./images/probvdist.png
[image4]: ./images/2steps.png
[image5]: ./images/newbelief.png
[image6]: ./images/newmean.png
[image7]: ./images/posterior.png
[image8]: ./images/posterior2.png
[image9]: ./images/sp_mu.png
[image10]: ./images/2dGaus.png
[image11]: ./images/2d_gauss_alternate.png
[image12]: ./images/correlated.png
[image13]: ./images/mVG_equation.png
[image14]: ./images/formulas_4MVG.png
[image15]: ./images/state_est.png 
[image16]: ./images/graph1.png
[image17]: ./images/correlation_vel_pos.png
[image18]: ./images/posterir_belief.png
[image19]: ./images/linear_trans.png
[image20]: ./images/nonlin_trans.png
[image21]: ./images/approximated.png
[image22]: ./images/Taylor_series.png
[image23]: ./images/first2terms.png
[image24]: ./images/summary.png
[image25]: ./images/multidimen_TS.png
[image26]: ./images/1st2.png 
[image27]: ./images/jacobian.png
[image28]: ./images/expanded_jacobian.png
[image29]: ./images/meas_function.png
[image30]: ./images/polar_cart.png
[image31]: ./images/hofxprime.png 
[image32]: ./images/ts_hofx.png 
[image33]: ./images/H.png
[image34]: ./images/compute_jacobian.png
[image35]: ./images/ekf_equations.png
[image36]: ./images/summary1.png
[image37]: ./images/drone.pmg
[image38]: ./images/perp.png
[image39]: ./images/Jacobian_quad.png
[image40]: ./images/partials.png
[image41]: ./images/calculated_H.png
[image42]: ./images/ekf_eqs.png
[image43]: ./images/
[link1]: https://www.udacity.com/course/artificial-intelligence-for-robotics--cs373 




# Introduction

Localization is the challenge of determining the robot's *pose* in a mapped environment. It implements a probablistic algorithm to filter out noisey sensor measurements. It is considered a key problem in mobile robotics and is the most fundamental problem solving when provideing mobile robots with autonomous capabilities.  

Pose - is the robot's (x,y) coordinates + the orientation (theta) relative to a known start point. Usual represented as a 3x1 vector as shown below.

[x, y, theta]' 


Localization Challenges
--

There are 3 localization problems that range simple to most difficult, depending on the the deired goal. 

1. Position tracking
In this problem the initial robot pose is known, the challenge is to estimate the robot's pose as it moves around an environment. Unvertainty in the robot's movements make this a challenging problem in the field. However, to compensate for the incremental errors in the robot's odometry, the algorithm makes restrictive assumptions about the size of the error and the shape of the robot's unvertainity distribution.

2. Global localization problem
 In this problem, uncertainity is much higher than the position tracking problem. The robot is unaware of its initial pose and it must determine it on its on. Unlike the position tracking problem, the error in the robot's estimate cannot be assumed to be small.
 
3. Kidnapped robot problem
This problem is similar to the Global localization one but the robot can be removed from its current position and relocated to another random position on the map. This indeed the most difficult localization problem to solve. Efforts to solve this problem are made because it can help the robot prepare for the worst possible case. More specifically, it teaches the robot to revocer from localization failures and miscalculations.

Tools
---
 
 * Kalman filters 
 Kalman filters are best used to solve  *position tracking* where the nature of small, incremental errors are exploited. It estimates *posterior distributions* of robot poses that are conditioned on sensor data. 
 
 Kalman Filter Assumptions (restrictive)
 1. Gaussian-distributed noise
 2. Gaussian-distributed intitial uncertainty
 
 Kalman filters offer an elegant and efficient algorithm for localization, but the restrictions on the belief representation makes plain Kalman filters unable to handle global localization problems.
 
 2 families of algorithms help us overcome this limitation:
 
 1. Multi-hypohesis Kalman filter
  This algorithms uses a mixture of Gaussians to represent beliefs. This enables the the system to pursue multiple disticnt hypotheses, each of which is represented by a separate Gaussian. In this approach, the Gaussian noise assumption is inherited from Kalman Filters. In general, all pratical implementations of this algorithm extract low-dimensional features from the sensor data, therby ignoring much of the information acquired by a robot's sensors. 
 2. Markov localization 
 This form of loxcalization represents beliefs by piece-wise constant function (i.e histograms) over the space space of all possible poses. Piece-wise constant functions are capable of representign complex multi-modal representations. 
 
To summarize the different types of localization algorithms, we breifly revisit the each localization problem:

* The position tracking problem is much easier to solve than the global localization one, because the estimate of the robot's error is assumed to be small. Furthermore, in the global localization problem the robot is completely unware of its pose and must estimate it with a technique called *sensor fusion*

* The kidnapped robot problem is useful when designing a robot that can recover from localization malfunctions and miscalculations. These are events that are likely to occur in dynamic environments where predictablity is limited. 

Starting with a known map of the environment, and a sensor susceptible to sensor noise like lidar or ultrasonics, the robot will start off with guesses (probablities) about where it is relative to the start point. Over time the robots should narrow down on a precise location. 

More Localization algorithms
----

* Extended Kalman Filter - The most commonly used estimates the state of non linear models.

* Markov - a base filter localization, that maintains a probablitlity distribution over the set of possible poses (i.e position and orientation values of the robot.)

* Gird - histogram filter that estimates the pose of a robot using grid. See Udacity's Free Course in Artificial Intelligence for Robotics [link1] 
* Monte Carlo - A particle filter approach that is relatively easy to implement. 


  
  
 
  
  
  
  
  
  Montel Carlo Localization 
  
  Build MCL in C++ 
  
  Project: Where am I? 
  
 







