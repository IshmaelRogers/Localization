
[image1]: ./images/mclvsekf.png
[image2]: ./images/compare.png
[image3]: ./images/map.png
[image4]: ./images/pose.png
[image5]:



Mobile robot localization is the problem of determining a robot's pose from sensor data in a mapped environment. 

Monte Carlo Localization algorithms represent a robot’s belief by a set of weighted hypotheses (samples), which approximate the *posterior* under a common *Bayesian formulation* of the localization problem.

Mixture-MCL: ntegrates two complimentary ways of generating samples in the estimation. 

This algorithm is applied to mobile robots equipped with range finders. A kd-tree is learned that permits fast sampling. robustness and computational efficiency are important system parameters. k can take on an positive integre value to specificy the dimension of the tree.



# Monte Carlo Localization 

Monte Carlo Localization solves the gloal localization and kidnapped robot problem in a highly robust and efficient way. It avoids the need extract features from sensor data by accomodating arbitrary noise distributions and non-linearities. Choosing MCL to keep track of the robot's pose is suggested because of how easy it is to code. 

The robot will be navigating inside its known map and collecting sensory information. The MCL will use these measurements to keep track of the robot's pose. MCL is refered to as a particle filter localization algorithm, because it uses particles to localize the robot. In robotics we consider particles as a virtual element that resembles the robot. Each particle has a position and orientation and has a guess about where the robot might be located. These particles are resampled each time the robot moves and senses uts environment. 

MCL can be used to localize any robot with pretty accurate sensors. We studied EKF previously to gain an appreciation for the power of the MCL algorithm. A further comparison is made below

![alt text][image1]

The computational memory and resolution of the solution can be changed by changing the number of particles distributed uniformly and randomly throughout the map.

Full comparison
---

![alt text][image2]


# Particle Filters

In the map below the robot does not know where it is located in the map. Since the initial state is unknown, the robot tries to estimate its pose by solving the Global localization problem. 

![alt text][image3]


The on-board range sensors permit it to avoid obstacle such as walls or doors and help the robot determine its location. The current robot pose are represented using the 3 state variables shown below.


![alt text][image4]


Particles are initial spread out randomly and unifpormly throughout the map. Each partlice has the same 3 state variables. Each particle represents the hypothesis of where the robot might be. Particles are assigned a weight, which is the difference between the robot's actual pose and the particle's predicted pose. The importance of the particle is detemined by its weight. The bigger the particle the more accruate. Bigger particles are more likely to survive after a resampling. 

After several iterations of the monte carlo locationize algortihm and different stages of resampling, particles converge and estimate the robot's pose.


# Bayes Filtering

A recursive Bayes filter will estimate the posterior distribution of a robot's position and orientation based on sensory information.

Using this approach roboticist can estimate the state of a dynamical system from sensor measurements.

Definition 

* **Dynamical System** : The mobile robot and its environment
* **State** : The robot's pose, including its position and orientation
* **measurements** : Perception data (i.e laser scanners) and odometry data (rotary encoders)


The goal of Bayes filtering is to estimate a probability density over the state space conditions on the measurements. The probability density (the posterior) is called the belied and is denoted as: 

<a href="https://www.codecogs.com/eqnedit.php?latex=Bel(X_t)&space;=&space;P(X_t&space;|Z_1...t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Bel(X_t)&space;=&space;P(X_t&space;|Z_1...t)" title="Bel(X_t) = P(X_t |Z_1...t)" /></a>

Where:

* <a href="https://www.codecogs.com/eqnedit.php?latex=X_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_t" title="X_t" /></a> is the ``state`` at time t
* <a href="https://www.codecogs.com/eqnedit.php?latex=Z_1...t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z_1...t" title="Z_1...t" /></a> is the ``measurement`` from time 1 up to time t. 

Probability:
---
Given a set of probabilities, <a href="https://www.codecogs.com/eqnedit.php?latex=P(A|B)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(A|B)" title="P(A|B)" /></a>. is calculated as follows:

<a href="https://www.codecogs.com/eqnedit.php?latex=P(A|B)=\frac{P(B|A)&space;\ast&space;P(A)}{P(B)}&space;=\frac{P(B|A)\ast&space;P(A)}{P(A)\ast&space;P(B|A)&plus;P(\sim&space;A)&space;\ast&space;P(B|&space;\sim&space;A)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(A|B)=\frac{P(B|A)&space;\ast&space;P(A)}{P(B)}&space;=\frac{P(B|A)\ast&space;P(A)}{P(A)\ast&space;P(B|A)&plus;P(\sim&space;A)&space;\ast&space;P(B|&space;\sim&space;A)}" title="P(A|B)=\frac{P(B|A) \ast P(A)}{P(B)} =\frac{P(B|A)\ast P(A)}{P(A)\ast P(B|A)+P(\sim A) \ast P(B| \sim A)}" /></a>


Concept check: 

```
This robot is located inside of a 1D hallway which has three doors. The robot doesn't know where it is located in this hallway, but it has sensors onboard that can tell it, with some amount of precision, whether it is standing in front of a door, or in front of a wall. The robot also has the ability to move around - with some precision provided by its odometry data. Neither the sensors nor the movement is perfectly accurate, but the robot aims to locate itself in this hallway.

The mobile robot is now moving in the 1D hallway and collecting odometry and perception data. With the odometry data, the robot is keeping track of its current position. Whereas, with the perception data, the robot is identifying the presence of doors.
n this quiz, we are aiming to calculate the state of the robot, given its measurements. This is known by the belief: P(Xt|Z)!

Given:

P(POS): The probability of the robot being at the actual position
P(DOOR|POS): The probability of the robot seeing the door given that it’s in the actual position
P(DOOR|¬POS): The probability of the robot seeing the door given that it’s not in the actual position

Compute:

P(POS|DOOR): The belief or the probability of the robot being at the actual position given that it’s seeing the door.


```


The Key Idea in MCL
---

Represent the belief about the state of the system with a set of samples aka *particles, drawn according to the *posterior* distribution over robot poses. 

The MCL represents the posteriors by a random collection of weighted particles which approxumates the desired distribution. 

Particles filters are known as condensation algorithms in computer vision fields.

Advantages
---

1. Can accommodate arbitary sensor characteristics, motion dynamics and noise distributions
2. Are universal density approximators, weakening the restrictive assumptions on the shape of the posterior density when compared to previous parametric approaches. 
3. Focuses computational resources in area that are most relevant, by sampling in proportion to the posterior likelihood.
4. Controlling the number of samples on-line, particle filters can adapt to available computational resources. The same code can be executed on computers with different speeds without the need for modification. 
5. Easy to implement 


Disadvantages
---

1. The *stochastic nature of the approximation can cause pitfall. If the sample set size is small, even a well-localized robot might lose track of its position jus because MCL fails to generate a sample in the right location. 
2. MCL is cannot handle the kidnapped robot problem, since there might not be any surving samples nearby the robot's new pose after it has been kidnapped. 
3. The basic algorithm degrades poorly when sensors are too accurate. Regular MCL will fail with perfect, noise-free sensors.

The above disadvantages can be overcome with the following: 

 * By augmenting the sample set through uniformly distributed samples
 * Generating samples consistent with the most recent sensor reading
 * Assuming a higher level of sensor noise than is actually the case 
 
 The above modification will yield improved performance but they are questionable from a mathematical standpoint. Interperting the results of these modifications are difficult because they do not approximate the correct density. 


The above disadvantages can be overcome with an extension of the MCL.

Mixture-MCl - address. all these problems in a way that is mathematically motivated. It works by modifying the way samples are generated in MCL. 

Mixture-MCL combines: 
1. Regular MCL - first it guesses a new pose using odometery, then uses sensor measurements to assess the "importance" of this sample.
2. Dual MCL - inverts the sampling process. Guesses poses using the most recent sensor measurement, then uses odometry to assess the compliance of this guess with the robot's previous belief and odometry data. 

Mixture-MCL works well if the sample set size is small (i.e 50 samples) it recovers faster from robot kidnapping than any previous variation of MCL. It also works well when sensor models are too narrow for regular MCL. 

Mixtrue-MCL is uniforly superior to regular MCL and particle filters. 

Key disadvantage- a sensor model that permits fast sampling of poses is required! Model is not always able to be trivally obtained. 

Overcoming the disadvantages - sufficient use of statistics and density trees to learn a sampling model from the data. 

Further, during a pre-procesing phase sensor readings are mappd into a set of discriminationg features and potential robot poses are then drawn randomly using tree generated. After the tree is made dual sampling can be done very efficiently. 






MCL vs. EKF 

# Partical Filters 

Intro: Particle filters can be used for localizing autonomous robots. They can equip robots with the tool of probability, allowing them to represent their belief about their current state with random *samples. Furthermore, it can be used to estimate non-Gaussian, nolinear processes. 






Bayes Filtering
Pseudo Code
MCL in action
