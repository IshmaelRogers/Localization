Mobile robot localization is the problem of determining a robot's pose from sensor data.

Probablistic localization algorithm 

MCL algorithms represent a robot’s belief by a set of weighted hy- potheses (samples), which approximate the *posterior under a common Bayesian formulation of the localization problem.

Mixture-MCL: integrates two complimentary ways of generating samples in the estimation. 

This algorithm is applied to mobile robots equipped with range finders. A kd-tree is learned that permits fast sampling. 

robustness and computational efficiency are important system parameters. 



# Monte Carlo Localization 

# ( /Transfer to Localization ReadMe.md )

Introduction
---

Localization is the key problem in mobile robotics. The most fundamental problem to provideing mobile robots with autonomous capabilities. 

Localization problem (simple to most difficult)

# 1. Position tracking 
   * The initial robot pose is known, the problem is to compensate incremental errors in a *robot's odometery.*
   * Algorithms for position tracking often make restrictive assumptions on the size of the error and the shape of the robot's uncertainity.
# 2. Global localization problem
   * A robot is not told its initial pose but instead has to determine it on its on. 
   * The error in the robot's estimate cannot be assumed to be small.
# 3. Kidnapped robot problem
   * Used to test a robot's ability to recover from catastrphic localization failures. 
 
 Kalman filters are best used *position tracking* where the nature of small, incremental errors are used. 
 Kalman filters estimate posterior distributions of robot poses conditioned on sensor data. 
 
 Assumptions (restrictive)
 1. Gaussian-distributed noise
 2. Gaussian-distributed intitial uncertainty
 
 Kalman filters offer an elegant and efficient algorithm for localization, but the restrictions on the belief representation makes plain Kalman filters unable to handle global localization problems.
 
 Two families of algorithms help us overcome this limitation:
 
 1. Multi-hypohesis Kalman filter
   * Uses mixture of Gaussian to represent beliefs. This enables them to pursue multiple disticnt hypotheses, each of which is represented by a separate Gaussian. 
   * The Gaussian noise assumption is inherited from Kalman Filters. All pratical implementations extract low-dimensional features from the sensor data, therby ignoring much of the information acquired by a robot's sensors. 
 2. Markov localization 
   * Represents beliefs by piecewise constant functions (histograms) over the space of all possible poses. 
   * Piecewise constant functions are capable of representing complex multi-modal representations.
# (Transfer to Localization ReadMe.md/)

Monte Carlo Localization Concept

Monte Carlo Localization solves the gloal localization and kidnapped robot problem in a highly robust and efficient way. It avoids the need extract features from sensor data by accomodating arbitrary noise distributions and non-linearities. 

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
