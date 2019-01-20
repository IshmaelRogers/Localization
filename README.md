# Localization

[image1]: ./images/filter_example.png
[image2]: .images/bellCurve.png
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
[image19]: ./images/
[image20]: ./images/
[image21]: ./images/
[image22]: ./images/




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
   What is represented by a Gaussian distribution?
   
   * Predicted Motion
   * Sensor Measurement 
   * Estimated State of Robot
   
   
   Kalman Filters treat all noise as unimodal Gaussian. This is not the case in reality. 
   
   The algorithm is optimal if the noise is Gaussian. 
   
   NOTE: Optimal means that the algorithm minimizes the mean square error of the estimated parameters. 
   
   # Designing 1D Kalman Filters
   
   
   Naming conventions
   ---
   
   Since a robot is unable to sense the world around it with complete certainity it holds an internal belief
   
   A robot constrained to a plane can be identified with 3 state variables 
   
   *state: x*
   *measurement: z*
   *control action: u*
   
   
  1. Measurement update 
  
  Sensors provide values called meaurements denoted z_t, where t respresents the time that measurement was taken.
  
  Control actions change the state of the system.
  
 2. State Prediction 
  
  u_t is the predicted state at time t. 
  
  ![alt text][image4]
  
  Starting the Kalman cycle with an initial estimate of the state. 
  
  Measurement update: gain knowledge
  State prediction: lose knowledge due to uncertaintity of robot motion
  
 # Measurement update 
 
  Mean Calculation:
 ---
 
 We'll start the Kalman filter implementation with the measurement update 
 
 1D Robot Example: 
 
 A roaming robot. 
 
 The robot is thinks its current position is near 20- m mark, but it is not very certain. There the Gaussian of the PRIOR BELIF N(x:mu=20, sig2=9) has a wide probability distribution. 
 
 The robot then takes its very first data measurement. The measurement N(x:v=30, r2=3) data Z is more certain so it has a more narrower gaussian. 
 
 Given our prior belief about the robot's state and the measurement that it collected the robot's new belief lies somewhere between the two gaussians since it is a combination of them. 
 
 NOTE: Since the measurement is more certain the new belief lies closer to the measurement. label as *A*
 in the image below
 
  ![alt text][image5]
 
 mue: Mean of the prior belief 
 sig2: Variance of the prior belief
 
 v: Mean of the measurement
 r2: Variance of the measurement
 
 
 The new measn is a weighted sum of the prior belief and the measurement means. 
 
 uncertainity - a larger number represents a more uncertain probability distribution. 
 
 The mew mean should be biased towards the measurement update which has a smaller standard deviation than the prior.
 
mu_prime = (r2*mu + sig2*v)/(r2 + sig2)

NOTE: uncertainity of the prior is multiplied by the mean measurement to give it more weight

NOTE: uncertainity of the measurement is multiplied with the mean of the prior. 

After applying the formula above we generate a new mean of 27.5 wich we label on the following graph:

![alt text][image6]


 

 
 Variance Calculation
 ---
 
 The variance of the new state estimate will be more confident than our measurement. 
 
 The two Gaussians provide us with more informaton together than either Gaussian offered alone. As a result, our new state estimate is more confidenet than our prior belief and our measurement. 
 
 It has a higher peak and is narrower.
 
 the formula for the new variance 
 
 sig2prime = (1)/((1/r2)+(1/sig2))

Entering the variances from our example into this formula produces a new variance of 2.25

![alt text][image7]
 
 
 
  mue: Mean of the prior belief 
 sig2: Variance of the prior belief
 
 v: Mean of the measurement
 r2: Variance of the measurement
 
 tau: Mean of the posterior
 s2: Variance of the posterior
 
 
 PROGRAMMING: Mean and Variance Formulas in C++ 
 ---
 
 
 ``` cpp
 
 //the following code implements measurement_update in the Kalaman Cycle
 
 #include <iostream>
 #include <math.h>
 #include <tuple>
 
 using namespace std;
 
 double new_mean, new_var;
 
 tuple < double, double> measurement_update(double mean1, double var1, double mean2, double var2)
 {
 new_mean = (var2*mean1 + var1*mean2)/(var1 + var2)
 new_var = (1)/((1/var2)+(1/var1)) 
     return make_tuple(new_mean, new_var);
}

int main()
{

    tie(new_mean, new_var) = measurement_update(10, 8, 13, 2);
    printf("[%f, %f]", new_mean, new_var);
    return 0;
}

```

# State Prediction 

The second half of the Kalman filter's iterative cycle

Estimation that takes place after an uncertain motion. 

We pick up where we left off and the posterior belief becomes the prior belief. This is the robot's best estimate of its current location. 

Robot execute a motion command, move forward 7.5 meters N(x:mu=7.5, sig2=5), the results of this motion is a gaussian distribution centered around 7.5 m with variance of 5 m.

Calculating the new estimate aka the new posterior Gaussian:

1. add the mean of the motion to the mean of the prior --> Posterior N(x:=(mu1+mu2), sig2=(sig2_1 +sig2_2)) 
2. Add the variance of the motion to the variance of the prior (see line 1)
 
 ![alt text][image8]
 
 PROGRAMING: THE STATE PREDICTION 
 
 ```cpp
 
 //the following code implements state_prediction in the Kalaman Cycle
 
#include <iostream>
#include <math.h>
#include <tuple>

using namespace std;

double new_mean, new_var;

tuple<double, double> state_prediction(double mean1, double var1, double mean2, double var2)
{
    new_mean = mean1 + mean2;
    new_var =  var1 + var2;
    return make_tuple(new_mean, new_var);
}

int main()
{

    tie(new_mean, new_var) = state_prediction(10, 4, 12, 4);
    printf("[%f, %f]", new_mean, new_var);
    return 0;
}

```
 
  # 1-D Kalman Filter
  
  ![alt text][image9]
  
Below is support code to call the two functions one after the other as long as measurement data is available and the robot has motion commands. 
  
  ``` cpp
  
 #include <iostream>
#include <math.h>
#include <tuple>

using namespace std;

double new_mean, new_var;

tuple<double, double> measurement_update(double mean1, double var1, double mean2, double var2)
{
    new_mean = (var2 * mean1 + var1 * mean2) / (var1 + var2);
    new_var = 1 / (1 / var1 + 1 / var2);
    return make_tuple(new_mean, new_var);
}

tuple<double, double> state_prediction(double mean1, double var1, double mean2, double var2)
{
    new_mean = mean1 + mean2;
    new_var = var1 + var2;
    return make_tuple(new_mean, new_var);
}

int main()
{
    //Measurements and measurement variance
    double measurements[5] = { 5, 6, 7, 9, 10 };
    double measurement_sig = 4;
    
    //Motions and motion variance
    double motion[5] = { 1, 1, 2, 1, 1 };
    double motion_sig = 2;
    
    //Initial state
    double mu = 0;
    double sig = 1000;

    for (int i = 0; i < sizeof(measurements) / sizeof(measurements[0]); i++) {
        tie(mu, sig) = measurement_update(mu, sig, measurements[i], measurement_sig);
        printf("update:  [%f, %f]\n", mu, sig);
        tie(mu, sig) = state_prediction(mu, sig, motion[i], motion_sig);
        printf("predict: [%f, %f]\n", mu, sig);
    }

    return 0;
}

 ```
  
  
  # Multivariate Gaussian 
  
  Most robots models are moving in more than one dimension i.e a robot on a plane would have an x & y position. 
  
  We can't use multiple 1-D Gaussians to represent multi-dimensional systems because there may be correlations between dimensions that we would not be able to model by using independent 1-dimensional Gaussians. 
  
  2-D Gaissian Distribution
  ![alt text][image10]
  
  Consider x,y coordinate of a robot:
  
  A 2-D Gaussian can also be represented as follows
  
  ![alt text][image11]
  
  Where the contour lines show variations in height
  
  mean is a vector:
  
  mu = [mux
        muy]
        
 NOTE: An N-dimensional Gaussian would have a mean vector that are sized N by 1 
 
 Covariance is a vector:
 
 SIGMA = [ sig2_x   sigxsigy
           sigysigx  sig2y]
           
  - represensts the spread of the Gaussian into two dimensions. An N dimensional Gaussian would have a covariance matrix that is of size N x N
  
  PICTURE OF FORMULAS FOR MULTIVARIATE GAUSSIAN
  
  ![alt text][image14]
  
  NOTE: The eigenvalues and egienectors of the covariance matrix describe the amount and direction of uncertainity. 
  
  The diagonal value of the matrix represent the variance 
  
  The off diagonals represent the corrleation terms
  
  NOTE: The covariance matrix is always symetrical. 
  
  In the case of the 2-D Gaussian the two off diagonal elements are equal.
  
  If the correlation terms are non-zero the two axis are corrleatied and the gaussian will apear as a skewed oval
  
  ![alt text][image12]
  
  The equation to model multivariate Gaussian:
  
  ![alt text][image13]
  
  * D represents the number of dimensions present
  
  # Multidimensional KF
  
  The state is represented in a Nx1 vector 
  
  
 TOPIC: Position and velocity of robot
  
   The robot's state needs to be Observable in 1D case which means it can be directly measured through sensors.
   
   In multi-dimensional states there may exist *hidden state variables, such as velcoity since it is not directly measure. It is infered from other states and measurements. The position is observable.
   
   The position and velocity are linked through a formula 
   
   x_prime = x + x_dot*delta_t
   
  Given:
  
  initial_pos;
  
  Find:
  
  velocity;
 
 The estimate of the robot's state would look like the following: 
 
 ![alt text][image15]
 
 A Gaussian that is:
 
 1. very narrow in the x direction implying that the robot is very certain about it's position
 
 2. very wide in the y direction implying that the robot is not very certain about its velocity
 
 3. State prediction is calculated 
 
 NOTE: Knowing the relationship between the hidden variable and observable variable is key to calculating the state prediction
  
  Example: 1 iteration of the Kalman Filter takes 1s
  
  Use the formula to calculate the posterior state for each possible velocity  
  
  velocity = 0 
  
  the posterior state would be identical to the prior. 
  
  ![alt text][image16]
  
  We can expand the above graph to show the correlation between the velocity and the location of the robot
  
![alt text][image17]
  
  
  Measurement Update Step
  ---
  The initial belief was useful to calculate the state prediction but has no additional value. The result of the state prediction is the Prior Belief for the measurement update. 
  
  If the new measurement suggests a location of X equals 50 then, we apply the measurement update to the prior, the posterior will be very small and centered around X equals 50 and x_dot equals 50. 
  
  ![alt text][image18]
  
  REVIEW!
  
  # Design of Multidimensional KF
  
  State prediction 
  ---
  
  State Transition
  --
  
  - the state transition function advances the state from time _t_ to time t+ 1 --> the relationship between the robot's position, x, and veloctity, x_dot.
  
  NOTE: We are assuming that the robot's velocity is not changing here.
  
  xprime = x + delta_t*xdot
  
  xdot_prime = xdot
  
  
 In matrix form 
 
 [x   ]' = [1 delta_t; 0   1] * [x;, xdot]
 [xdot]
  
  
  The State Transisition Function is denoted *F
  
  xprime = F * *x
  
  
  NOTE The above equation can be expanded to account for process noise with a term in the equation:
  
  xprime = F * *x + *noise 
  
  *nois ~ N(0,Q)
  
  NOTE: *P* represents the state covariance in localization
  
  Multiplying the state, x by *F*, the the covariance wll be affected by the square of F
  
  In matrix form:
  
  Pprime = F(times)P(times)F_transpose
  
  To calculate posterior covariance, the prior covariance is multiplied by the state transition function square, and Q added as an increase of uncertainity due to process noise. 
  
  NOTE: Q can account for a robot slowing down unexpectedly, or being drawn off course by an external influence 
  
  Pprime = FPF_transpose + Q 
  
  NOTE: The mean and covariance has been update as part of the state prediction
  
  Measurement Update
  --
  
  Consider the example where we are tracking the position and velocity of a robot in the x-dimension, the robot was taking measurements of the location only (velocity is a hidden state variable)
  
  The measurement function is very simple - a matrix that contain a one and a zero. 
  
  z = [1 0] * [x; xdot]
  
  The Measurement Function, *H*
  
- demonstatres how to map the state to the observation, z

Formulas for measurement update step 

* measurement residual, *y - the difference between the measurement and the expected measurement based on the prediction
i.e a comparision of where the measurement tells us we are vs. where we think we are. 

1. y = z - H * xprime

Consider the measurement noise, R. This formula maps the state prediction covariance into the measurement space and adds the measurement noise. 

NOTEL The result "S", will be used in a coming equation to calulate the Kalman Gain

2. S = HPprimeH_transpose + R

Kalman Gain
---

- determines how much weight should be placed on the state prediction and how much on the measurement update. It is an averaging factor that changes depending on the uncertainity of the state prediction and measurement update.

K = Pprime * H_transpose * S^-1 

x = xprime + Ky

# PROGRAMMING THE MULTI-DIMENSIONAL KALMAN FLITER

``` cpp

#include <iostream>
#include <math.h>
#include <tuple>
#include "Core" // Eigen Library
#include "LU"   // Eigen Library

using namespace std;
using namespace Eigen;

float measurements[3] = { 1, 2, 3 };

tuple<MatrixXf, MatrixXf> kalman_filter(MatrixXf x, MatrixXf P, MatrixXf u, MatrixXf F, MatrixXf H, MatrixXf R, MatrixXf I)
{
    for (int n = 0; n < sizeof(measurements) / sizeof(measurements[0]); n++) {

        // Measurement Update
        MatrixXf Z(1, 1);
        Z << measurements[n];

        MatrixXf y(1, 1);
        y << Z - (H * x);

        MatrixXf S(1, 1);
        S << H * P * H.transpose() + R;

        MatrixXf K(2, 1);
        K << P * H.transpose() * S.inverse();

        x << x + (K * y);

        P << (I - (K * H)) * P;

        // Prediction
        x << (F * x) + u;
        P << F * P * F.transpose();
    }

    return make_tuple(x, P);
}

int main()
{

    MatrixXf x(2, 1);// Initial state (location and velocity) 
    x << 0,
    	 0; 
    MatrixXf P(2, 2);//Initial Uncertainty
    P << 100, 0, 
    	 0, 100; 
    MatrixXf u(2, 1);// External Motion
    u << 0,
    	 0; 
    MatrixXf F(2, 2);//Next State Function
    F << 1, 1,
    	 0, 1; 
    MatrixXf H(1, 2);//Measurement Function
    H << 1,
    	 0; 
    MatrixXf R(1, 1); //Measurement Uncertainty
    R << 1;
    MatrixXf I(2, 2);// Identity Matrix
    I << 1, 0,
    	 0, 1; 

    tie(x, P) = kalman_filter(x, P, u, F, H, R, I);
    cout << "x= " << x << endl;
    cout << "P= " << P << endl;

    return 0;
}



```

# START HERE AND FINISH ATER PROJECT!

  Extended Kalman Filter 
  
  EKF Example 
  
  Limitations 
  
  Extended Kalman Filter 
  
  Lab: Kalman Filter 
  
  Montel Carlo Localization 
  
  Build MCL in C++ 
  
  Project: Where am I? 
  
 







