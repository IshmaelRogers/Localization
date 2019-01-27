# Ishmael Rogers
# Robotics Software Engineer
# Infinitely Deep Robotics Group, LLC
# 2019

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
[image11]: ./2d_gauss_alternate.png
[image12]: ./images/correlated.png
[image13]: ./images/mVG_equation.png
[image14]: ./images/formulas_4MVG.png
[image15]: ./images/state_est.png 
[image16]: ./Limages/graph1.png
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
[link1]: ./https://www.udacity.com/course/artificial-intelligence-for-robotics--cs373

# Kalman Filters

 ### Background
The Kalman filter is an estimation algorithm that is used widely in controls. It works by estimating the value of a variable in real time as the variable is being collected by a sensor. This variable can be position or velocity of the robot. The Kalman Filter can take data with a lot of uncertainty or noise in the measurements and provide an accurate estimate of the real value very quickly 
  
  ### Applications
  
   * Position tracking for a mobile robot
   * Feature tracking 
    
  ### Variations 
  
  Kalman Filter - applied to linear systems
  
  Extended Kalman Filter - applied to nonlinear system 
  
  UKF - highly nonlinear - http://ais.informatik.uni-freiburg.de/teaching/ws12/mapping/pdf/slam05-ukf.pdf
 
 Example: Underwater Robotics
 ---
 
 Consider an Underwater robot equiped with a barometer for monitoring the pressure as the robot swims through the water. 
 
 Problems: 
 
 * The pressure measurements from the barometer are not perfectly accurate 
 * Electrcial noise from the sensor introduces more errors into the measurement. 
 
 Solution: 
 
 When the pressure sensor starts collecting data, the Kalman filter begins to narrow in and estimates the actual pressure. In addition to the sensor readings, the Kalman filter accounts for the uncertainity of the sensor readings which are specific to the type of sensor being used and the environment it is being used in. 
 
 The process
 ---
 
 What happens every time a measurement is recorded? 
 
 Kalman filter is an iteration of the following 2 steps: 
 
 1. Measurement update 
 
 2. State prediction 
 
 ![alt text][image1]
 
  Robot Uncertainty 
  ---
  
  Ideal vs Real world
  
It is often useful to model robots using an "Ideal World" assumption. This assumes that sensors are 100% accurate and the robots executes perfect motion. If a mobile robot is commeded to move 40 meter due west, the robot is gauranteed to reach its desired goal in a finite amount of time. However, in the real world, robots may encounter complexities that result in the robot's movement being imprecise. These complexities can include uneven terrain, wheel slipping (in wheeled mobile robots) and environmental factors such as wind speed, temperature and humidity. In both situations the robot knows it's starting position. However, In the "Real World" the same movement command will not produce the same results as in the ideal world assumption; it will like not reach the desired goal in a reasonable amount of time. Using the Real World assumption we know that sensors are inheriently prone to error. Therefore, when designing a complex system using sensors, we use datasheets to gain an understanding of the limits of the sensor and design our solutions to compensate for that error. 
 
  
  Example:
  
  If we record the robot moving 10 meter forward a total of 100 times we would get a plot like this:
  ![alt text][image2] 
  
   This graph displays a probabilty distribution of the robot's fianl position after multiple iterations of the movement.
   
   The X-axis is the distance traveled by the robot
   The Y-axis is how often the robot stopped at that distance
   
   It should be clear that the sensor data plays an important role in the localization problem. As discussed previously, real world sensors contain a lot of noise and must be filtered out in order to be useful for any localization problem. 
   
   The robots internal belief about its state is represented by a gaussian disribution. The shape of the bell curve determines how "confident" our robot is about their current state. In an environment with only a few enviornmental factors, movement commands are precise and the shape of the Gaussian distribution is narrow (the robot is confident). Conversely, evnironments that contain many environmental factors (as is the case with a rescue mission robot), the Gaussian distribution is much wider; therefore the robot is less certain about its current state. 
  
   Advantage
   ---
   
   Movement and Sensory measurements are uncertain, the Kalman Filter takes in account the uncertainity of each sensor's measurement to help the robot better sense its own state. This estimatation is fast; happening after only a few sensor measurements. 
   
   STEPS
   ---
   
   To begin the Kalman cycle, the system generates an intitial guess about its state and takes in account of expected uncertainity. 
   
   Sensor Fusion - is the process in which the Kalman filter is used to calculate an accurate estimate of the system's state using data from multiple sensors. 
   
  # 1D Gaussian
  
   It is worth remphasizing the basis of the Kalman Filter. At the core of this probablisitic algorithm is the Gaussian distribution also known as a bell curve. It can be used to represent the robot's certainity about its on pose relative to the world. For example if Hexpod Zero was commanded to execute 1 motion, the system's final location can be represented as a Gaussian, where the overall shape can illustrate how certain the robot is. Considering the real world assumption, the exact location is not certain, however the level of uncertainity is bounded. 
   
   The role of a Kalman Filter
   ---
   
   After a movement command or a measurement update, the KF outputs a *unimoda* Gaussian distribution. This is considered its best guess at the true value of a parameter
   
  NOTE: Unimodal - implies a single peak present in the distribution.  
   
  NOTE: A Gaussian distribution is a probability distribution which is a continous function. 
   
   Claim:
   
   The probability that a random variable, x, will take a value between x1 and x2 is given by the integral of the function from x1 to x2. 
   
   <a href="https://www.codecogs.com/eqnedit.php?latex=P(x_1&space;<&space;x&space;<&space;x_2)&space;=&space;x_2\int&space;x_1&space;f_x(x)dx" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(x_1&space;<&space;x&space;<&space;x_2)&space;=&space;x_2\int&space;x_1&space;f_x(x)dx" title="P(x_1 < x < x_2) = x_2\int x_1 f_x(x)dx" /></a>
   
   In the image below, the probability of the rover being located between 8.7m and 9m is 7%
   
   ![alt text][image3]
   
   
   Mean and variance
   ---
   
   A gaussian is characterized by two parameters 
   
  <a href="https://www.codecogs.com/eqnedit.php?latex=\mu" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu" title="\mu" /></a> -  *mean* represents the most probable occurrence. 
   <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma^{2}" title="\sigma^{2}" /></a> - *Variance* represents the width of the curve 
   
 Gaussian distributions are frequently abbreviated as:
   
   <a href="https://www.codecogs.com/eqnedit.php?latex=N(x:&space;\mu,&space;$&space;\sigma&space;^{2}&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N(x:&space;\mu,&space;$&space;\sigma&space;^{2}&space;)" title="N(x: \mu, $ \sigma ^{2} )" /></a>
   
   Formula
   ---
   
   <a href="https://www.codecogs.com/eqnedit.php?latex=p(x)&space;=&space;\frac{e^{\frac{-(x-\mu&space;)^{2}}{2\sigma&space;^{2}}}}{\sigma&space;\sqrt{2\pi&space;}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(x)&space;=&space;\frac{e^{\frac{-(x-\mu&space;)^{2}}{2\sigma&space;^{2}}}}{\sigma&space;\sqrt{2\pi&space;}}" title="p(x) = \frac{e^{\frac{-(x-\mu )^{2}}{2\sigma ^{2}}}}{\sigma \sqrt{2\pi }}" /></a>
   
   NOTE: This formula contains an exponential of a quadratic function. The quadratic compares the value of x to <a href="https://www.codecogs.com/eqnedit.php?latex=\mu" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu" title="\mu" /></a>. In the case that x=<a href="https://www.codecogs.com/eqnedit.php?latex=\mu" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu" title="\mu" /></a> the exponential is equal to 1 (e^0 = 1). 
   
   NOTE: The constant in front of the exponential is a necessary normalizing factor. 
   
   In discrete proababiity, the probabilities of all the options must sum to one. The area underneath the function always suns to one i.e <a href="https://www.codecogs.com/eqnedit.php?latex=\int&space;p(x)&space;$&space;dx&space;=&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\int&space;p(x)&space;$&space;dx&space;=&space;1" title="\int p(x) $ dx = 1" /></a>
   
   Coding the 1D Gaussian in C++
---
The following code will allow us to calculate of a value occuring given mean and variance

```cpp

#include <iostream>
#include <math.h>

using namespace std;

double f(double mu, double sigma2, double x) // create a 3 input function that takes in the mean, variance and the value variable
{

  prob = 1.0 / sqrt(2.0 * M_PI * sigma2) * exp(-0.5 * pow((x - mu), 2.0) / sigma2);  // calculate the 1D Gaussian using the formula 
  return prob; // return the results to be used throughout the program 
  
}

int main() // initialize the main function
{
cout<< f(10.0, 4.0, 8.9) << endl; // call the function with mean = 10, variance = 4.0 and x = 8.9 
return 0; 
}

```
Results : 0.120985



What can be represented by a Gaussian distribution?
   ---
   
   * Predicted Motion
   * Sensor Measurement 
   * Estimated State of Robot
   
   
   Kalman Filters treat all noise as unimodal Gaussian. However, this is not the case in reality but the algorithm is optimal if the noise is Gaussian. 
   
   NOTE: Optimal means that the algorithm minimizes the mean square error of the estimated parameters. 
   
   # Designing 1D Kalman Filters
   
   
   Naming conventions
   ---
   
   Since a robot is unable to sense the world around it with complete certainity, it holds an internal belief *Bel( )* It's best guess at the state of the environment including itself.
   
   A robot constrained to a plane can be identified with *3 state variables* Two coordinates, x and y to identify its position, and one angle *yaw* to identify its orientation. 
   
   **<a href="https://www.codecogs.com/eqnedit.php?latex=x_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_t" title="x_t" /></a>: state** - Since the state of the system changes over time, we use a subscript, <a href="https://www.codecogs.com/eqnedit.php?latex=t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t" title="t" /></a>, to denote the state at a particular. 
   **<a href="https://www.codecogs.com/eqnedit.php?latex=z_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z_t" title="z_t" /></a>: measurement** - Obtained through the use of sensors. We use a subscript, <a href="https://www.codecogs.com/eqnedit.php?latex=t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t" title="t" /></a>, to represent the measurement obtained at time <a href="https://www.codecogs.com/eqnedit.php?latex=t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t" title="t" /></a>. 
   **<a href="https://www.codecogs.com/eqnedit.php?latex=u_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u_t" title="u_t" /></a>: control action** - Represents the change in state that occured between time <a href="https://www.codecogs.com/eqnedit.php?latex=t-1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t-1" title="t-1" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t" title="t" /></a> 
   

  
  ![alt text][image4]
  
  
  The Kalman cycle starts with an initial estimate of the state. From there the 2 iterative steps describe above are carried out. 
  
  1. Measurement update: where the robot uses sensors to gain knowledge about its environment.
  2. State prediction: the robot loses knowledge due to uncertaintity of robot motion.
  
  Below we explore these 2 steps in much greater detail. 
  
 # Measurement update 
 
 We'll start the Kalman filter implementation with the measurement update 
  Mean Calculation: 1-D Robot 
 ---
 
 Consider a roaming mobile robot starting at some origin. It travels the positive x direction and stops momentarily. The robot thinks its current position is near 20 m mark, but it is not very certain. Therefore, the Gaussian of the *prior belief*  <a href="https://www.codecogs.com/eqnedit.php?latex=N(x:&space;\mu&space;=20&space;,$&space;\sigma^2&space;=&space;9)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N(x:&space;\mu&space;=20&space;,$&space;\sigma^2&space;=&space;9)" title="N(x: \mu =20 ,$ \sigma^2 = 9)" /></a> has a wide probability distribution. 
 
 The robot then takes its very first data measurement. The measurement data, z, is more certain so it has a more narrower Gaussian represented as <a href="https://www.codecogs.com/eqnedit.php?latex=N(x:&space;\upsilon&space;=30&space;,$&space;r^2&space;=&space;3)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N(x:&space;\upsilon&space;=30&space;,$&space;r^2&space;=&space;3)" title="N(x: \upsilon =30 ,$ r^2 = 3)" /></a> 
 
 Given our prior belief about the robot's state and the measurement that it collected the robot's new belief lies somewhere between the two gaussians since it is a combination of them. The new mean is a weighted sum of the prior belief and the measurement means.
 
 NOTE: Since the measurement is more certain the new belief lies closer to the measurement. label as *A*
 in the image below
 
  ![alt text][image5]
 
  <a href="https://www.codecogs.com/eqnedit.php?latex=\mu" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu" title="\mu" /></a> : Mean of the prior belief 
  <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma^{2}" title="\sigma^{2}" /></a>: Variance of the prior belief
 
 <a href="https://www.codecogs.com/eqnedit.php?latex=\upsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\upsilon" title="\upsilon" /></a>: Mean of the measurement
 <a href="https://www.codecogs.com/eqnedit.php?latex=r^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r^2" title="r^2" /></a>: Variance of the measurement
 
 When it comes to uncertainity, a larger number represents a more uncertain probability distribution. It is intuitive to assume that the new mean should be biased towars the measurement which has a smaller stanard distribution than the prior. To accomplish this, the uncertaintiy of the prior is multiplied by the mean of the measure to give it more weight. Also, the uncertainity of the measurement is multiplied with the mean of the prior. See the equation below.

<a href="https://www.codecogs.com/eqnedit.php?latex=\mu{}'&space;=&space;\frac{r^2\mu&space;&plus;&space;\sigma^2\upsilon&space;}{r^2&plus;&space;\sigma^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu{}'&space;=&space;\frac{r^2\mu&space;&plus;&space;\sigma^2\upsilon&space;}{r^2&plus;&space;\sigma^2}" title="\mu{}' = \frac{r^2\mu + \sigma^2\upsilon }{r^2+ \sigma^2}" /></a>
 
After applying the formula above we generate a new mean of 27.5 and a variance of 2.25 which we label on the following graph:

![alt text][image6]

 Variance Calculation
 ---
 
 The variance of the new state estimate will be more confident than our measurement because the two Gaussians provide us with more informaton together than either Gaussian offered alone. As a result, our new state estimate is more confidenet than our prior belief and our measurement. This is easily noticed by a higher peak and is narrower shape. The formula for calculating the variance is illustrated below. 
 
<a href="https://www.codecogs.com/eqnedit.php?latex=\sigma^2^'&space;=&space;\frac{1}{\frac{1}{r^2}&plus;\frac{1}{\sigma^2}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma^2^'&space;=&space;\frac{1}{\frac{1}{r^2}&plus;\frac{1}{\sigma^2}}" title="\sigma^2^' = \frac{1}{\frac{1}{r^2}+\frac{1}{\sigma^2}}" /></a>

Entering the variances from our example into this formula produces a new variance of 2.25

![alt text][image7]
 
 
 
<a href="https://www.codecogs.com/eqnedit.php?latex=\mu" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu" title="\mu" /></a> : Mean of the prior belief 
  <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma^{2}" title="\sigma^{2}" /></a>: Variance of the prior belief
 
 <a href="https://www.codecogs.com/eqnedit.php?latex=\upsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\upsilon" title="\upsilon" /></a>: Mean of the measurement
 <a href="https://www.codecogs.com/eqnedit.php?latex=r^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r^2" title="r^2" /></a>: Variance of the measurement
 
 <a href="https://www.codecogs.com/eqnedit.php?latex=\tau" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau" title="\tau" /></a>: Mean of the posterior
 <a href="https://www.codecogs.com/eqnedit.php?latex=s^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s^2" title="s^2" /></a>: Variance of the posterior
 
 
 PROGRAMMING: Mean and Variance Formulas in C++ 
 ---
 The followign code creates a measurement_update function which is one part of the Kalman Filter Implementation
 
 ``` cpp
 
 //the following code implements measurement_update in the Kalaman Cycle
 
 #include <iostream>
 #include <math.h>
 #include <tuple>
 
 using namespace std;
 
 double new_mean, new_var; //initialize variables
 
 tuple < double, double> measurement_update(double mean1, double var1, double mean2, double var2) //create a function that returns a tuple of two double data types
 {
 new_mean = (var2*mean1 + var1*mean2)/(var1 + var2) // calculates the new mean
 new_var = (1)/((1/var2)+(1/var1)) // calculates the new variance
     return make_tuple(new_mean, new_var); // return a tuple to be used throughout the program 
}

int main()
{

    tie(new_mean, new_var) = measurement_update(10, 8, 13, 2); //unpacks the tuple into two variables 
    printf("[%f, %f]", new_mean, new_var);
    return 0;
}

```

Results: [12.40000, 1.60000]

# State Prediction 

Now we address the second half of the Kalman filter's iterative cycle. The State Prediction is the estimation that takes place after an uncertain motion or an innately noisey sensor. Since the Kalman filter is an iterative process, we pick up where we left off right after the measurement update. The posterior belief that we obtained in the last step now becomes the prior belief for the State Prediction step. This can be considered the robots best estimation of its current location (and a better measurement of it's initial estimation)

The robot now executes a motion command: move forward 7.5 meters, the results of this motion is a gaussian distribution centered around 7.5 m with variance of 5 m represented as <a href="https://www.codecogs.com/eqnedit.php?latex=N(x:\mu=7.5,&space;$&space;\sigma^2&space;=5)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N(x:\mu=7.5,&space;$&space;\sigma^2&space;=5)" title="N(x:\mu=7.5, $ \sigma^2 =5)" /></a>.

Formulas
---

Calculating the new estimate aka the new posterior Gaussian:

*Posterior Mean:* <a href="https://www.codecogs.com/eqnedit.php?latex=\mu'&space;=&space;\mu_1&space;&plus;&space;\mu_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu'&space;=&space;\mu_1&space;&plus;&space;\mu_2" title="\mu' = \mu_1 + \mu_2" /></a>

*Posterior Variance:* <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma^2^{'}&space;=&space;\sigma^2_1&space;&plus;&space;\sigma^2_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma^2^{'}&space;=&space;\sigma^2_1&space;&plus;&space;\sigma^2_2" title="\sigma^2^{'} = \sigma^2_1 + \sigma^2_2" /></a>

 ![alt text][image8]
 
 PROGRAMING: THE STATE PREDICTION 
 
 ```cpp
 
 //the following code implements state_prediction in the Kalaman Cycle
 
#include <iostream>
#include <math.h>
#include <tuple>

using namespace std;

double new_mean, new_var; // initialize the variables

tuple<double, double> state_prediction(double mean1, double var1, double mean2, double var2) //create a function that returns a tuple of 2 doubles

{
    new_mean = mean1 + mean2; // in the State Prediction step, the new mean is the sum of the prior blief's mean and the executed motion's mean. 
    new_var =  var1 + var2; // in the State Prediction step, the new variance is the sum of the prior belief's varianca and the executed motion's variance
    return make_tuple(new_mean, new_var);
}

int main()
{

    tie(new_mean, new_var) = state_prediction(10, 4, 12, 4); 
    printf("[%f, %f]", new_mean, new_var);
    return 0;
}

```
 Results: [22.00000, 8.0000]
 
 
  # 1-D Kalman Filter
  
  ![alt text][image9]
  
Below is support code to call the two functions one after the other as long as measurement data is available and the robot has motion commands. 
  
  ``` cpp
  
#include <iostream>
#include <math.h>
#include <tuple>

using namespace std;

double new_mean, new_var;

tuple<double, double> measurement_update(double mean1, double var1, double mean2, double var2)//measurement_update function
{
    new_mean = (var2 * mean1 + var1 * mean2) / (var1 + var2);
    new_var = 1 / (1 / var1 + 1 / var2);
    return make_tuple(new_mean, new_var);
}

tuple<double, double> state_prediction(double mean1, double var1, double mean2, double var2) // state prediction function
{
    new_mean = mean1 + mean2;
    new_var = var1 + var2;
    return make_tuple(new_mean, new_var);
}

int main()
{
    //Measurements and measurement variance
    double measurements[5] = { 5, 6, 7, 9, 10 }; // create an array with 5 elements
    double measurement_sig = 4; // define a variance for the measurement data
    
    //Motions and motion variance
    double motion[5] = { 1, 1, 2, 1, 1 }; // create an array with 5 elements 
    double motion_sig = 2; // define a variance for the motion data 
    
    //Initial state
    double mu = 0;  //initialize the robot with a zero mean
    double sig = 1000; // initialize the robot with a variance of 1000 

    for (int i = 0; i < sizeof(measurements) / sizeof(measurements[0]); i++) { // used to loop through measurement and motion arrays.
        tie(mu, sig) = measurement_update(mu, sig, measurements[i], measurement_sig); // uses the measurement_update update the mean and variance 
        printf("update:  [%f, %f]\n", mu, sig);
        tie(mu, sig) = state_prediction(mu, sig, motion[i], motion_sig); // uses the state_prediction update the mean and the variance
        printf("predict: [%f, %f]\n", mu, sig);
    }

    return 0;
}

 ```
  
  
  # Multivariate Gaussian 
  
So far we've looked at 1-D Gaussian distributions which are useful for representing a robots internal belief about its state. We implemented this idea in a 1-D Kalman filter which is a 2 step iterative process. The 1-D robot started out with an inital estimate of its state and then initiated a motion command, *u*. The robot then calculated a new Gaussian distribution that was more certain than the previous two (the initial estimate, and the motion command). This calulation is passed to the state prediction step as the *prior belief*. We used this 1D model to illustrate the most simple case.

In the real world most robots models are moving in multiple dimensions; for example a robot on a plane would have an x & y coordinates to identify its position. It would be conveient if we could use multiple 1-D Gaussians to represent multi-dimensional systems. However we can't because there may be correlations between dimensions that we would not be able to model by using independent 1-D Gaussians. 
  
  2-D Gaissian Distributions
  ---
  ![alt text][image10]
  
 A 2-D Gaussian can be use represent the x and y coordinates of a robot. The contour lines show variations in height. The same concepts can be applied to higher dimension Gaussian Distributions.  
  
  
  ![alt text][image11]


When we implemented the 1-D Gaussian previously,the mean,<a href="https://www.codecogs.com/eqnedit.php?latex=\mu" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu" title="\mu" /></a> and the variance,  <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma^{2}" title="\sigma^{2}" /></a>: was represened as a single value. In 2-D space, <a href="https://www.codecogs.com/eqnedit.php?latex=\mu" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu" title="\mu" /></a> and the variance,  <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma^{2}" title="\sigma^{2}" /></a> is a vector that contains a mean value for each dimension. 
  
  mean is a vector:
  
 <a href="https://www.codecogs.com/eqnedit.php?latex=\mu" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu" title="\mu" /></a> = <a href="https://www.codecogs.com/eqnedit.php?latex=\begin{bmatrix}&space;\mu_x\\&space;\mu_y&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;\mu_x\\&space;\mu_y&space;\end{bmatrix}" title="\begin{bmatrix} \mu_x\\ \mu_y \end{bmatrix}" /></a>
        
 
 NOTE: An N-dimensional Gaussian would have a mean vector that are sized N by 1 
 
 Variance is also represented differently than in 1-D example. In 2-D space it is a 2 x 2 matrix that is referred to as the *covariance* matrix. This matrix represents the spread of the Gaussian into two dimensions. 
 
  
 <a href="https://www.codecogs.com/eqnedit.php?latex=\Sigma&space;=&space;\begin{bmatrix}&space;\sigma^2&space;&&space;\sigma_x\sigma_y&space;\\&space;\sigma_y\sigma_x&&space;\sigma^2_y&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Sigma&space;=&space;\begin{bmatrix}&space;\sigma^2&space;&&space;\sigma_x\sigma_y&space;\\&space;\sigma_y\sigma_x&&space;\sigma^2_y&space;\end{bmatrix}" title="\Sigma = \begin{bmatrix} \sigma^2 & \sigma_x\sigma_y \\ \sigma_y\sigma_x& \sigma^2_y \end{bmatrix}" /></a>

The diagonal values of the matrix represent the variance while the off-diagonal values represent the correlation terms. 
  
  NOTE: The covariance matrix is always symmetrical i.e <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma_x\sigma_y&space;=&space;\sigma_y\sigma_x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma_x\sigma_y&space;=&space;\sigma_y\sigma_x" title="\sigma_x\sigma_y = \sigma_y\sigma_x" /></a>. In the Case of the 2-D Gaussian the off diagonal elements are always equivalent
  
  
Cases:
--- 

If the x and y variances are both small and equal to each other. Then the distribution will be circular. This indicates that our system is equal sure about it's position in the x and y axis.

If the x and y variances are not equal then you will be more or less certain about the location of the robot in the either the x or y axis, and vice versa. The Gaussian will become more oval. 

If the correlation terms are non-zero the two axis are correlated. The results are a skewed oval. If the terms multiply out to a positive or negative value, the oval will skew to the right or the left respectively. Once you get info about one axis it will reveal information about the other due to the correlation. 

The 2-D Gaussian can be modeled with the following equation: 
  
  
  PICTURE OF FORMULAS FOR MULTIVARIATE GAUSSIAN
  
  ![alt text][image14]
  
  NOTE: The *eigenvalues* and *egienvectors* of the covariance matrix describe the amount and direction of uncertainity. 
  
  ![alt text][image12]
  
  The equation to model multivariate Gaussian:
  
  ![alt text][image13]
  
  * D represents the number of dimensions present
  
  # Multidimensional KF
  
  In our one dimensional example, the system's state was represented by one variable. In 2-D the state is represented by an Nx1 vector that contains N state variables. Considering the 2-D robot, these two state variables could be its x and y position on a given map <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\begin{bmatrix}&space;x\\&space;y&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\begin{bmatrix}&space;x\\&space;y&space;\end{bmatrix}" title="\begin{bmatrix} x\\ y \end{bmatrix}" /></a>, Or the position and velocity of the robot <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\begin{bmatrix}&space;x\\&space;\dot{x}&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\begin{bmatrix}&space;x\\&space;\dot{x}&space;\end{bmatrix}" title="\begin{bmatrix} x\\ \dot{x} \end{bmatrix}" /></a>.
  
  
   The robot's state needs to be Observable in 1D case which means it can be directly measured through sensors. However, in multi-dimensional states there may exist *hidden state variables*, such as velocity. The velocity of the robot must instead be infered from other states and measurements. The position is observable because it can be measured directly with the various sensors.
   
   NOTE: Hidden state variables cannot be measured with the sensors available.
   
   
   In Physics, the position and velocity over time are linked through a formula:
   
 <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;{x}'=x&space;&plus;&space;\dot{x}\Delta&space;t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;{x}'=x&space;&plus;&space;\dot{x}\Delta&space;t" title="{x}'=x + \dot{x}\Delta t" /></a>
  
  Looking at this one the graph, if the robot's position is given but its velocity is not, the estimate of the robot's state would look like the following: 
 
 ![alt text][image15]
 
 Below are features of this Gaussian that must be understoo:
 
 1. The distribution is very narrow in the <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;x" title="x" /></a> direction implying that the robot is very certain about it's position.
 
 2. The distribution is very wide in the <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\dot{x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\dot{x}" title="\dot{x}" /></a> direction because the robot's velocity is completely unknown
 
State prediction 
---
 
Knowing the relationship between the hidden variable and observable variable is key to calculating the state prediction. To illustrate this point let's assume that one iteration of the Kalman Filter takes 1 second <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\Delta&space;t&space;=1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\Delta&space;t&space;=1" title="\Delta t =1" /></a>. We can use the formula to calculate the *posterior state* for each possible velocity. For example, if the velocity is zero, the posterior state would be identical to the prior. 
  
  ![alt text][image16]
  
  We can expand the above graph to show the correlation between the velocity and the location of the robot
  
![alt text][image17]
  
  
  Measurement Update
  ---
  The initial belief was useful to calculate the state prediction but is not useful otherwise. The result of the state prediction is the Prior Belief for the measurement update. 
  
  If the new measurement suggests a location of <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;x=50" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;x=50" title="x=50" /></a> then, we apply the measurement update to the prior, the posterior will be very small and centered around <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;x=50" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;x=50" title="x=50" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\dot{x}=50" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\dot{x}=50" title="\dot{x}=50" /></a>4. 
  
  ![alt text][image18]
 
 The results are the exact same as in the 1-D case, the Posterior Belief is just the weighted sum of the Prior Belief and the Measurement. As expected, it is more confident than either of the 2. The relationship between the two dimesnions narrows down the Posterior for the a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\dot{x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\dot{x}" title="\dot{x}" /></a> axis 
 
 If the robot moved from, <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;x=35" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;x=35" title="x=35" /></a> to <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;x=50" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;x=50" title="x=50" /></a> in 1 second, speed can be calculated. Two iterations of the Kalman filter cycle is enough to infer the robot's velocity. If we continue iterating through the measurement update and state prediction step, the robot's internal state with be updated to keep it aligned with where it is in the real world. 
 
  # Design of Multidimensional KF
  
  Linear algebra will allow us to easily work with multi-dimensional problems. We'll start with the state prediction in linear algebra form. 
  
  State Transition
  --
  State transition function advances the state from time <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;t" title="t" /></a> to <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;t&plus;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;t&plus;1" title="t+1" /></a> .The function is the relationship between the robot's position, x, and veloctity, <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\dot{x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\dot{x}" title="\dot{x}" /></a>. Let's assume that the robot's velocity is not changing. 
  
 <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;{x}'=x&space;&plus;&space;\dot{x}\Delta&space;t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;{x}'=x&space;&plus;&space;\dot{x}\Delta&space;t" title="{x}'=x + \dot{x}\Delta t" /></a>
  
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\dot{x}{}'=\dot{x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\dot{x}{}'=\dot{x}" title="\dot{x}{}'=\dot{x}" /></a>
  
  In matrix form the left side of the equation represents the posterior state . On the right is the state transition function and the prior state. The equation show how the state changes over the time period, <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\Delta&space;t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\Delta&space;t" title="\Delta t" /></a>
  
  
  NOTE: We are only working with the means for now. A note will appear informing you when the covariance matrix discussion is near. 
 
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\begin{bmatrix}&space;x\\&space;\dot{x}&space;\end{bmatrix}'&space;=&space;\begin{bmatrix}&space;1&space;&&space;\Delta&space;t&space;\\&space;0&space;&&space;1&space;\end{bmatrix}&space;\begin{bmatrix}&space;x\\&space;\dot{x}&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\begin{bmatrix}&space;x\\&space;\dot{x}&space;\end{bmatrix}'&space;=&space;\begin{bmatrix}&space;1&space;&&space;\Delta&space;t&space;\\&space;0&space;&&space;1&space;\end{bmatrix}&space;\begin{bmatrix}&space;x\\&space;\dot{x}&space;\end{bmatrix}" title="\begin{bmatrix} x\\ \dot{x} \end{bmatrix}' = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x\\ \dot{x} \end{bmatrix}" /></a>

The State Transisition Function is denoted *F and can be written as follows:
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;x'=Fx" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;x'=Fx" title="x'=Fx" /></a>
  
  To account for real world uncertainities the equation should also account for process noise. Therefore we introduce *noise* as a term in the above equation: 
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;x'=Fx&space;&plus;noise" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;x'=Fx&space;&plus;noise" title="x'=Fx +noise" /></a>
  
  <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;noise&space;\sim&space;N(0,Q)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;noise&space;\sim&space;N(0,Q)" title="noise \sim N(0,Q)" /></a>
  
  
  We move our discussion to the covariance. In that mathematics we use <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\Sigma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\Sigma" title="\Sigma" /></a> to represent the covariance of a Gaussian distribution. When solving the localization problem, it is common to use *P* to represent the state covariance.
  
  If you multiply the state, x by *F*, then the covariance will be affected by the square of F. In matrix form:
  
  In matrix form:
  
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;P'&space;=&space;FPF^T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;P'&space;=&space;FPF^T" title="P' = FPF^T" /></a>
  
  The equation should be affected by more than just the state transition function. Additional uncertainity may arise from the prediction itself. 
  
  To calculate posterior covariance, the prior covariance is multiplied by the state transition function squared, and Q is added as an increase of uncertainity due to process noise. Q can account for a robot slowing down unexpectedly, or being drawn off course by an external influence. 
  
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;P'&space;=&space;FPF^T&space;&plus;Q" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;P'&space;=&space;FPF^T&space;&plus;Q" title="P' = FPF^T +Q" /></a>
  
  At this point we have updated the mean and the covariance as part of the state prediction. 
  
  Measurement Update
  --
  
  Consider the example where we are tracking the position and velocity of a robot in the x-dimension, the robot was taking measurements of the location only because remember, velocity is a hidden state variable. The measurement fuction contains on a 1 and 0
  
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;z&space;=\begin{bmatrix}&space;1&space;&&space;0&space;\end{bmatrix}\begin{bmatrix}&space;x\\&space;\dot{x}&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;z&space;=\begin{bmatrix}&space;1&space;&&space;0&space;\end{bmatrix}\begin{bmatrix}&space;x\\&space;\dot{x}&space;\end{bmatrix}" title="z =\begin{bmatrix} 1 & 0 \end{bmatrix}\begin{bmatrix} x\\ \dot{x} \end{bmatrix}" /></a>
  
  The matrix <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\begin{bmatrix}&space;1&space;&&space;0&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\begin{bmatrix}&space;1&space;&&space;0&space;\end{bmatrix}" title="\begin{bmatrix} 1 & 0 \end{bmatrix}" /></a> is the Measurement Function, <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;H" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;H" title="H" /></a> . It demonstatres how to map the state to the observation, <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;z" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;z" title="z" /></a>

Formulas for measurement update step
--

*measurement residual*, <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{200}&space;y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{200}&space;y" title="y" /></a> 

This formula represents the difference between the measurement and the expected measurement based on the prediction i.e a comparision of where the measurement tells us we are vs. where we think we are. 

1. <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{200}&space;y&space;=&space;z&space;-&space;Hx'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{200}&space;y&space;=&space;z&space;-&space;Hx'" title="y = z - Hx'" /></a>

*measurement noise* <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{200}&space;R" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{200}&space;R" title="R" /></a>. 

This formula maps the state prediction covariance into the measurement space and adds the measurement noise. 

NOTE: The result "S", will be used in a coming equation to calulate the Kalman Gain

2. <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{200}&space;S&space;=&space;HP'H^T&space;&plus;&space;R" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{200}&space;S&space;=&space;HP'H^T&space;&plus;&space;R" title="S = HP'H^T + R" /></a>

Kalman Gain
---

The Kalman Gain determines how much weight should be placed on the state prediction and how much weight on the measurement update. It is an averaging factor that evaluates which of the measurement update or state prediction should be trusted more based on the measurement noise and the process noise.

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{200}&space;K&space;=&space;P'H^TS^{-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{200}&space;K&space;=&space;P'H^TS^{-1}" title="K = P'H^TS^{-1}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{200}&space;x&space;=&space;x'&space;&plus;&space;Ky" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{200}&space;x&space;=&space;x'&space;&plus;&space;Ky" title="x = x' + Ky" /></a>

The last step in the Kalman Filter is to update the new state's covariance using the Kalman Gain

Covariance calculation: 

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{200}&space;P&space;=(I&space;-&space;KH)P'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{200}&space;P&space;=(I&space;-&space;KH)P'" title="P =(I - KH)P'" /></a>

The Kalman Filter can successfully recover from incaccurate initial estimates, but it is very important to estimte the noise parameters, Q and R as accurately as possible because they are used to determine which of the estimate or the measurement to believe more.



# PROGRAMMING THE MULTI-DIMENSIONAL KALMAN FLITER

We will now code the multi-dimesnional Kalman Filter in C++. The code below uses the C++ ``eigen`` library to define matrices and easily compute their inverse and transpose. Here is a link to the full ``eigen`` library documentation. [click here](https://eigen.tuxfamily.org/dox/group__QuickRefPage.html)


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

  # Extended Kalman Filter
  
  Kalman assumptions 
  ---
  * Motion and measurement models are linear
  * State space can be represebted by a unimodal Gaussian distribution
  
  These assumptions only work for a primitive robot. Not ones that are non-linear and can move in a circle or follow a curve. 
  
  Why can't we use Kalman in non-linear robotics? 
  
  Given a unimodal Gaussian distribution with a mean = (mu) and a variance = (sigma2)
  
  When the distribution undergoes linear transformation (y = mx + b) the posterior distribution is a Gaussian with
  
  1. mean = a x (mu) + b
  2. Variance = a2(sigma2)
  
  This is what can happen in a state prediction .
  
  NOTE: A linear transformation that takes in a Gaussian for an input will have a Gaussian for an ouput
  
  ![alt text][image19]
  
  
  QUESTION
  --
  What happens if the transformation is nonlinear? 
  
  As before the prior belief is a unimodal gaussian distributiion with 
  
  mean = mu 
  variance = sigma2 
  
  Now the function is nonlinear
  
  f(x) = atan(x) 
  
  The resulting graph is not a Gaussian distribution 
  
  ![alt text][image20] 
  
  The distribution cannot be computer in closed form i.e with in a finite number of operations. To model this distribution, thousands of samples must be collected according to the prior distribution and passed through the function f(x) 
  
  Doing this will make the filter more computationally intensive which is not what the filter is designed for. 
  
  If we examine the graph of f(x) more closesly we see that for very short intervals, the function may be approximated by a linear function. 
  
  ![alt text][image21]
  
  The linear estimate is only valid for a small section of the function but if its centered on the best estimate (the mean) and updated with every step, it can produce great results. 
  
  * The mean can be updated by the nonlinear function-->  f(x) 
  * The covariance must be updated by the linearization of the function f(x)
  
  
  To calculate he local linear approximation, use the Taylor series. 
  
  Taylor series - a function can be represented by the sum of an infinite number of terms as represented by the following formula 
  
  ![alt text][image22]
  
  An approximation can be obtained by using just a few terms. 
  
  A linear approximation can be obtained by using the first two terms of the Taylor series 
  
 ![alt text][image23] 
 
 This linear approximation is center around the mean and used to update the covariance matrix of the prior state Gaussian. 
 
 EKF vs KF
 --
 
 1. EKF
 
 either the state transformation function, measurement function or both are nonlinear. These nonlinear functions update the mean but not the variance 
 
 Locally linear approximations are calculate and used to update the variance
 
 Summary
 ---
 ![alt text][image24]
  
  Multi-dimensional Extended Kalman Filter
  ---
   
  When implementing the Extended Kalman Filter, non-linear motion or measurement functions need to be linearized to be able to update the variance
  
  To do the for multiple dimensions, we use a multi-dimensional Taylor Series: 
  
  ![alt text][image25]
  
  Just like in the 1-Dimensional Taylor series, we only need the first 2 terms: 
  
  ![alt text][image26]
  
  The new term, *Df(a) is the Jacobian matrix and it holds the partial derivative terms for the multi-dimensiona equation
  
  ![alt text][image27]
  
  The Jacobian is a matrix of partial derivative that tell us how each of the components of *f changes as we change the components of the state vector. 
  
  ![alt text][image28]
  
  The rows correspond to the dimension of the function, f
  The columns relate to the dimension (state variable) of x
  
  The first element of the matrix is the first dimensionn of the function derived with respect to the first dimension of x. 
  
  The Jacobian is a generalization of the 1D case. In the 1D case, the Jacobian would only have the term df/dx.
  
  Example
  ---
  
  We are tracking the x-y coordinate of an object. The state vecto is x, with the state variable x and y 
  
  x = [x; y;]
  
  Our sensors does not allow us to measure the x and y ccoordinate of the object directly. Our sensor measures the distance from the robot to the object, r, as well as the angle between r and the x-axis, theta.
  
  z = [r; theta]
  
  
  Our state is in Cartesian representation 
  
  Our measurement is in the polar representation 
  
  The measurement function maps the state to the observation, as so, 
  
  ![alt text][image29]
  
  NOTE: Our measurement function must map from Cartesian to Polar coordinates. 
  
  The relationship between Cartesian and polar coordinates is nonlinear, therefore there is no matrix, H, that will successflly make this conversion. 
  
  ![alt text][image30]
  
  Instead of using the measurement residual equation y  =  - Hxprime , the mapping must be made with a dedication function, h(x').
  
  ![alt text][image31] 
  
 
 The measurement residual equation becomes y = z - h(x')
 
 Our measurement covariance matrix cannot be updated the same way because it would turn into a non-Gaussian distribution. 
 
 Now we calculate a linearization, H and use it instead
 
 The Taylor series for the function h(x), centered about the mean mu is defined. 
 
![alt text][image32]


The Jacobian, Df(mu), is defined below. We'll call it H since it is the linearization of measurement function, h(x)

![alt text][image33]

Computing each of the partial derivates, would result in the following matrix 

![alt text][image34]

This matrix, H, can be used to update the state's covariance.

Extended Kalman Filter Equation

![alt text][image35]

Summary
--

![alt text][image36]


  
  
  
  
  # EKF Example 
  
  A quadrotor with motion contsrained to the y-axis:
  
  State Vector
  
  x = [phi; ydot; y]
  
  phi = roll angle 
  ydot = velocity
  y = position 
  
  The quadrotor below is equipped with a ranger finder, so it to knoes the distance between it and the wall. 
  
  
  1[alt text][image37]
  
  
  In the current configuration the expected measurement to the wall would be 
  
  * h(x) = wall - y
  
  Now consider what would happen if the quadrotor were to roll to some angle phi:
  
  ![alt text][image38]
  
  The equation for the measurement when the quadrotor has roll angle of phi is derived from basic trignonometry
  
  
  h(x) = (wall -  y) / cos(phi)
  
  NOTE: The cosine in the denominator makes this function non-linear. Therefore an Extended Kalman Filter is needed to estimate and linearize the function. 
  
  Calculating H
  ---
  
  To apply the Extended Kalman Filter, we need to calculate H, the Jacobian of the measurement model defined above. 
  
  ![alt text][image39] 
  
  
  Calculating the three partial derivatives will result in the following 
  
  ![alt text][image40]
  
  
  After calculating H:
  
  ![alt text][image41]
  
  Now H can be used in the Extended Kalman Filter equations to update the covariance of the state. 
  
  Equations: 
  
  ![alt text][image42]
  
  # Lab: Kalman Filter 
  
