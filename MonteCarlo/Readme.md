
[image1]: ./images/mclvsekf.png
[image2]: ./images/compare.png
[image3]: ./images/map.png
[image4]: ./images/pose.png
[image5]: ./images/mcl.png
[image6]: ./images/mclvsekf.png
[image7]: ./images/ekf.png
[image8]: ./images/resample_wheel.png
[image9]: ./images/step0.png
[image10]: ./images/step49.png
[image11]: ./images/
[image12]: ./images



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

Particle filters can be used for localizing autonomous robots. They can equip robots with the tool of probability, allowing them to represent their belief about their current state with random *samples. Furthermore, it can be used to estimate non-Gaussian, nolinear processes. 
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
The solution to this problem is implemented in C++ 

```cpp
#include <iostream>
using namespace std;

int main() {
	
	//Given P(POS), P(DOOR|POS) and P(DOOR|¬POS)
	double a = 0.0002 ; //P(POS) = 0.002
	double b = 0.6    ; //P(DOOR|POS) = 0.6
	double c = 0.05   ; //P(DOOR|¬POS) = 0.05
	
	//TODO: Compute P(¬POS) and P(POS|DOOR)
	double d = 1-a ;                  //P(¬POS)  
	double e =  (b*a)/((a*b)+(d*c)) ; //P(POS|DOOR)
	
	//Print Result
	cout << "P(POS|DOOR)= " <<    e    << endl;
	
	return 0;
}
```
The belief is P(POS|DOOR)= 0.00239473

# The MCL Algorthim

The Monte Carlo Localization Algorithm is comprosed of two main sections that contain for loops:

The first section is the motion and sensor update
The section section is the resampling process.

Given a map of an environment the goal of the MCL is to determine the robot's pose represented by the belief. At each iteration, the algortihm takes the previous belief, the actuation command and the sensor measurements as input.

<a href="https://www.codecogs.com/eqnedit.php?latex=MCL(X_t-1,&space;u_t,&space;z_t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?MCL(X_t-1,&space;u_t,&space;z_t)" title="MCL(X_t-1, u_t, z_t)" /></a>

In the beginning the blief is obtained by randomly generating m particles. The hypothetical state is computed whenever the robot moves.Next, the particles weight is computed after using the latest sensor measurements. Now, motion and measurment are both beign added to the previous state.

Resampling 
Samples with the high probability survive and are re-spawned in the next iteration while the others die. Then the algorithm outputs it belief. Starting the cylce over again from new measurements. 

![alt text][image5]

Orientation matters in the resampling stage since prefiction is diferent for different orientations.



Steps of the MCL algorithm
---

1. Previous Belief
2. Motion Update
3. Measurement Update
4. Resampling 
5. New Belief

MCL vs EKF in Action 

1. MCL 

![alt text][image6] 

At time:

t=1, Particles are drawn randomly and uniformly over the entire pose space.
t=2, Measurement is updated and an importance weight is assigned to each particle.
t=3, Motion is updated and a new particle set with uniform weights and high number of particles around the three most likely places is obtained in resampling.
t=4, Measurement assigns non-uniform weight to the particle set.
t=5, Motion is updated and a new resampling step is about to start.


2. EKF 

![alt text][image7] 

At time:

t=1, Initial belief represented by a Gaussian distribution around the first door.
t=2, Motion is updated and the new belief is represented by a shifted Gaussian of increased weight.
t=3, Measurement is updated and the robot is more certain of its location. The new posterior is represented by a Gaussian with a small variance.
t=4, Motion is updated and the uncertainty increases.


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

Mixtrue-MCL is uniformly superior to regular MCL and particle filters. 

Key disadvantage- a sensor model that permits fast sampling of poses is required! Model is not always able to be trivally obtained. 

Overcoming the disadvantages - sufficient use of statistics and density trees to learn a sampling model from the data. 

Further, during a pre-procesing phase sensor readings are mappd into a set of discriminationg features and potential robot poses are then drawn randomly using tree generated. After the tree is made dual sampling can be done very efficiently. 


# Programming MCL in C++ 

We will program the MCL in the following sections:

1. Motion and sensing
2. Noise 
3. Particle Filters 
4. Importance weight
5. Error
6. Graphing 

# Robot Class 

``` cpp
//#include "src/matplotlibcpp.h"//Graph Library
#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include <stdexcept> // throw errors
#include <random> //C++ 11 Random Numbers

//namespace plt = matplotlibcpp;
using namespace std;

// Landmarks
double landmarks[8][2] = { { 20.0, 20.0 }, { 20.0, 80.0 }, { 20.0, 50.0 },
    { 50.0, 20.0 }, { 50.0, 80.0 }, { 80.0, 80.0 },
    { 80.0, 20.0 }, { 80.0, 50.0 } };

// Map size in meters
double world_size = 100.0;

// Random Generators
random_device rd;
mt19937 gen(rd());

// Global Functions
double mod(double first_term, double second_term);
double gen_real_random();

class Robot {
public:
    Robot()
    {
        // Constructor
        x = gen_real_random() * world_size; // robot's x coordinate
        y = gen_real_random() * world_size; // robot's y coordinate
        orient = gen_real_random() * 2.0 * M_PI; // robot's orientation

        forward_noise = 0.0; //noise of the forward movement
        turn_noise = 0.0; //noise of the turn
        sense_noise = 0.0; //noise of the sensing
    }

    void set(double new_x, double new_y, double new_orient)
    {
        // Set robot new position and orientation
        if (new_x < 0 || new_x >= world_size)
            throw std::invalid_argument("X coordinate out of bound");
        if (new_y < 0 || new_y >= world_size)
            throw std::invalid_argument("Y coordinate out of bound");
        if (new_orient < 0 || new_orient >= 2 * M_PI)
            throw std::invalid_argument("Orientation must be in [0..2pi]");

        x = new_x;
        y = new_y;
        orient = new_orient;
    }

    void set_noise(double new_forward_noise, double new_turn_noise, double new_sense_noise)
    {
        // Simulate noise, often useful in particle filters
        forward_noise = new_forward_noise;
        turn_noise = new_turn_noise;
        sense_noise = new_sense_noise;
    }

    vector<double> sense()
    {
        // Measure the distances from the robot toward the landmarks
        vector<double> z(sizeof(landmarks) / sizeof(landmarks[0]));
        double dist;

        for (int i = 0; i < sizeof(landmarks) / sizeof(landmarks[0]); i++) {
            dist = sqrt(pow((x - landmarks[i][0]), 2) + pow((y - landmarks[i][1]), 2));
            dist += gen_gauss_random(0.0, sense_noise);
            z[i] = dist;
        }
        return z;
    }

    Robot move(double turn, double forward)
    {
        if (forward < 0)
            throw std::invalid_argument("Robot cannot move backward");

        // turn, and add randomness to the turning command
        orient = orient + turn + gen_gauss_random(0.0, turn_noise);
        orient = mod(orient, 2 * M_PI);

        // move, and add randomness to the motion command
        double dist = forward + gen_gauss_random(0.0, forward_noise);
        x = x + (cos(orient) * dist);
        y = y + (sin(orient) * dist);

        // cyclic truncate
        x = mod(x, world_size);
        y = mod(y, world_size);

        // set particle
        Robot res;
        res.set(x, y, orient);
        res.set_noise(forward_noise, turn_noise, sense_noise);

        return res;
    }

    string show_pose()
    {
        // Returns the robot current position and orientation in a string format
        return "[x=" + to_string(x) + " y=" + to_string(y) + " orient=" + to_string(orient) + "]";
    }

    string read_sensors()
    {
        // Returns all the distances from the robot toward the landmarks
        vector<double> z = sense();
        string readings = "[";
        for (int i = 0; i < z.size(); i++) {
            readings += to_string(z[i]) + " ";
        }
        readings[readings.size() - 1] = ']';

        return readings;
    }

    double measurement_prob(vector<double> measurement)
    {
        // Calculates how likely a measurement should be
        double prob = 1.0;
        double dist;

        for (int i = 0; i < sizeof(landmarks) / sizeof(landmarks[0]); i++) {
            dist = sqrt(pow((x - landmarks[i][0]), 2) + pow((y - landmarks[i][1]), 2));
            prob *= gaussian(dist, sense_noise, measurement[i]);
        }

        return prob;
    }

    double x, y, orient; //robot poses
    double forward_noise, turn_noise, sense_noise; //robot noises

private:
    double gen_gauss_random(double mean, double variance)
    {
        // Gaussian random
        normal_distribution<double> gauss_dist(mean, variance);
        return gauss_dist(gen);
    }

    double gaussian(double mu, double sigma, double x)
    {
        // Probability of x for 1-dim Gaussian with mean mu and var. sigma
        return exp(-(pow((mu - x), 2)) / (pow(sigma, 2)) / 2.0) / sqrt(2.0 * M_PI * (pow(sigma, 2)));
    }
};

// Functions
double gen_real_random()
{
    // Generate real random between 0 and 1
    uniform_real_distribution<double> real_dist(0.0, 1.0); //Real
    return real_dist(gen);
}

double mod(double first_term, double second_term)
{
    // Compute the modulus
    return first_term - (second_term)*floor(first_term / (second_term));
}

double evaluation(Robot r, Robot p[], int n)
{
    //Calculate the mean error of the system
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        //the second part is because of world's cyclicity
        double dx = mod((p[i].x - r.x + (world_size / 2.0)), world_size) - (world_size / 2.0);
        double dy = mod((p[i].y - r.y + (world_size / 2.0)), world_size) - (world_size / 2.0);
        double err = sqrt(pow(dx, 2) + pow(dy, 2));
        sum += err;
    }
    return sum / n;
}
double max(double arr[], int n)
{
    // Identify the max element in an array
    double max = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i] > max)
            max = arr[i];
    }
    return max;
}
```
# Interaction 1

We start by learning how to instaniate the robot object from the robot class with a random position and orientation and change its intial positon and orientation. Next, we learn how to print the robot's pose, rotate and move it forward in the 2-D map. Finally we will print the distances from the robot to 8 landmarks. 

1. Instantiating the robot 
2. Change the position and orientation to x=10.0,  y =10.0, theta = 0 
3. Print the position and orientation
4. Rotate the robot by PI/2 and move forward by 10 (use M_PI) for the pi value
5. Print robot's position and orinetaion 
6. Print the distance from the robot toward the eight landmarks. 



``` cpp

int main()
{
    // Instantiating a robot object from the Robot class
    Robot myrobot;

    // Set robot new position to x=10.0, y=10.0 and orientation=0
    // Fill in the position and orientation values in myrobot.set() function
    myrobot.set(10, 10, 0);

    // Printing out the new robot position and orientation
    cout << myrobot.show_pose() << endl;

    // Rotate the robot by PI/2.0 and then move him forward by 10.0
    // Use M_PI for the pi value
    myrobot.move(M_PI/2.0, 10);

    // Print out the new robot position and orientation
    cout << myrobot.show_pose() << endl;

    // Printing the distance from the robot toward the eight landmarks
    cout << myrobot.read_sensors() << endl;

    return 0;
}
```

Motion and Sensing 
---

The next part of our MCL code involves:

1. Instantiating a robot objec
2. Simulate motion update and sensor update 

``` cpp

int main()
{
    // Instantiate a robot object from the Robot class
Robot myRobot;

    // Set robot new position to x=30.0, y=50.0 and orientation=PI/2
myRobot.set(30.0, 50.0, M_PI/2.0);

    // Turn clockwise by PI/2 and move by 15 meters
    
myRobot.move(-M_PI/2.0, 15.0);


    // Print the distance from the robot toward the eight landmarks
cout << myRobot.read_sensors() << endl;

    //  Turn clockwise by PI/2 and move by 10 meters
myRobot.move(-M_PI/2.0,  10.0);

    //  Print the distance from the robot toward the eight landmarks
cout << myRobot.read_sensors() << endl;

    return 0;
}

```

# Noise 

Let's alter the robot's **pose** and **measurement** values to **noisy** ones by adding the following noise values

```Forward Noise ``` = 50

```Turn Noise ``` = 0.1 

```Sense Noise ``` = 5.0 



```cpp

int main()
{
    Robot myrobot;
    // TODO: Simulate Noise
     double Forward_Noise=5.0, Turn_Noise=0.1, Sense_Noise=5.0;
    myrobot.set_noise(Forward_Noise, Turn_Noise, Sense_Noise);
    
    myrobot.set(30.0, 50.0, M_PI / 2.0);
    myrobot.move(-M_PI / 2.0, 15.0);
    cout << myrobot.read_sensors() << endl;
    myrobot.move(-M_PI / 2.0, 10.0);
    cout << myrobot.read_sensors() << endl;

    return 0;
}
```
# Particle Filter 

The most important step is to generate particles.

```cpp
int main()
{
    //Practice Interfacing with Robot Class
    Robot myrobot;
    myrobot.set_noise(5.0, 0.1, 5.0);
    myrobot.set(30.0, 50.0, M_PI / 2.0);
    myrobot.move(-M_PI / 2.0, 15.0);
    //cout << myrobot.read_sensors() << endl;
    myrobot.move(-M_PI / 2.0, 10.0);
    //cout << myrobot.read_sensors() << endl;

    //####   DON'T MODIFY ANYTHING ABOVE HERE! ENTER CODE BELOW ####

    // Instantiating 1000 Particles each with a random position and orientation
    int n = 1000;
    Robot p[n];
    //TODO: Your job is to loop over the set of particles
    for(int i=0; i < n; i++){
        p[i].set_noise(0.05, 0.05, 5.0);// For each particle add noise: Forward_Noise=0.05, Turn_Noise=0.05, and Sense_Noise=5.0

        cout << p[i].show_pose() << endl; //show the pose
    }
    
    
    
    
    return 0;
}


```

Simulating Motion
---

Here we simulate motion  using C++

``` cpp


int main()
{
    //Practice Interfacing with Robot Class
    Robot myrobot;
    myrobot.set_noise(5.0, 0.1, 5.0);
    myrobot.set(30.0, 50.0, M_PI / 2.0);
    myrobot.move(-M_PI / 2.0, 15.0);
    //cout << myrobot.read_sensors() << endl;
    myrobot.move(-M_PI / 2.0, 10.0);
    //cout << myrobot.read_sensors() << endl;

    // Create a set of particles
    int n = 1000;
    Robot p[n];

    for (int i = 0; i < n; i++) {
        p[i].set_noise(0.05, 0.05, 5.0);
        //cout << p[i].show_pose() << endl;
    }

    //####   DON'T MODIFY ANYTHING ABOVE HERE! ENTER CODE BELOW ####

    //Now, simulate motion for each particle
    //TODO: Create a new particle set 'p2'
    int m = 1000;
    Robot p2[m];
    
    //TODO: Rotate each particle by 0.1 and move it forward by 5.0
    
    for (int j = 0; j < m; j++){
        p2[j].move(0.1, 5.0);
        p[j] = p2[j]; //Assign 'p2' to 'p' and print the particle poses, each on a single line
        cout << p[j].show_pose() << endl;
    }
    

    return 0;
}

```

# Importance weight 

So far we have generated particles and simulated motion. Now we will assign an importance weight to each one of the generated particles.

```cpp
int main()
{
    //Practice Interfacing with Robot Class
    Robot myrobot;
    myrobot.set_noise(5.0, 0.1, 5.0);
    myrobot.set(30.0, 50.0, M_PI / 2.0);
    myrobot.move(-M_PI / 2.0, 15.0);
    //cout << myrobot.read_sensors() << endl;
    myrobot.move(-M_PI / 2.0, 10.0);
    //cout << myrobot.read_sensors() << endl;

    // Create a set of particles
    int n = 1000;
    Robot p[n];

    for (int i = 0; i < n; i++) {
        p[i].set_noise(0.05, 0.05, 5.0);
        //cout << p[i].show_pose() << endl;
    }

    //Re-initialize myrobot object and Initialize a measurment vector
    myrobot = Robot();
    vector<double> z;

    //Move the robot and sense the environment afterwards
    myrobot = myrobot.move(0.1, 5.0);
    z = myrobot.sense();

    // Simulate a robot motion for each of these particles
    Robot p2[n];
    for (int i = 0; i < n; i++) {
        p2[i] = p[i].move(0.1, 5.0);
        p[i] = p2[i];
    }

    //####   DON'T MODIFY ANYTHING ABOVE HERE! ENTER CODE BELOW ####

    //TODO: Generate particle weights depending on robot's measurement
    //TODO: Print particle weights, each on a single line
  double w[n];
  
    for (int i = 0; i < n; i++) {
        w[i] = p[i].measurement_prob(z);
        cout << w[i] << endl;
    }

    return 0;
}
```
# Resampling

Thus far, the sensor and motion update was successfully implemented in C++. Now we look into the second major section of the MCL algorithm. The resampling process. Given a set of n particle p1....pN, each particle has:

1. A pose (x1,y1,theta1)...(xN,yN,thetaN)

2. Weights w1....wn - denoting how the close the particle is from the robot position 

3. We can sum all these weights and call the result, W then we computer the normalized weight (alpha) by dividing each weight by the normalizer, W 

<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /></a> is the probability of each particle being selected from the whole. 

The sum of all <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /></a> must equal 1

The resampler regorups all N normalized weights into a bag, Particle 2 and 3 have a large weight . By the end of the resampling process there will be N number of particle as there were originally.

At each iteration the resampling process is more likely to pick particles with large probablilites or alphas to save while the others die. 

``` cpp
#include <iostream>

using namespace std;

double w[] = { 0.6, 1.2, 2.4, 0.6, 1.2 };
double sum = 0;

void ComputeProb(double w[], int n)
{
    for (int i = 0; i < n; i++) {
        sum = sum + w[i];
    }
    for (int j = 0; j < n; j++) {
        w[j] = w[j] / sum;
        cout << "P" << j + 1 << "=" << w[j] << endl;
    }
}

int main()
{
    ComputeProb(w, sizeof(w) / sizeof(w[0]));
    return 0;
}
```

# Resampling Wheel 

We'll look at how particles are picked for the resampling step. In the picture below we are given 8 particles, each with a specific weight. They are initially group in a wheel, where each particle occupies a space depending on it weight.

Psuedo Code

``` 
index = U[1...N]
<a href="https://www.codecogs.com/eqnedit.php?latex=\beta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\beta" title="\beta" /></a> = 0
for i = 1...N
<a href="https://www.codecogs.com/eqnedit.php?latex=\beta&space;=&space;\beta&space;&plus;&space;U&space;[0..2&space;\ast&space;w_{max}]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\beta&space;=&space;\beta&space;&plus;&space;U&space;[0..2&space;\ast&space;w_{max}]" title="\beta = \beta + U [0..2 \ast w_{max}]" /></a>
while <a href="https://www.codecogs.com/eqnedit.php?latex=w_{index}&space;<&space;\beta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_{index}&space;<&space;\beta" title="w_{index} < \beta" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\beta&space;=&space;\beta&space;-&space;w_{index}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\beta&space;=&space;\beta&space;-&space;w_{index}" title="\beta = \beta - w_{index}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=index&space;=&space;index&space;&plus;&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?index&space;=&space;index&space;&plus;&space;1" title="index = index + 1" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=select$&space;p_{index}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?select$&space;p_{index}" title="select$ p_{index}" /></a>

Now we will code the resampling wheel in C++

``` cpp

int main()
{
    //Practice Interfacing with Robot Class
    Robot myrobot;
    myrobot.set_noise(5.0, 0.1, 5.0);
    myrobot.set(30.0, 50.0, M_PI / 2.0);
    myrobot.move(-M_PI / 2.0, 15.0);
    //cout << myrobot.read_sensors() << endl;
    myrobot.move(-M_PI / 2.0, 10.0);
    //cout << myrobot.read_sensors() << endl;

    // Create a set of particles
    int n = 1000;
    Robot p[n];

    for (int i = 0; i < n; i++) {
        p[i].set_noise(0.05, 0.05, 5.0);
        //cout << p[i].show_pose() << endl;
    }

    //Re-initialize myrobot object and Initialize a measurment vector
    myrobot = Robot();
    vector<double> z;

    //Move the robot and sense the environment afterwards
    myrobot = myrobot.move(0.1, 5.0);
    z = myrobot.sense();

    // Simulate a robot motion for each of these particles
    Robot p2[n];
    for (int i = 0; i < n; i++) {
        p2[i] = p[i].move(0.1, 5.0);
        p[i] = p2[i];
    }

    //Generate particle weights depending on robot's measurement
    double w[n];
    for (int i = 0; i < n; i++) {
        w[i] = p[i].measurement_prob(z);
        //cout << w[i] << endl;
    }

    // resampling process

   Robot p3[n];
    int index = gen_real_random() * n;
    //cout << index << endl;
    double beta = 0.0;
    double mw = max(w, n);
    //cout << mw;
    for (int i = 0; i < n; i++) {
        beta += gen_real_random() * 2.0 * mw;
        while (beta > w[index]) {
            beta -= w[index];
            index = mod((index + 1), n);
        }
        p3[i] = p[index];
    }
    for (int k=0; k < n; k++) {
        p[k] = p3[k];
        cout << p[k].show_pose() << endl;
    }

    return 0;
}

```

# Error

Now that we have coded the MCL algorithm, we should make an effort to evaluate the overall quality of the solution. To accomplish this we'll compute the average distance between the particle and the robot. An ideal solution will result in an average distance smaller than a meter. We'll call the evaluation function.

``` cpp

int main()
{
    //Practice Interfacing with Robot Class
    Robot myrobot;
    myrobot.set_noise(5.0, 0.1, 5.0);
    myrobot.set(30.0, 50.0, M_PI / 2.0);
    myrobot.move(-M_PI / 2.0, 15.0);
    //cout << myrobot.read_sensors() << endl;
    myrobot.move(-M_PI / 2.0, 10.0);
    //cout << myrobot.read_sensors() << endl;

    // Create a set of particles
    int n = 1000;
    Robot p[n];

    for (int i = 0; i < n; i++) {
        p[i].set_noise(0.05, 0.05, 5.0);
        //cout << p[i].show_pose() << endl;
    }

    //Re-initialize myrobot object and Initialize a measurment vector
    myrobot = Robot();
    vector<double> z;

    //Iterating 50 times over the set of particles
    int steps = 50;
    for (int t = 0; t < steps; t++) {

        //Move the robot and sense the environment afterwards
        myrobot = myrobot.move(0.1, 5.0);
        z = myrobot.sense();

        // Simulate a robot motion for each of these particles
        Robot p2[n];
        for (int i = 0; i < n; i++) {
            p2[i] = p[i].move(0.1, 5.0);
            p[i] = p2[i];
        }

        //Generate particle weights depending on robot's measurement
        double w[n];
        for (int i = 0; i < n; i++) {
            w[i] = p[i].measurement_prob(z);
            //cout << w[i] << endl;
        }

        //Resample the particles with a sample probability proportional to the importance weight
        Robot p3[n];
        int index = gen_real_random() * n;
        //cout << index << endl;
        double beta = 0.0;
        double mw = max(w, n);
        //cout << mw;
        for (int i = 0; i < n; i++) {
            beta += gen_real_random() * 2.0 * mw;
            while (beta > w[index]) {
                beta -= w[index];
                index = mod((index + 1), n);
            }
            p3[i] = p[index];
        }
        for (int k=0; k < n; k++) {
            p[k] = p3[k];
            //cout << p[k].show_pose() << endl;
        }

        //####   DON'T MODIFY ANYTHING ABOVE HERE! ENTER CODE BELOW ####
        
        // TODO: Evaluate the error by priting it in this form:
       cout << "Step = " << t << ", Evaluation = " << evaluation(myrobot, p, n) << endl;


    } //End of Steps loop
    return 0;
}

```
Each number generated denotes the average distance between the particles and the robot in a world of 100mx100m. Notice how the number starts relatively high and converges to a smaller number after several iterations.

# Graph 

After evaluating we need to graph what we have coded to visualize the results. The ```visualization() ``` function takes in as input:

1. The number of particles
2. Robot's pose
3. Particles at each time step

``` cpp
void visualization(int n, Robot robot, int step, Robot p[], Robot pr[])
{
	//Draw the robot, landmarks, particles and resampled particles on a graph
	
    //Graph Format
    plt::title("MCL, step " + to_string(step));
    plt::xlim(0, 100);
    plt::ylim(0, 100);

    //Draw particles in green
    for (int i = 0; i < n; i++) {
        plt::plot({ p[i].x }, { p[i].y }, "go");
    }

    //Draw resampled particles in yellow
    for (int i = 0; i < n; i++) {
        plt::plot({ pr[i].x }, { pr[i].y }, "yo");
    }

    //Draw landmarks in red
    for (int i = 0; i < sizeof(landmarks) / sizeof(landmarks[0]); i++) {
        plt::plot({ landmarks[i][0] }, { landmarks[i][1] }, "ro");
    }
    
    //Draw robot position in blue
    plt::plot({ robot.x }, { robot.y }, "bo");

	//Save the image and close the plot
    plt::save("./Images/Step" + to_string(step) + ".png");
    plt::clf();
}
```

# MCL LAB

1. Clone Lab from GitHub:

``` BASH
$ cd /home/workspace/
$ git clone https://github.com/udacity/RoboND-MCL-Lab

```

2. Edit ``` main.cpp```:

``` cpp

// TODO: Graph the position of the robot and particles at each step




```

3. Compile the program:

``` bash

$ cd RoboND-MCL-Lab/
$ rm -rf Images/*
$ g++ main.cpp -o app -std=c++11 -I/usr/include/python2.7 -lpython2.7

```

4. Run the program (make sure the ``images`` folder is empty before running)

``` bash

./app

```
The results from this lab are discussed below:

- The Green dots represent particles
- The Yellow dots reprsent resample particles 
- The Red dots represent land marks
- The blue dot is the robots position

The image below is a representation of the system that includes the robot, the particles, the resampled particles, and landmarks. The amount of particles are determined at the start of the localization. As the algorithm progresses, resampling occurs. Killing off particles whose weights are small; leaving only the particles with the heightest weights. What do the weights represent? 

![alt text][image9]


At Step # 49, the algorithm has eliminated most of the uncertain particles. 

![alt text][image10]








