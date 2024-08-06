# [VAPOR: Legged Robot Navigation in Outdoor Vegetation using Offline Reinforcement Learning (ICRA 2024)](https://arxiv.org/pdf/2309.07832.pdf)
Kasun Weerakoon, Adarsh Jagan Sathyamoorthy, Mohamed Elnoor, Dinesh Manocha

## Abstract

We present VAPOR, a novel method for autonomous legged robot navigation in unstructured, densely vegetated outdoor environments using offline Reinforcement Learning (RL). 
Our method trains a novel RL policy using an actor-critic network and arbitrary data collected in real outdoor vegetation. 
Our policy uses height and intensity-based cost maps derived from 3D LiDAR point clouds, a goal cost map, and processed proprioception data as state inputs, 
and learns the physical and geometric properties of the surrounding obstacles such as height, density, and solidity/stiffness. 
The fully-trained policy's critic network is then used to evaluate the quality of dynamically feasible velocities generated from a novel context-aware planner. 
Our planner adapts the robot's velocity space based on the presence of entrapment inducing vegetation, and narrow passages in dense environments. 
We demonstrate our method's capabilities on a Spot robot in complex real-world outdoor scenes, including dense vegetation. 
We observe that VAPOR's actions improve success rates by up to 40%, decrease the average current consumption by up to 2.9%, 
and decrease the normalized trajectory length by up to 11.2% compared to existing end-to-end offline RL and other outdoor navigation methods.

# Video
A video summary and demonstrations of the system can be found [here](https://www.youtube.com/watch?v=toPUHt6Mn8A&t=1s)

# Dependencies

This implementation builds on the Robotic Operating System (ROS-Noetic) and Pytorch. 

# Environment

## 1. Create a Conda Environment

```
conda env create --name vapor --file=environment.yml
conda activate vapor
```
              
## 2. Installing VAPOR
To build from source, clone the latest version from this repository into your catkin workspace and compile the package using,

```
cd catkin_ws/src
git https://github.com/kasunweerkoon/VAPOR.git
catkin_make
```

# Testing

## 1. Run the planner
```
cd catkin_ws/src/VAPOR/testing
python offline_holonomic_planner.py
```
## 2. Publish a goal
```
rostopic pub /target/position geometry_msgs/Twist "linear:
  x: 6.0
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.0"  
```
# Training
```
cd catkin_ws/src/VAPOR/training
python train_offline.py
```
