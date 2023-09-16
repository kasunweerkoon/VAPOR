#!/usr/bin/env python3

# Author: Connor McGuile
# Latest authors: Adarsh Jagan Sathyamoorthy, and Kasun Weerakoon
# Feel free to use in any way.


import os
import rospy
import math
import numpy as np
from numpy.lib.stride_tricks import as_strided
from std_msgs.msg import Float32, Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Twist, Point, Pose, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
import sensor_msgs.msg
from sensor_msgs.msg import Image, Imu, CompressedImage, JointState, PointCloud2, LaserScan
from spot_msgs.msg import FootState,FootStateArray
from tf.transformations import euler_from_quaternion
import time
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
import sys
import csv

# Headers for local costmap subscriber
from matplotlib import pyplot as plt
from matplotlib.path import Path
from PIL import Image

import sys
# OpenCV
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

#adding policy scripts folder to sys paths
sys.path.insert(0, './VAPOR/training')
sys.path.insert(0,'/VAPOR/dataset_generation')

import torch
import argparse
import torch.nn as nn
from agent import CQLSAC
from variance_from_proprioception import JointData

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL", help="Run name, default: CQL")
    parser.add_argument("--env", type=str, default="Spot Outdoors", help="default: Spot Outdoors")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes, default: 100")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size, default: 256")
    parser.add_argument("--hidden_size", type=int, default=256, help="")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="")
    parser.add_argument("--temperature", type=float, default=1.0, help="")
    parser.add_argument("--cql_weight", type=float, default=1.0, help="")
    parser.add_argument("--target_action_gap", type=float, default=10, help="")
    parser.add_argument("--with_lagrange", type=int, default=0, help="")
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    parser.add_argument("--eval_every", type=int, default=1, help="")
    
    args = parser.parse_args()
    return args


class Config():
    # Configuration parameters

    def __init__(self):
        
        prop_prior_dir = './VAPOR/dataset_generation'
        JointData(prop_prior_dir) # run the proprioception variance generation code

        # Robot parameters
        self.max_vx = 0.7     # [m/s]
        self.min_vx = -0.6     # [m/s]
        self.max_vy = 0.5     # [m/s]
        self.min_vy = -0.5     # [m/s]
        self.max_accel = 1       # [m/ss]

        self.max_yawrate = 0.05   # [rad/s]
        self.max_dyawrate = 3.2  # [rad/ss]
        
        self.v_reso = 0.30 #0.20              # [m/s]
        self.yawrate_reso = 0.04#0.30 #0.20  # [rad/s]
        
        self.dt = 0.5  # [s]
        self.predict_time = 3.5 #2.0 #1.5  # [s]
        
        # 1===
        self.to_goal_cost_gain = 5.0       # lower = detour
        self.veg_cost_gain = 1.0
        self.speed_cost_gain = 0.1   # 0.1   # lower = faster
        self.obs_cost_gain = 3.2            # lower z= fearless
        
        self.robot_radius = 0.6  # [m]
        self.x = 0.0
        self.y = 0.0
        self.v_x = 0.0
        self.w_z = 0.0
        self.goalX = 0.0006
        self.goalY = 0.0006
        self.th = 0.0
        self.r = rospy.Rate(20)

        self.collision_threshold = 0.3 # [m]

        self.conf_thresh = 0.80

        # Optimal action output
        self.min_u = []

        #goal costmap params
        self.map_resolution =0.25
        self.goalmap_shape = (40, 40)
        self.goalmap_local = np.zeros(self.goalmap_shape, dtype=np.uint8)
        self.to_goal_weight = 0.09

        self.hmap_factor = -100 #-10
        self.gmap_factor = 1 #1.3

        self.costmap_resolution = 0.25
        self.costmap_shape = (40,40)#(200, 200)
        self.heightmap_baselink_inflated = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.intensitymap_baselink_inflated = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.costmap_rgb = cv2.cvtColor(self.intensitymap_baselink_inflated,cv2.COLOR_GRAY2RGB)

        self.prop_variance =  None


        # ROS topic names
        self.img_topic_name = "/camera/color/image_raw/compressed" 
        self.velodyne_topic_name = "/velodyne_points"
        self.odom_topic_name = "/spot/odometry"
        self.joints_topic_name = "/joint_states"
        self.footstate_topic_name = "/spot/status/feet"
        self.scan_topic_name = '/scan'
        self.intensity_map_topic_name = '/lidargrid_i'
        self.height_map_topic_name = '/lidargrid_h'
        self.variance_topic_name = '/prop_value'


        # For on-field visualization
        self.height_pub = rospy.Publisher("/height_map", sensor_msgs.msg.Image, queue_size=10) 
        self.intensity_pub = rospy.Publisher("/intensity_map", sensor_msgs.msg.Image, queue_size=10)  
        self.goalmap_pub = rospy.Publisher("/goal_map", sensor_msgs.msg.Image, queue_size=10) 

        # self.plan_map_pub = rospy.Publisher("/planning_costmap", sensor_msgs.msg.Image, queue_size=10)
        self.viz_pub = rospy.Publisher("/viz_costmap", sensor_msgs.msg.Image, queue_size=1) 
        self.intensity_pub = rospy.Publisher("/intensity_map", sensor_msgs.msg.Image, queue_size=10) 
        self.br = CvBridge()

        self.goal_received = False

        #model params
        self.path = '/Media/offRL/dataset/' 

        self.model_path = './VAPOR/training'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.config = get_config()
        self.batch_size = 8

        # Loading the trained CQL_SAC model
        self.agent = CQLSAC(state_size= 3,
                        action_size= 3,
                        tau=self.config.tau,
                        hidden_size=self.config.hidden_size,
                        learning_rate=self.config.learning_rate,
                        temp=self.config.temperature,
                        with_lagrange=self.config.with_lagrange,
                        cql_weight=self.config.cql_weight,
                        target_action_gap=self.config.target_action_gap,
                        device=self.device)

        self.agent.load_state_dict(torch.load(os.path.join(self.model_path,'trained_models/offrl_r1_attn_'+str(160)+'.pkl')))
        self.agent.eval()

        print("Offline RL Model loaded")


    # Callback for Odometry
    def assignOdomCoords(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation
        (roll,pitch,theta) = euler_from_quaternion ([rot_q.x,rot_q.y,rot_q.z,rot_q.w])
        # (roll,pitch,theta) = euler_from_quaternion ([rot_q.z, -rot_q.x, -rot_q.y, rot_q.w]) # used when lego-loam is used
        
        self.th = theta
        # print("Theta of body wrt odom:", self.th/0.0174533)

        # Get robot's current velocities
        self.v_x = msg.twist.twist.linear.x
        self.w_z = msg.twist.twist.angular.z 
        # print("Robot's current velocities", [self.v_x, self.w_z])



    # Callback for goal from POZYX
    def target_callback(self, data):
        print("---------------Inside Goal Callback------------------------")

        radius = data.linear.x # this will be r
        theta = data.linear.y * 0.0174533 # this will be theta
        print("r and theta:",data.linear.x, data.linear.y)
        
        # Goal wrt robot frame        
        goalX_rob = radius * math.cos(theta)
        goalY_rob = radius * math.sin(theta)

        # Goal wrt odom frame (from where robot started)
        self.goalX =  self.x + goalX_rob*math.cos(self.th) - goalY_rob*math.sin(self.th)
        self.goalY = self.y + goalX_rob*math.sin(self.th) + goalY_rob*math.cos(self.th)

        self.tot_dist_to_goal = radius
        
        # print("Self odom:",self.x, self.y)
        # print("Goals wrt odom frame:", self.goalX, self.goalY)

        # If goal is published as x, y coordinates wrt odom uncomment this
        # self.goalX = data.linear.x
        # self.goalY = data.linear.y

    def proprioception_callback(self,data):
        self.prop_variance = data.data

    def intensity_callback(self,data):
        # print('---- getting intensity data---: ', int(math.sqrt(len(data.data))))
        intensitymap_2d = np.reshape(data.data, (-1, int(math.sqrt(len(data.data)))))
        intensitymap_2d = np.reshape(data.data, (int(math.sqrt(len(data.data))), -1))
        intensitymap_2d = np.rot90(np.fliplr(intensitymap_2d), 1, (1, 0))

        im_image = Image.fromarray(np.uint8(intensitymap_2d))
        yaw_deg = 0 # Now cost map published wrt baselink
        im_baselink_pil = im_image.rotate(-yaw_deg)
        self.intensitymap_baselink = np.array(im_baselink_pil)

        kernel = np.ones((3,3),np.uint8)
        dilated_im = cv2.dilate(self.intensitymap_baselink,kernel,iterations =1)
        self.intensitymap_baselink_inflated = np.array(dilated_im)
        self.intensitymap_baselink_inflated = np.rot90(np.uint8(self.intensitymap_baselink_inflated),2)
        # print('min max intensity data: ', np.min(self.intensitymap_baselink_inflated),np.max(self.intensitymap_baselink_inflated))

        self.intensity_pub.publish(self.br.cv2_to_imgmsg(self.intensitymap_baselink_inflated, encoding="mono8"))

        # return self.intensitymap_baselink_inflated


    def height_callback(self,data):
        # print('---- getting height data---: ', int(math.sqrt(len(data.data))))
        heightmap_2d = np.reshape(data.data, (-1, int(math.sqrt(len(data.data)))))
        heightmap_2d = np.reshape(data.data, (int(math.sqrt(len(data.data))), -1))
        heightmap_2d = np.rot90(np.fliplr(heightmap_2d), 1, (1, 0))

        im_image = Image.fromarray(np.uint8(heightmap_2d))
        yaw_deg = 0 # Now cost map published wrt baselink
        im_baselink_pil = im_image.rotate(-yaw_deg)
        self.heightmap_baselink = np.array(im_baselink_pil)

        kernel = np.ones((1,1),np.uint8)
        dilated_im = cv2.dilate(self.heightmap_baselink,kernel,iterations =1)
        self.heightmap_baselink_inflated = np.array(dilated_im)
        self.heightmap_baselink_inflated = np.rot90(np.uint8(self.heightmap_baselink_inflated),2)

        # print('min max height data: ', np.min(self.heightmap_baselink_inflated),np.max(self.heightmap_baselink_inflated))

        self.height_pub.publish(self.br.cv2_to_imgmsg(self.heightmap_baselink_inflated, encoding="mono8"))

        # return self.heightmap_baselink_inflated

    def generate_goal_map(self, goalmap_local,tot_dist_to_goal):
        goalmap_local = goalmap_local
        # print('shape goalmap local :',np.shape(goalmap_local),goalmap_local.shape[0])
        for l in range(goalmap_local.shape[0]):
            for j in range(goalmap_local.shape[1]):
                goalmap_weighted = self.get_dist_cost(l,j,goalmap_local,tot_dist_to_goal)

        goalmap_current = goalmap_weighted  

        # print('tot dist goal',tot_dist_to_goal)
        # print("MIN MAX GOAL :",np.min(goalmap_current),np.max(goalmap_current))

        # if np.max(goalmap_current) >180:
        #   print("-----------------?????????????_____________________")
        #   r,c = np.where(goalmap_current<=110)
        #   for m in range(len(r)):
        #       goalmap_current[r[m],c[m]] = 220

        return goalmap_current

    def get_dist_cost(self,row,column,goalmap_raw,tot_dist_to_goal):

        distX_from_rob = (20 -row) * 0.25
        distY_from_rob = (20 -column) * 0.25

        odomX =  self.x + distX_from_rob*math.cos(self.th) - distY_from_rob*math.sin(self.th)
        odomY = self.y + distX_from_rob*math.sin(self.th) + distY_from_rob*math.cos(self.th)

        to_goal_distance = math.hypot(self.goalX - odomX, self.goalY - odomY)
        # print("current to goal dist and tot-dist : ",to_goal_distance, tot_dist_to_goal)

        if to_goal_distance < 2 * self.robot_radius:
            goalmap_raw[row,column] = 0
        else:
            goalmap_raw[row,column] = goalmap_raw[row,column] + ((to_goal_distance+0.01)/self.to_goal_weight) #((to_goal_distance)/tot_dist_to_goal)*150 #
        # print("current to goal dist and tot-dist and cost : ",to_goal_distance, tot_dist_to_goal,goalmap_raw[row,column])

        return goalmap_raw  


    def costmap_sum(self):
        costmap_sum = self.costmap_baselink_low + self.costmap_baselink_mid + self.costmap_baselink_high
        
        # self.obs_low_mid_high = np.argwhere(costmap_sum > self.height_thresh) # (returns row, col)

        # New Marking
        # self.obs_low_mid_high = np.argwhere(self.costmap_baselink_high > self.height_thresh)
        self.obs_low_mid_high = np.argwhere(self.intensitymap_baselink_inflated > self.intensity_thresh)

        if(self.obs_low_mid_high.shape[0] != 0):
            self.costmap_rgb = self.tall_obstacle_marker(self.costmap_rgb, self.obs_low_mid_high)
        else:
            pass


    def tall_obstacle_marker(self, rgb_image, centers):
        # Marking centers red = (0, 0, 255), or orange = (0, 150, 255)
        rgb_image[centers[:, 0], centers[:, 1], 0] = 0
        rgb_image[centers[:, 0], centers[:, 1], 1] = 0
        rgb_image[centers[:, 0], centers[:, 1], 2] = 255
        return rgb_image
                


# Model to determine the expected position of the robot after moving along trajectory
def motion_3D(x, u, dt):
    # motion model
    # x = [x(m), y(m), theta(rad), v(m/s), omega(rad/s)]
    x[2] += u[2] * dt  # theta += wz*dt
    x[0] += (u[0] * math.cos(x[2]) - u[1] * math.sin(x[2])) * dt
    x[1] += (u[0] * math.sin(x[2]) + u[1] * math.cos(x[2])) * dt

    x[3] = u[0]
    x[4] = u[1]
    x[5] = u[2]

    return x


# Determine the dynamic window from robot configurations
# Suffix 3D indicates a 3-dimensional velocity space
def calc_dynamic_window_3D(x, config):

    # Dynamic window from robot specification
    Vs = [config.min_vx, config.max_vx, config.min_vy, config.max_vy,
          -config.max_yawrate, config.max_yawrate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_accel * config.dt,
          x[4] + config.max_accel * config.dt,
          x[5] - config.max_dyawrate * config.dt,
          x[5] + config.max_dyawrate * config.dt]

    #  [vx_min, vx_max, vy_min, vy_max, yawrate min, yawrate max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3]),
          max(Vs[4], Vd[4]), min(Vs[5], Vd[5]),]

    # print("Dynamic Window: ", dw)
    return dw


# Calculate a trajectory sampled across a prediction time

def calc_trajectory_3D(xinit, action, config):

    x = np.array(xinit)
    traj = np.array(x)  # many motion models stored per trajectory
    time = 0
    
    while time <= config.predict_time:
        # store each motion model along a trajectory
        x = motion_3D(x, action, config.dt)

        traj = np.vstack((traj, x))
        time += config.dt # next sample

    return traj



# Calculate the optimal action from the feasible action space using the offline RL value function
def calc_final_input_3D(x, u, dw, config):

    xinit = x[:]
    min_cost = 10000.0
    config.min_u = u
    config.min_u[0] = 0.0
    
    yellow = (0, 255, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    orange = (0, 150, 255)

    count = 0
    
    # For all possible vx, vy, wz combinations using the values from dw
    vx_array = np.arange(dw[0], dw[1], config.v_reso) # TODO: Tune v_reso
    vy_array = np.arange(dw[2], dw[3], config.v_reso)
    wz_array = np.arange(dw[4], dw[5], config.yawrate_reso)

    combo_array = np.array(np.meshgrid(vx_array, vy_array, wz_array)).T.reshape(-1, 3)
    action_space_size = np.shape(combo_array)[0]
    # print("Size of the action space :",np.shape(combo_array))

    # GENERATING OBSERVATION SPACE FOR OFFLINE RL BASED VALUE PREDICTIONS

    #height map
    height_map = config.heightmap_baselink_inflated

    #goal map generation
    goalmap_local = np.zeros(config.goalmap_shape, dtype=np.uint8)
    goalmap_current = config.generate_goal_map(goalmap_local,config.tot_dist_to_goal)

    # creating a 3 channel state map stacking intensity, height and goal maps
    current_intensity_map = config.intensitymap_baselink_inflated
    current_height_map = height_map *config.hmap_factor
    current_goal_map = goalmap_current *config.gmap_factor

    current_state_1 =torch.from_numpy((np.dstack((current_intensity_map,current_height_map,current_goal_map))).astype(np.uint8)).T.unsqueeze(0)
    current_state_2 = torch.from_numpy(np.array(config.prop_variance))

    # Clone the current state observations to the size of action space to make state action pairs
    current_states1_cloned = current_state_1.repeat(action_space_size,1,1,1)
    current_states2_cloned = current_state_2.repeat(action_space_size,1)   
    actions = torch.from_numpy(np.array(combo_array))

    # print("Size cloned states and actions :",current_states1_cloned.shape,current_states2_cloned.shape, actions.shape)

    states_1 = current_states1_cloned.to(config.device).float()
    states_2 = current_states2_cloned.to(config.device).float()
    actions = actions.to(config.device).float() 

    t1 = time.time()

    # Offline RL model inferencing

    # Compute Q values for the given state action pairs
    q1 =config.agent.critic1(states_1,states_2, actions)
    q2 =config.agent.critic2(states_1,states_2, actions)

    min_u_position1 = torch.argmin(q1).detach().cpu().numpy()
    min_u_position2 = torch.argmin(q2).detach().cpu().numpy()

    # End to end action prediction
    # policy_out_action = config.agent.get_action(states_1,states_2, eval=True)

    t2 = time.time()
    t_dist = t2-t1

    config.min_u = combo_array[min_u_position1]

    # print("min u positions :",min_u_position1,min_u_position2)
    print("min u action Vx, Vy, Wz :",config.min_u)

    print("Inference rate in s :",t_dist)

    traj = calc_trajectory_3D(xinit, config.min_u, config)
    to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(traj, config)
    
    config.costmap_rgb = draw_traj(config, traj, green)

    config.viz_pub.publish(config.br.cv2_to_imgmsg(config.costmap_rgb, encoding="bgr8"))

    #Publishing goal map for visualization
    config.goalmap_pub.publish(config.br.cv2_to_imgmsg(current_goal_map, encoding="mono8"))

    return config.min_u
    


# Calculate obstacle cost inf: collision, 0:free
def calc_obstacle_cost(traj, ob, config):
    skip_n = 2
    minr = float("inf")

    # Loop through every obstacle in set and calc Pythagorean distance
    # Use robot radius to determine if collision
    for ii in range(0, len(traj[:, 1]), skip_n):
        for i in ob.copy():
            ox = i[0]
            oy = i[1]
            dx = traj[ii, 0] - ox
            dy = traj[ii, 1] - oy

            r = math.sqrt(dx**2 + dy**2)

            if r <= config.robot_radius:
                return float("Inf")  # collision

            if minr >= r:
                minr = r

    return 1.0 / minr


# Calculate goal cost via Pythagorean distance to robot
def calc_to_goal_cost(traj, config):
    
    # If-Statements to determine negative vs positive goal/trajectory position
    # traj[-1,0] is the last predicted X coord position on the trajectory
    if (config.goalX >= 0 and traj[-1,0] < 0):
        dx = config.goalX - traj[-1,0]
    elif (config.goalX < 0 and traj[-1,0] >= 0):
        dx = traj[-1,0] - config.goalX
    else:
        dx = abs(config.goalX - traj[-1,0])
    
    # traj[-1,1] is the last predicted Y coord position on the trajectory
    if (config.goalY >= 0 and traj[-1,1] < 0):
        dy = config.goalY - traj[-1,1]
    elif (config.goalY < 0 and traj[-1,1] >= 0):
        dy = traj[-1,1] - config.goalY
    else:
        dy = abs(config.goalY - traj[-1,1])

    # print("dx, dy", dx, dy)
    cost = math.sqrt(dx**2 + dy**2)
    # print("Cost: ", cost)
    return cost


def draw_traj(config, traj, color):
    traj_array = np.asarray(traj)
    x_odom_list = np.asarray(traj_array[:, 0])
    y_odom_list = np.asarray(traj_array[:, 1])

    # print(x_odom_list.shape)

    x_rob_list, y_rob_list = odom_to_robot(config, x_odom_list, y_odom_list)
    cm_col_list, cm_row_list = robot_to_costmap(config, x_rob_list, y_rob_list)

    costmap_traj_pts = np.array((cm_col_list.astype(int), cm_row_list.astype(int))).T
    # print(costmap_traj_pts) 

    costmap_traj_pts = costmap_traj_pts.reshape((-1, 1, 2))
    config.costmap_rgb = cv2.polylines(config.costmap_rgb, [costmap_traj_pts], False, color, 1)
    
    return config.costmap_rgb



# NOTE: x_odom and y_odom are numpy arrays
def odom_to_robot(config, x_odom, y_odom):
    
    # print(x_odom.shape[0])
    x_rob_odom_list = np.asarray([config.x for i in range(x_odom.shape[0])])
    y_rob_odom_list = np.asarray([config.y for i in range(y_odom.shape[0])])

    x_rob = (x_odom - x_rob_odom_list)*math.cos(config.th) + (y_odom - y_rob_odom_list)*math.sin(config.th)
    y_rob = -(x_odom - x_rob_odom_list)*math.sin(config.th) + (y_odom - y_rob_odom_list)*math.cos(config.th)
    # print("Trajectory end-points wrt robot:", x_rob, y_rob)

    return x_rob, y_rob


def robot_to_costmap(config, x_rob, y_rob):

    costmap_shape_list_0 = [config.costmap_shape[0]/2 for i in range(y_rob.shape[0])]
    costmap_shape_list_1 = [config.costmap_shape[1]/2 for i in range(x_rob.shape[0])]

    y_list = [math.floor(y/config.costmap_resolution) for y in y_rob]
    x_list = [math.floor(x/config.costmap_resolution) for x in x_rob]

    cm_col = np.asarray(costmap_shape_list_0) - np.asarray(y_list)
    cm_row = np.asarray(costmap_shape_list_1) - np.asarray(x_list)
    # print("Costmap coordinates of end-points: ", (int(cm_row), int(cm_col)))

    return cm_col, cm_row


# Begin DWA calculations
def dwa_control(x, u, config):
    # Dynamic Window control

    dw = calc_dynamic_window_3D(x, config)

    u = calc_final_input_3D(x, u, dw, config)

    return u


# Determine whether the robot has reached its goal
def atGoal(config, x):
    # check at goal
    if math.sqrt((x[0] - config.goalX)**2 + (x[1] - config.goalY)**2) <= config.robot_radius:
        return True
    return False


def main():
    print(__file__ + " start!!")
    
    config = Config()

    subOdom = rospy.Subscriber("/spot/odometry", Odometry, config.assignOdomCoords)
    subGoal = rospy.Subscriber('/target/position', Twist, config.target_callback)
    subIntensity = rospy.Subscriber(config.intensity_map_topic_name, OccupancyGrid, config.intensity_callback)
    subHeight = rospy.Subscriber(config.height_map_topic_name, OccupancyGrid, config.height_callback)
    subProprioception = rospy.Subscriber(config.variance_topic_name, Float32MultiArray, config.proprioception_callback)      

    choice = input("Publish actions to /cmd_vel ? 1 or 0 :")
    if(int(choice) == 1):
        pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        print("Publishing to cmd_vel")
    else:
        pub = rospy.Publisher("/dont_publish", Twist, queue_size=1)
        print("Not publishing!")

    speed = Twist()
    
    # initial state [x(m), y(m), theta(rad), vx(m/s), vy(m/s) wz(rad/s)]
    x = np.array([config.x, config.y, config.th, 0.0, 0.0, 0.0])
    
    # initial vx, vy, wz
    u = np.array([0.0, 0.0, 0.0])


    # runs until terminated externally
    while not rospy.is_shutdown():

        config.stuck_status = False

        # Initial
        if config.goalX == 0.0006 and config.goalY == 0.0006:
            print("Waiting for a goal input")
            speed.linear.x = 0.0
            speed.linear.y = 0.0
            speed.angular.z = 0.0
            x = np.array([config.x, config.y, config.th, 0.0, 0.0, 0.0])
        
        # Pursuing but not reached the goal
        elif (atGoal(config,x) == False):  

            # Robot is heading to its goal
            u = dwa_control(x, u, config)

            x[0] = config.x
            x[1] = config.y
            x[2] = config.th
            x[3] = u[0] # vx
            x[4] = u[1] # vy
            x[5] = u[2] # wz
            speed.linear.x = x[3]
            speed.linear.y = x[4]
            speed.angular.z = x[5]

        # If at goal then stay there until new goal published
        else:
            print(" --- Goal reached!! ---")
            speed.linear.x = 0.0
            speed.linear.y = 0.0
            speed.angular.z = 0.0
            x = np.array([config.x, config.y, config.th, 0.0, 0.0, 0.0])
        
        pub.publish(speed)
        config.r.sleep()

    cv2.destroyAllWindows()



if __name__ == '__main__':
    rospy.init_node('offline_holonomic_planner')
    main()