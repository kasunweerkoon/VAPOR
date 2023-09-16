#!/usr/bin/env python
from __future__ import print_function
import os
import roslib
import sys
import time
import math
import numpy as np
import rospy
import cv2
import tf
import struct
import ctypes
import geometry_msgs
from std_msgs.msg import String
import sensor_msgs.msg
from sensor_msgs.msg import Image, Imu, CompressedImage, JointState, PointCloud2, LaserScan
from geometry_msgs.msg import Twist, Point, Pose, PoseStamped
import sensor_msgs.point_cloud2 as pc2
from spot_msgs.msg import FootState,FootStateArray
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from cv_bridge import CvBridge, CvBridgeError
import csv

#adding policy scripts folder to sys paths
sys.path.insert(0, '/home/kasun/offlineRL/CQL/CQL-SAC-w-prop/training')

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from agent import CQLSAC

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.collections as mcoll
from matplotlib import cm
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.interpolate import griddata

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL", help="Run name, default: CQL")
    parser.add_argument("--env", type=str, default="halfcheetah-medium-v2", help="Gym environment name, default: Pendulum-v0")
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

class OffRL_Inference:
	def __init__(self):
		self.bridge = CvBridge()
		self.img_topic_name = "/camera/color/image_raw/compressed" #/camera/color/image_raw
		self.velodyne_topic_name = "/velodyne_points"
		self.odom_topic_name = "/spot/odometry"
		self.joints_topic_name = "/joint_states"
		self.footstate_topic_name = "/spot/status/feet"
		self.scan_topic_name = '/scan'
		self.intensity_map_topic_name = '/lidargrid_i'
		self.height_map_topic_name = '/lidargrid_h'


		# Topic names
		self.action_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
		self.height_pub = rospy.Publisher("/height_map", sensor_msgs.msg.Image, queue_size=10) 
		self.intensity_pub = rospy.Publisher("/intensity_map", sensor_msgs.msg.Image, queue_size=10)  
		# self.image_sub = rospy.Subscriber(self.img_topic_name, CompressedImage, self.img_callback)
		# self.footstate_sub = rospy.Subscriber(self.footstate_topic_name, FootStateArray, self.footstate_callback)
		self.odom_sub = rospy.Subscriber(self.odom_topic_name, Odometry, self.odometry_callback)
		self.goal_sub = rospy.Subscriber('/target/position', Twist, self.goal_callback)
		# self.velodyne_sub = rospy.Subscriber(self.velodyne_topic_name, PointCloud2, self.velodyne_callback)
		# self.joints_sub = rospy.Subscriber(self.joints_topic_name , JointState, self.joints_callback)


		self.yaw =0
		self.goalX_odom = 0
		self.goalY_odom = 0
		self.odom_goal =[]

		self.current_vel = [0,0,0]
		self.v_x =0
		self.v_y = 0
		self.w_z = 0 

		# Input velocities
		self.iter = 0
		self.t1 = 0.0
		self.vels = [] #np.empty([1, 2])

		self.vel_diff = []
		self.yaw_rate_diff = []
		self.yaw =0

		self.predict_time = 3  # [s]
		self.time_reso = 0.02
		self.camera_tilt_angle = -20
		self.white_img = np.zeros([720,1280],dtype=np.uint8)
		self.rows,self.cols = None, None

		self.pointcloud = None

		self.joint_data_array =[]
		self.current_position_xyz = Pose()

		self.current_odom =[]
		self.current_joint =[]

		#goal costmap params
		self.map_resolution =0.25
		self.goalmap_shape = (40, 40)
		self.goalmap_local = np.zeros(self.goalmap_shape, dtype=np.uint8)
		self.to_goal_weight = 0.09

		self.hmap_factor = -100 #-10
		self.gmap_factor = 1 #1.3

		self.costmap_shape = (40,40)#(200, 200)
		self.heightmap_baselink_inflated = np.zeros(self.costmap_shape, dtype=np.uint8)
		self.intensitymap_baselink_inflated = np.zeros(self.costmap_shape, dtype=np.uint8)
		self.goal_reach_threshold = 0.1

		self.observations = []

		self.pub_action = Twist()

		self.goal_received = False

		choice = input("Publish end to end actions to /cmd_vel? 0 or 1 :")
		self.pub_end2end_action = int(choice) #False # Stop publishing end to end actions to /cmd_vel

		#model params
		self.path = '/media/kasun/Media/offRL/dataset/' #'/home/kasun/offlineRL/dataset/'

		self.model_path = '/home/kasun/offlineRL/CQL/CQL-SAC/'
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

		self.agent.load_state_dict(torch.load(os.path.join(self.model_path,'trained_models/offrl_'+str(214)+'.pkl')))
		self.agent.eval()

		print("Offline RL Model loaded")





	def goal_callback(self, data):
		odom_data = rospy.wait_for_message(self.odom_topic_name, Odometry, timeout=5)
		current_position_xyz = odom_data.pose.pose.position
		orientation = odom_data.pose.pose.orientation
		q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
		roll, pitch, yaw = self.euler_from_quaternion(q_x, q_y, q_z, q_w) # roll, pitch , yaw in radians

		radius = data.linear.x # this will be r
		theta = data.linear.y * 0.0174533# this will be theta

		goalX_rob = radius * math.cos(theta)
		goalY_rob = radius * math.sin(theta)

		self.tot_dist_to_goal = math.hypot(goalX_rob,goalY_rob)

		self.goalX_odom =  current_position_xyz.x + goalX_rob*math.cos(yaw) - goalY_rob*math.sin(yaw)
		self.goalY_odom = current_position_xyz.y + goalX_rob*math.sin(yaw) + goalY_rob*math.cos(yaw)

		self.odom_goal = [self.goalX_odom,self.goalY_odom]
		print(' goal odometery: ',self.odom_goal)

		self.goal_received = True


	def odometry_callback(self,odom_data):
		self.current_position_xyz = odom_data.pose.pose.position
		orientation = odom_data.pose.pose.orientation
		q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
		roll, pitch, self.yaw_rad = self.euler_from_quaternion(q_x, q_y, q_z, q_w) # roll, pitch , yaw in radians

		# Get robot's current velocities
		self.v_x = odom_data.twist.twist.linear.x
		self.v_y = odom_data.twist.twist.linear.y
		self.w_z = odom_data.twist.twist.angular.z 

		current_odom = [self.current_position_xyz.x,self.current_position_xyz.y,self.yaw_rad] #robot's current position [x,y,yaw angle] w.r.t. odom 
		self.current_vel = [self.v_x,self.v_y,self.w_z]

		#Obtain joint data
		joint_data= rospy.wait_for_message(self.joints_topic_name, JointState, timeout=None)
		self.current_joint = self.joints_callback(joint_data)

		#Obtain intensity and height map data
		intensity_data = rospy.wait_for_message(self.intensity_map_topic_name, OccupancyGrid, timeout=5)
		intensity_map = self.getIntensityMap(intensity_data)

		height_data = rospy.wait_for_message(self.height_map_topic_name, OccupancyGrid, timeout=5)
		height_map = self.getHeightMap(height_data)

		# self.observations = [intensity_map,height_map, current_odom,self.current_joint,np.array(self.current_vel)]
		# print("Obs shape :",np.shape(self.observations))

		if self.goal_received:
			#goal map generation
			goalmap_local = np.zeros(self.goalmap_shape, dtype=np.uint8)
			goalmap_current = self.generate_goal_map(goalmap_local,current_odom,self.odom_goal,self.tot_dist_to_goal)

			# creating a 3 channel state map stacking intensity, height and goal maps
			current_intensity_map = intensity_map
			current_height_map = height_map *self.hmap_factor
			current_goal_map = goalmap_current *self.gmap_factor

			current_state = torch.from_numpy((np.dstack((current_intensity_map,current_height_map,current_goal_map ))).astype(np.uint8)).T.unsqueeze(0)
			current_states_cloned = current_state.repeat(self.batch_size,1,1,1)

			actions = torch.from_numpy(np.array(self.current_vel)).T.unsqueeze(0)
			actions_cloned = actions.repeat(self.batch_size,1)

			states = current_states_cloned.to(self.device).float()
			actions = actions_cloned.to(self.device).float() 

			# print("state action shape :",states.shape,actions.shape)

			t1 = time.time()

			# Compute Q values for the given state action pairs
			q1 = self.agent.critic1(states, actions)
			q2 = self.agent.critic2(states, actions)

			policy_out_action = self.agent.get_action(states, eval=True)

			if self.pub_end2end_action == 1:
				self.pub_action.linear.x = policy_out_action[0]
				self.pub_action.linear.y = policy_out_action[1]
				self.pub_action.angular.z = policy_out_action[2]

				# publishing cmd vel
				self.action_pub.publish(self.pub_action)

			t2 = time.time()
			t_dist = t2-t1
			# print('Inference time :',t_dist)
			# print('Critic outputs :',q1,q2)
			# print('Actor action outputs :',policy_out_action)

			cv2.imshow("Current Intensity Map", cv2.resize(current_intensity_map, (400,400), interpolation = cv2.INTER_AREA))
			cv2.imshow("Current Height Map", cv2.resize(current_height_map, (400,400), interpolation = cv2.INTER_AREA))
			cv2.imshow("Current Goal Map", cv2.resize(goalmap_current, (400,400), interpolation = cv2.INTER_AREA))
			cv2.waitKey(10)



	def joints_callback(self,joint_data):
		# # The state of each joint (revolute or prismatic) is defined by:
		#  * the position of the joint (rad or m),
		#  * the velocity of the joint (rad/s or m/s) and 
		#  * the effort that is applied in the joint (Nm or N).
		
		data_vec = [joint_data.position[0:11],joint_data.velocity[0:11],joint_data.effort[0:11]]
		# print('Data vector: ',data_vec[2])

		# if self.write_data:
		# 	self.joint_data_array.append(data_vec)

		return data_vec
		# pass

	def footstate_callback(self,foot_data):
		# contact: Is the foot in contact with the ground? 
		# 1 --> The foot is currently in contact with the ground.  2 --> The foot is not in contact with the ground.

		# foot_positions: The foot position described relative to the body. Required positions for spot: "fl", "fr", "hl", "hr".
		## foot_data.states[i].foot_position_rt_body.x --> i = 0,1,2,3 (i.e length 4 array for 4 foots)
		## foot_data.states[i].contact --> --> i = 0,1,2,3 (i.e length 4 array)

		# print(foot_data.states[3].foot_position_rt_body.x)
		# print(foot_data.states[3].contact)
		pass

	def velodyne_callback(self,velodyne_data):

		xyz = np.array([[0,0,0]])
		rgb = np.array([[0,0,0]])
		#self.lock.acquire()
		gen = pc2.read_points(velodyne_data, skip_nans=True)
		int_data = list(gen)
		# print(np.shape(int_data))

		for x in int_data:
			test = x[3] 
			# cast float32 to int so that bitwise operations are possible
			s = struct.pack('>f' ,test)
			i = struct.unpack('>l',s)[0]
			# you can get back the float value by the inverse operations
			pack = ctypes.c_uint32(i).value
			r = (pack & 0x00FF0000)>> 16
			g = (pack & 0x0000FF00)>> 8
			b = (pack & 0x000000FF)
			# prints r,g,b values in the 0-255 range
			            # x,y,z can be retrieved from the x[0],x[1],x[2]
			xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)
			rgb = np.append(rgb,[[r,g,b]], axis = 0)

		# np.save(bag_name+'/pointcloud/'+str(self.iter-1)+'.npy',xyz)
		self.pointcloud = xyz
		# return xyz
			
		# print(np.shape(xyz))


	def getIntensityMap(self,data):
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

		self.intensity_pub.publish(self.bridge.cv2_to_imgmsg(self.intensitymap_baselink_inflated, encoding="mono8"))

		return self.intensitymap_baselink_inflated


	def getHeightMap(self,data):
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

		self.height_pub.publish(self.bridge.cv2_to_imgmsg(self.heightmap_baselink_inflated, encoding="mono8"))

		return self.heightmap_baselink_inflated

	def generate_goal_map(self, goalmap_local,odom_current,odom_goal,tot_dist_to_goal):
		goalmap_local = goalmap_local
		# print('shape goalmap local :',np.shape(goalmap_local),goalmap_local.shape[0])
		for l in range(goalmap_local.shape[0]):
			for j in range(goalmap_local.shape[1]):
			    goalmap_weighted = self.get_dist_cost(l,j,goalmap_local,odom_current,odom_goal,tot_dist_to_goal)

		goalmap_current = goalmap_weighted	

		# print('tot dist goal',tot_dist_to_goal)
		# print("MIN MAX GOAL :",np.min(goalmap_current),np.max(goalmap_current))

		# if np.max(goalmap_current) >180:
		# 	print("-----------------?????????????_____________________")
		# 	r,c = np.where(goalmap_current<=110)
		# 	for m in range(len(r)):
		# 		goalmap_current[r[m],c[m]] = 220

		return goalmap_current


	def get_dist_cost(self,row,column,goalmap_raw,odom_current,odom_goal,tot_dist_to_goal):

		distX_from_rob = (20 -row) * 0.25
		distY_from_rob = (20 -column) * 0.25

		odomX =  odom_current[0] + distX_from_rob*math.cos(odom_current[2]) - distY_from_rob*math.sin(odom_current[2])
		odomY = odom_current[1] + distX_from_rob*math.sin(odom_current[2]) + distY_from_rob*math.cos(odom_current[2])

		to_goal_distance = math.hypot(odom_goal[0] - odomX, odom_goal[1] - odomY)
		# print("current to goal dist and tot-dist : ",to_goal_distance, tot_dist_to_goal)

		if to_goal_distance < 2 * self.goal_reach_threshold:
			goalmap_raw[row,column] = 0
		else:
			goalmap_raw[row,column] = goalmap_raw[row,column] + ((to_goal_distance+0.01)/self.to_goal_weight) #((to_goal_distance)/tot_dist_to_goal)*150 #
		# print("current to goal dist and tot-dist and cost : ",to_goal_distance, tot_dist_to_goal,goalmap_raw[row,column])

		return goalmap_raw
		
	

	def euler_from_quaternion(self, x, y, z, w):
	    """
	    Convert a quaternion into euler angles (roll, pitch, yaw)
	    roll is rotation around x in radians (counterclockwise)
	    pitch is rotation around y in radians  (counterclockwise)
	    yaw is rotation around z in radians  (counterclockwise)
	    """
	    t0 = +2.0 * (w * x + y * z)
	    t1 = +1.0 - 2.0 * (x * x + y * y)
	    roll_x = math.atan2(t0, t1)

	    t2 = +2.0 * (w * y - z * x)
	    t2 = +1.0 if t2 > +1.0 else t2
	    t2 = -1.0 if t2 < -1.0 else t2
	    pitch_y = math.asin(t2)

	    t3 = +2.0 * (w * z + x * y)
	    t4 = +1.0 - 2.0 * (y * y + z * z)
	    yaw_z = math.atan2(t3, t4)

	    return roll_x, pitch_y, yaw_z # in radians



def main(args):
	ic = OffRL_Inference()
	rospy.init_node('offlineRL_inference', anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)