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
from std_msgs.msg import String, Float32MultiArray
import sensor_msgs.msg
from sensor_msgs.msg import Image, Imu, CompressedImage, JointState, PointCloud2, LaserScan
from geometry_msgs.msg import Twist, Point, Pose, PoseStamped
import sensor_msgs.point_cloud2 as pc2
from spot_msgs.msg import FootState,FootStateArray, BatteryStateArray
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from cv_bridge import CvBridge, CvBridgeError
import csv

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# from GraPE import GraPENet
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.collections as mcoll
from matplotlib import cm
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.interpolate import griddata



class Data_Subscriber:
	def __init__(self):
		self.bridge = CvBridge()
		self.img_topic_name = "/camera/color/image_raw/compressed" 
		self.velodyne_topic_name = "/velodyne_points"
		self.odom_topic_name = "/spot/odometry"
		self.joints_topic_name = "/joint_states"
		self.footstate_topic_name = "/spot/status/feet"
		self.variance_topic_name = '/prop_value'
		self.scan_topic_name = '/scan'
		self.intensity_map_topic_name = '/lidargrid_i'
		self.height_map_topic_name = '/lidargrid_h'

		self.write_data = True
		self.bag_num = '_1'
		self.bag_name = 'sep14_4'



		# Topic names
		self.height_pub = rospy.Publisher("/height_map", sensor_msgs.msg.Image, queue_size=10) 
		self.intensity_pub = rospy.Publisher("/intensity_map", sensor_msgs.msg.Image, queue_size=10)  
		self.image_sub = rospy.Subscriber(self.img_topic_name, CompressedImage, self.img_callback)


		# Input and Label Parameters
		self.vel_vector_len = 50

		# Wheel Odometry Attributes
		self.odom_flag = False
		self.odom_t1 = 0.0
		self.odom_t2 = 0.0
		self.odom_prev_x = 0.0
		self.odom_prev_y = 0.0
		self.odom_curr_x = 0.0
		self.odom_curr_y = 0.0 
		self.odom_dist = 0.0
		self.odom_vel = 0.0
		self.odom_prev_yaw = 0.0
		self.odom_curr_yaw = 0.0
		self.odom_yaw_rate = 0.0

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

		self.costmap_shape = (40,40)#(200, 200)
		self.heightmap_baselink_inflated = np.zeros(self.costmap_shape, dtype=np.uint8)
		self.intensitymap_baselink_inflated = np.zeros(self.costmap_shape, dtype=np.uint8)

		self.observations = []
		self.variance_msg = Float32MultiArray()

		self.tot_battery_current = []

		#goal costmap params
		# self.map_resolution =0.05
		self.to_goal_weight = 0.2

		#model

		self.transform = transforms.Compose([transforms.Resize(280),
		                    transforms.CenterCrop(280),
		                    transforms.ToTensor()])

		self.path = '/Media/offRL/dataset_w_prop/' 
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


		print("Data Subscriber Started!")

		self.figure, self.ax = plt.subplots(figsize=(10, 8))

	def img_callback(self,img_data):
		try:
			cv_image  = self.bridge.compressed_imgmsg_to_cv2(img_data) # for compressed images

		except CvBridgeError as e:
			print(e)


		(self.rows,self.cols,channels) = cv_image.shape


		# Obtain velocity vector for this instant and append on to a vector
		odom_data = rospy.wait_for_message(self.odom_topic_name, Odometry, timeout=None)
		self.current_position_xyz = odom_data.pose.pose.position
		orientation = odom_data.pose.pose.orientation
		q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
		roll, pitch, yaw = self.euler_from_quaternion(q_x, q_y, q_z, q_w) # roll, pitch , yaw in radians

		# Get robot's current velocities
		self.v_x = odom_data.twist.twist.linear.x
		self.v_y = odom_data.twist.twist.linear.y
		self.w_z = odom_data.twist.twist.angular.z 

		self.current_odom = [self.current_position_xyz.x,self.current_position_xyz.y,yaw] #robot's current position [x,y,yaw angle] w.r.t. odom 
		self.current_vel = [self.v_x,self.v_y,self.w_z]

		#Obtain joint data
		joint_data= rospy.wait_for_message(self.joints_topic_name, JointState, timeout=None)
		self.current_joint = self.joints_callback(joint_data)

		#Obtain intensity and height map data
		intensity_data = rospy.wait_for_message(self.intensity_map_topic_name, OccupancyGrid, timeout=5)
		intensity_map = self.getIntensityMap(intensity_data)

		height_data = rospy.wait_for_message(self.height_map_topic_name, OccupancyGrid, timeout=5)
		height_map = self.getHeightMap(height_data)

		#obtain variance from proprioception
		prop_data = rospy.wait_for_message(self.variance_topic_name, Float32MultiArray, timeout=5)
		prop_variance = prop_data.data
		# print("prop variance", prop_variance)

		# Record Battery Data
		battery_data = rospy.wait_for_message('/spot/status/battery_states', BatteryStateArray, timeout=None)
		battery_current = battery_data.battery_states[0].current * -1
		# print("battery current", battery_current)

		self.tot_battery_current.append(battery_current)

		scan_data= rospy.wait_for_message(self.scan_topic_name, LaserScan, timeout=None)
		scan_range = []
		for i in range(len(scan_data.ranges)):
		    if scan_data.ranges[i] == float('Inf'):
		        scan_range.append(20)
		    elif np.isnan(scan_data.ranges[i]):
		        scan_range.append(0)
		    else:
		        scan_range.append(scan_data.ranges[i])		

		# velodyne_data= rospy.wait_for_message(self.velodyne_topic_name, PointCloud2, timeout=None)

		# print("Received odometry message")
		self.iter = self.iter + 1
		self.vels.append([odom_data.twist.twist.linear.x, odom_data.twist.twist.angular.z])
		# print("Robot vels: ",[odom_data.twist.twist.linear.x,odom_data.twist.twist.linear.y, odom_data.twist.twist.angular.z])

		x_d = np.linalg.norm([odom_data.twist.twist.linear.x,odom_data.twist.twist.linear.y])
		theta_d = odom_data.twist.twist.angular.z

		self.observations = [np.array(cv_image),intensity_map,height_map, self.current_odom,self.current_joint,prop_variance,battery_current, np.array(self.current_vel)]
		# print("Obs shape :",np.shape(self.observations))
		if self.write_data:

			np.save(self.path+self.bag_name+'_'+str(self.iter-1)+'.npy',np.array(self.observations))


		# print("Velocity vector",np.shape(self.vels))
		# cv2.imshow("Image window", height_map)
		# cv2.waitKey(1)



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

		kernel = np.ones((1,1),np.uint8)
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
	ic = Data_Subscriber()
	rospy.init_node('data_subscriber', anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)