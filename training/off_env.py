import numpy as np
import math
from math import pi
import random

class Env():
    def __init__(self, is_training,data_path):
    	self.bag_name = 'umd_veg_'
    	self.bag_no = str(10)
    	self.observation_space = np.empty((3, 1))
    	self.odom_goal =[]
    	self.odom_current = []
    	self.tot_dist_to_goal = 0
    	self.init_position_no =0
    	self.state_no=0
    	self.action_space = np.empty((2, 1))
    	self.img_path = data_path+'/img/'
    	self.odom_path = data_path+'/odom/'
    	self.joint_path = data_path+'/joints/'
    	self.goal_reach_threshold = 0.1

    def reset(self):
    	self.init_position_no = np.random.randint(200,4000)
    	self.state_no = self.init_position_no 
    	goal_position_no = self.init_position_no + np.random.randint(180,200)
    	odom_init = np.load(self.odom_path+self.bag_name+self.bag_no+'_odom_'+str(self.init_position_no)+".npy")
    	self.odom_goal = np.load(self.odom_path+self.bag_name+self.bag_no+'_odom_'+str(goal_position_no)+".npy")
    	
    	self.tot_dist_to_goal = math.hypot(self.odom_goal[0] - odom_init[0], self.odom_goal[1] - odom_init[1])

    	state = np.expand_dims(np.array(odom_init),axis=1)
    	print("state dims",state.shape)
    	return state
    	
    def step(self,action):
    	self.odom_current = np.load(self.odom_path+self.bag_name+self.bag_no+'_odom_'+str(self.state_no)+".npy")
    	next_odom = np.load(self.odom_path+self.bag_name+self.bag_no+'_odom_'+str(self.state_no+1)+".npy")
    	self.state_no = self.state_no + 1

    	dist_to_goal = math.hypot(self.odom_goal[0] - self.odom_current[0], self.odom_goal[1] - self.odom_current[1])

    	reward, done = self.set_reward(dist_to_goal)
    	next_state = np.expand_dims(np.array(next_odom),axis=1)

    	return next_state, reward, done

    def set_reward(self,dist_to_goal):
    	done = False
    	if dist_to_goal < self.goal_reach_threshold:
    		reward = 500
    		done = True
    		print(' ------- Goal reached -----------')
    	else:
    		r_dist = (dist_to_goal / self.tot_dist_to_goal) * 100
    		reward = r_dist
    		done = False

    	return reward, done

