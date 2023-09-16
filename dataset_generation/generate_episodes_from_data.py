from torchvision import datasets, transforms
import os
import cv2
import numpy as np
import math
from math import pi
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ImportData(Dataset):
	def __init__(self):
		# self.odom_goal =[]
		# self.odom_current = []
		self.tot_dist_to_goal = 0
		self.init_position_no =0
		self.state_no=0
		self.goal_reach_threshold = 0.7
		self.solid_obs_intensity_th = 245

		self.dist_threshold = 2
		
		self.path = '/VAPOR/scenarios_samples/' 
		self.folder_name = '1/'#'sep10v2/' #'ap25/' #'aug27/'#'aug21/'  #
		self.file_name = 'sep14_'#'umd_sep10_' #'umd_ap25_' #'umd_aug27_' #'umd_aug21_' #
		self.bag_no = str(1)
		self.eps_folder_name = 'episodes_wo_prop_r1/'
		self.eps_folder_name2 = 'episodes_with_prop_r1/'
		self.eps_folder_name3 = 'episodes_with_prop_r2/'
		self.eps_folder_name4 = 'episodes_with_prop_r3/'
		self.eps_folder_name5 = 'episodes_with_prop_r4/'
		self.eps_folder_name6 = 'episodes_with_prop_r5/'

		self.map_resolution =0.25
		self.goalmap_shape = (40, 40)
		self.goalmap_local = np.zeros(self.goalmap_shape, dtype=np.uint8)
		self.to_goal_weight = 0.09

		self.hmap_factor = -10
		self.gmap_factor = 0.6

		self.dataset = []

		self.load_data()


	def load_data(self):

		dirc = self.path+self.folder_name
		lst = os.listdir(dirc) # your directory path
		num_samples= len(lst)

		# num_samples = 4427 #bag 10 for now


		self.episode_done = False

		self.ep_no = 1
		tmp_path = self.path+self.folder_name+self.file_name+self.bag_no+'_'

		for k in range(1):
			
			self.init_position_no = 0 #np.random.randint(0,1400)
			
			# print(tmp_path)
			obs_init = np.load(tmp_path+str(self.init_position_no)+'.npy',allow_pickle = True)
			odom_init = obs_init[3]
			print("Initial odom :",odom_init)
			tmp_goal_position_no = self.init_position_no + 50 #100 #(num_samples -2) #np.random.randint(70,100)

			for fname in range(num_samples):
				
				obs_goal = np.load(self.path+self.folder_name+self.file_name+self.bag_no+'_'+str(tmp_goal_position_no)+".npy",allow_pickle = True)
				odom_goal = obs_goal[3]
				print("tmp goal odom :",odom_goal)

				self.tot_dist_to_goal = math.hypot(odom_goal[0] - odom_init[0], odom_goal[1] - odom_init[1])

				if self.tot_dist_to_goal > self.dist_threshold and not(self.episode_done):
					goal_position_no = tmp_goal_position_no
					self.state_no = self.init_position_no
					eps_length = goal_position_no - self.init_position_no
					print("-- episode goal selected -- episode length: ", eps_length)
					print("-- tot dist to goal -- : ", self.tot_dist_to_goal)

					for i in range(eps_length):
						# print("inside loop episode",i)

						##obs = [img, intensity_map, height_map, odom, joints, proprioception_variances, battery_current, vels]
						##state = [intensity_map, height_map, odom, joints, goal_map

						obs_current = np.load(self.path+self.folder_name+self.file_name+self.bag_no+'_'+str(self.state_no)+".npy",allow_pickle = True)
						obs_next = np.load(self.path+self.folder_name+self.file_name+self.bag_no+'_'+str(self.state_no+1)+".npy",allow_pickle = True)

						odom_current = obs_current[3]
						odom_next = obs_next[3]

						# print("odom current and next and goal :",odom_current, odom_next,odom_goal)

						#goal map generation
						goalmap_local = np.zeros(self.goalmap_shape, dtype=np.uint64)
						goalmap_current = self.generate_goal_map(goalmap_local,odom_current,odom_goal,self.tot_dist_to_goal)

						current_state_list = obs_current[1:-1].tolist()
						current_state_list.append(goalmap_current)

						# creating a 3 channel state map stacking intensity, height and goal maps
						current_intensity_map = current_state_list[0]
						current_height_map = current_state_list[1]*self.hmap_factor
						current_goal_map = current_state_list[6]*self.gmap_factor

						
						# current_state = (np.dstack((current_intensity_map,current_height_map,current_goal_map )) * 255.999).astype(np.uint8) #np.array(current_state_list)
						current_state_1 =(np.dstack((current_intensity_map,current_height_map,current_goal_map ))).astype(np.uint8)
						current_state_2 = np.array(current_state_list[4])
						
						# print("min max AFTER :",np.min(current_state[:,:,0]),np.max(current_state[:,:,0]),np.min(current_state[:,:,1]),np.max(current_state[:,:,1]),np.min(current_state[:,:,2]),np.max(current_state[:,:,2]))						

						action = obs_current[-1]

						# next state
						#goal map generation
						goalmap_local2 = np.zeros(self.goalmap_shape, dtype=np.uint64)
						goalmap_next = self.generate_goal_map(goalmap_local2,odom_next,odom_goal,self.tot_dist_to_goal)
						# cv2.imshow("Next state goal", cv2.resize(goalmap_next,(400,400), interpolation = cv2.INTER_AREA))

						next_state_list  = obs_next[1:-1].tolist()
						next_state_list.append(goalmap_next)

						# creating a 3 channel state map stacking intensity, height and goal maps
						next_intensity_map = next_state_list[0]
						next_height_map = next_state_list[1]*self.hmap_factor
						next_goal_map = next_state_list[6]*self.gmap_factor						
						next_state_1 = (np.dstack((next_intensity_map,next_height_map,next_goal_map ))).astype(np.uint8)  
						next_state_2 =	np.array(next_state_list[4])
						# print("next state :",np.shape(next_state))

						dist_to_goal = math.hypot(odom_goal[0] - odom_current[0], odom_goal[1] - odom_current[1])
						# print("current dist to goal :",dist_to_goal)`

						# Saving samples with multiple different rewards

						reward, done = self.set_reward(dist_to_goal,current_state_1,current_intensity_map,current_state_list[5])

						reward2, done2 = self.set_reward_v2(dist_to_goal,current_state_1,current_intensity_map,current_state_list[5])

						reward3, done3 = self.set_reward_v3(dist_to_goal,current_state_1,current_intensity_map,current_state_list[5])

						reward4, done4 = self.set_reward_v4(dist_to_goal,current_state_1,current_intensity_map,current_state_list[5])

						reward5, done5 = self.set_reward_v5(dist_to_goal,current_state_1,current_intensity_map,current_state_list[5])

						reward6, done6 = self.set_reward_v6(dist_to_goal,current_state_1,current_intensity_map,current_state_list[5])
						# print("current reward :",reward)							

						self.state_no = self.state_no + 1

						ep_sample1 = [current_state_1,current_state_2, action, reward, next_state_1, next_state_2,done]
						ep_sample2 = [current_state_1,current_state_2, action, reward2, next_state_1, next_state_2, done2]
						ep_sample3 = [current_state_1,current_state_2, action, reward3, next_state_1, next_state_2, done3]

						ep_sample4 = [current_state_1,current_state_2, action, reward4, next_state_1, next_state_2, done4]

						ep_sample5 = [current_state_1,current_state_2, action, reward5, next_state_1, next_state_2, done5]

						ep_sample6 = [current_state_1,current_state_2, action, reward6, next_state_1, next_state_2, done6]
						# print("current action :",action)

						np.save(self.path+self.eps_folder_name+'ep'+str(self.ep_no)+'_sample'+str(i)+'.npy',np.array(ep_sample1))
						np.save(self.path+self.eps_folder_name2+'ep'+str(self.ep_no)+'_sample'+str(i)+'.npy',np.array(ep_sample2))
						np.save(self.path+self.eps_folder_name3+'ep'+str(self.ep_no)+'_sample'+str(i)+'.npy',np.array(ep_sample3))
						np.save(self.path+self.eps_folder_name4+'ep'+str(self.ep_no)+'_sample'+str(i)+'.npy',np.array(ep_sample4))
						np.save(self.path+self.eps_folder_name5+'ep'+str(self.ep_no)+'_sample'+str(i)+'.npy',np.array(ep_sample5))
						np.save(self.path+self.eps_folder_name6+'ep'+str(self.ep_no)+'_sample'+str(i)+'.npy',np.array(ep_sample6))


						cv2.imshow("Current Intensity Map", cv2.resize(current_state_1[:,:,0], (400,400), interpolation = cv2.INTER_AREA))
						cv2.imshow("Current Height Map", cv2.resize(current_state_1[:,:,1], (400,400), interpolation = cv2.INTER_AREA))
						cv2.imshow("Current Goal Map", cv2.resize(current_state_1[:,:,2], (400,400), interpolation = cv2.INTER_AREA))

						cv2.imshow("Next Intensity Map", cv2.resize(next_state_1[:,:,0], (400,400), interpolation = cv2.INTER_AREA))
						cv2.imshow("Next Height Map", cv2.resize(next_state_1[:,:,1], (400,400), interpolation = cv2.INTER_AREA))
						cv2.imshow("Next Goal Map", cv2.resize(next_state_1[:,:,2], (400,400), interpolation = cv2.INTER_AREA))
						cv2.waitKey(1)

						if i == eps_length-1:
							self.episode_done = True
							print("episode done no :",self.ep_no)
							break

				else:
					tmp_goal_position_no = tmp_goal_position_no +1

				if self.episode_done:
					self.ep_no = self.ep_no + 1
					self.episode_done = False
					break



	def set_reward(self,dist_to_goal,current_state, current_intensity_map,battery_current):
		done = 0 #False

		near_robot_intensity = current_intensity_map[18:22,18:22] #0.5m radius from the robot
		near2_robot_intensity = current_intensity_map[16:24,16:24] #0.5m radius from the robot

		if dist_to_goal < self.goal_reach_threshold*1.5:
			reward = 1000
			done = 1 #True
			print(' ------- Goal reached -----------')


		if not(dist_to_goal < self.goal_reach_threshold*1.5):
			r_obs = -(np.mean(near2_robot_intensity)/255)*70
			r_dist = (self.tot_dist_to_goal/dist_to_goal) * 30
			reward = r_dist + r_obs
			done = 0 #False

		if np.max(near_robot_intensity) >= self.solid_obs_intensity_th:
			print(' ------- Solid obstacle nearby -----------')
			r_obs = -100
			reward = -100
			done = 1

		print("Reward :",reward)
		return np.asarray(reward), np.asarray(done)

	def set_reward_v2(self,dist_to_goal,current_state, current_intensity_map,battery_current):
		done = 0 #False

		near_robot_intensity = current_intensity_map[18:22,18:22] #0.5m radius from the robot
		near2_robot_intensity = current_intensity_map[16:24,16:24] #0.5m radius from the robot

		r_power = battery_current * 3

		if dist_to_goal < self.goal_reach_threshold*1.5:
			reward = 1000 
			done = 1 #True
			print(' ------- Goal reached -----------')


		if not(dist_to_goal < self.goal_reach_threshold*1.5):
			r_obs = -(np.mean(near2_robot_intensity)/255)*70
			r_dist = (self.tot_dist_to_goal/dist_to_goal) * 30
			reward = r_dist + r_obs + r_power
			done = 0 #False

		if np.max(near_robot_intensity) >= self.solid_obs_intensity_th:
			print(' ------- Solid obstacle nearby -----------')
			r_obs = -100
			reward = -100 + r_power
			done = 1

		print("Reward :",reward)
		return np.asarray(reward), np.asarray(done)	

	def set_reward_v3(self,dist_to_goal,current_state, current_intensity_map,battery_current):
		done = 0 #False

		near_robot_intensity = current_intensity_map[18:22,18:22] #0.5m radius from the robot
		near2_robot_intensity = current_intensity_map[16:24,16:24] #0.5m radius from the robot

		r_power = battery_current * 5

		if dist_to_goal < self.goal_reach_threshold*1.5:
			reward = 1000 
			done = 1 #True
			print(' ------- Goal reached -----------')


		if not(dist_to_goal < self.goal_reach_threshold*1.5):
			r_obs = -(np.mean(near2_robot_intensity)/255)*70
			r_dist = (self.tot_dist_to_goal/dist_to_goal) * 30
			reward = r_dist + r_obs + r_power
			done = 0 #False

		if np.max(near_robot_intensity) >= self.solid_obs_intensity_th:
			print(' ------- Solid obstacle nearby -----------')
			r_obs = -100
			reward = -100 + r_power
			done = 1

		print("Reward :",reward)
		return np.asarray(reward), np.asarray(done)	

	def set_reward_v4(self,dist_to_goal,current_state, current_intensity_map,battery_current):
		done = 0 #False

		near_robot_intensity = current_intensity_map[18:22,18:22] #0.5m radius from the robot
		near2_robot_intensity = current_intensity_map[16:24,16:24] #0.5m radius from the robot

		r_power = -battery_current * 3

		if dist_to_goal < self.goal_reach_threshold*1.5:
			reward = 1000 
			done = 1 #True
			print(' ------- Goal reached -----------')


		if not(dist_to_goal < self.goal_reach_threshold*1.5):
			r_obs = -(np.mean(near2_robot_intensity)/255)*70
			r_dist = (self.tot_dist_to_goal/dist_to_goal) * 35
			reward = r_dist + r_obs + r_power
			done = 0 #False

		if np.max(near_robot_intensity) >= self.solid_obs_intensity_th:
			print(' ------- Solid obstacle nearby -----------')
			r_obs = -150
			reward = -150 + r_power
			done = 1

		print("Reward :",reward)
		return np.asarray(reward), np.asarray(done)	

	def set_reward_v5(self,dist_to_goal,current_state, current_intensity_map,battery_current):
		done = 0 #False

		near_robot_intensity = current_intensity_map[18:22,18:22] #0.5m radius from the robot
		near2_robot_intensity = current_intensity_map[16:24,16:24] #0.5m radius from the robot

		r_power = -battery_current * 3

		if dist_to_goal < self.goal_reach_threshold*1.5:
			reward = 1000 
			done = 1 #True
			print(' ------- Goal reached -----------')


		if not(dist_to_goal < self.goal_reach_threshold*1.5):
			r_obs = -(np.mean(near2_robot_intensity)/255)*70
			r_dist = (self.tot_dist_to_goal/dist_to_goal) * 40
			reward = r_dist + r_obs + r_power
			done = 0 #False

		if np.max(near_robot_intensity) >= self.solid_obs_intensity_th:
			print(' ------- Solid obstacle nearby -----------')
			r_obs = -500
			reward = -500 + r_power
			done = 1

		print("Reward :",reward)
		return np.asarray(reward), np.asarray(done)	

	def set_reward_v6(self,dist_to_goal,current_state, current_intensity_map,battery_current):
		done = 0 #False

		near_robot_intensity = current_intensity_map[17:23,17:23] #0.5m radius from the robot
		near2_robot_intensity = current_intensity_map[14:26,14:26] #1.5m radius from the robot
		near3_robot_intensity = current_intensity_map[10:30,10:30] #2.5m radius from the robot

		r_power = -battery_current * 2

		if dist_to_goal < self.goal_reach_threshold*1.5:
			reward = 2000 
			done = 1 #True
			print(' ------- Goal reached -----------')


		if not(dist_to_goal < self.goal_reach_threshold*1.5):
			r_obs = -(np.mean(near2_robot_intensity)/255)*50
			r_obs2 = -(np.mean(near3_robot_intensity)/255)*10

			r_dist = (self.tot_dist_to_goal/dist_to_goal) * 45
			reward = r_dist + r_obs + r_obs2 + r_power
			done = 0 #False

		if np.max(near_robot_intensity) >= self.solid_obs_intensity_th:
			print(' ------- Solid obstacle nearby -----------')
			r_obs = -300
			reward = -300 + r_power
			done = 1

		print("Reward :",reward)
		return np.asarray(reward), np.asarray(done)	

	def generate_goal_map(self, goalmap_local,odom_current,odom_goal,tot_dist_to_goal):
		goalmap_local = goalmap_local
		# print('shape goalmap local :',np.shape(goalmap_local),goalmap_local.shape[0])
		for l in range(goalmap_local.shape[0]):
			for j in range(goalmap_local.shape[1]):
			    goalmap_weighted = self.get_dist_cost(l,j,goalmap_local,odom_current,odom_goal,tot_dist_to_goal)

		goalmap_current = goalmap_weighted	

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

		# print('Dist x and y for each grid:',distX_from_rob,distY_from_rob)

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
		

if __name__ == '__main__':

    path = '/offlineRL/dataset'

    dataset = ImportData()

