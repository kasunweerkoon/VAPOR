#!/usr/bin/env python

import sys, time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from scipy.ndimage import filters
import pandas as pd
from sklearn.neighbors import KernelDensity
import cv2
import roslib
import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import JointState
from spot_msgs.msg import BatteryStateArray
from std_msgs.msg import Float32MultiArray
import math


VERBOSE=False


class JointData:
    def __init__(self, prior_path):

        prior_file = prior_path +'/prior.csv'
        df_prior = pd.read_csv(prior_file)
        sub = rospy.Subscriber("/joint_states", JointState, callback = self.joint_callback)

        self.variance_topic_name = '/prop_value'
        self.variance_pub = rospy.Publisher(self.variance_topic_name, Float32MultiArray,queue_size=10)
        self.variance_msg = Float32MultiArray()
        self.counter = 0

        self.max_effort_len = 12
        self.max_position_len = 10
        self.pop_window = 5

        self.sample_rate = 10


        self.Edata = []
        self.Pdata = []
        self.Vdata = []
        self.current = []
        self.pca_data = []
        self.variance_list_pc1 = []
        self.variance_list_pc2 = []

        self.prop_value = 0        # (0-5) 0 lowest vibration ||  5 Highest vibration
        self.prop_range0 = [0,1]         # range of variance values corrsponding to 
        self.prop_range1 = [1,2]         # each prop_value
        self.prop_range2 = [2,3]
        self.prop_range3 = [3,4]
        self.prop_range4 = [4,5]
        self.prop_range5 = [5,6]


        self.published_var = 0  # The published variance

        self.terr_type = 0     #terrain type:  solid = 0 ,   deformable/sand = 1 ,  bushes = 2

        for i in range(len(df_prior)):
            pca_values = [df_prior.loc[i,'t1'],
                          df_prior.loc[i,'t2'],
                          df_prior.loc[i,'t3'],
                          df_prior.loc[i,'t4'],
                          df_prior.loc[i,'t5'],
                          df_prior.loc[i,'t6'],
                          df_prior.loc[i,'t7'],
                          df_prior.loc[i,'t9']]
            self.pca_data.append(pca_values)
        #print(len(self.pca_data))
        self.prior_pca_len = len(self.pca_data)

    def joint_callback(self, msg):

        counter1 = self.counter
        #print(counter1)

        # Recording Joint data
        effort_data = [msg.effort[i] for i in range(12)]
        position_data = [msg.position[i] for i in range(12)]
        velocity_data = [msg.velocity[i] for i in range(12)]

        # Record Battery Data
        battery_data = rospy.wait_for_message('/spot/status/battery_states', BatteryStateArray, timeout=None)
        self.current.append(battery_data.battery_states[0].current * -1)
        #print(battery_data.battery_states[0].current)

        if len(self.Edata) < self.max_effort_len:
            self.Edata.append(effort_data)
        else:
            self.Edata[counter1 % self.max_effort_len] = effort_data

        if len(self.Pdata) < self.max_position_len:
            self.Pdata.append(position_data)
            self.Vdata.append(velocity_data)
        else:
            self.Pdata[counter1 % self.max_position_len] = position_data
            self.Vdata[counter1 % self.max_position_len] = velocity_data

        # Analysis
        if counter1 >= 10:
            effort_analysis_results = self.analyze_effort_data()
            #print(analysis_results[0]['V'], analysis_results[0]['Q'])

        if counter1 >= self.pop_window - 1:
            position_analysis_results = self.analyze_position_data()

            #velocity_analysis_results = self.analyze_velocity_data()
            #print(analysis_results[0]['AC'])

        if counter1 >= 10:
            pcaList = [
                effort_analysis_results[0]['V'],
                effort_analysis_results[0]['AC'],
                effort_analysis_results[0]['AD'],
                effort_analysis_results[0]['AE'],
                effort_analysis_results[0]['AF'],
                effort_analysis_results[0]['AG'],
                position_analysis_results[0]['AC'],
                battery_data.battery_states[0].current * -1
            ]

            #print(pcaList)

            self.pca_data.append(pcaList)

            if len(self.pca_data) >= self.prior_pca_len + 10:
                self.pca_data.pop(self.prior_pca_len)

            #if len(self.pca_data) >= 2:
            if counter1%self.sample_rate == 0:
                data_array = np.array(self.pca_data)

                scaled_data = StandardScaler().fit_transform(data_array)

                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(scaled_data)

                principalDf = pd.DataFrame(data = principal_components
                    , columns = ['principal component 1', 'principal component 2'])


                
                # check mean and variance
                mean_principal_component_1 = np.mean(principal_components[:, 0])
                mean_principal_component_2 = np.mean(principal_components[:, 1])

                variance_principal_component_1 = np.var(principalDf['principal component 1'][len(principalDf)-10:len(principalDf)])
                variance_principal_component_2 = np.var(principalDf['principal component 2'][len(principalDf)-10:len(principalDf)])

                # self.variance_msg.data = [variance_principal_component_1, variance_principal_component_2, self.terr_type]

                # Append variances to their respective lists
                self.variance_list_pc1.append(variance_principal_component_1)
                self.variance_list_pc2.append(variance_principal_component_2)
            
                self.variance_msg.data = np.array([np.clip(variance_principal_component_1,0,5),np.clip(variance_principal_component_2,0,5)])#[self.prop_value]

            self.variance_pub.publish(self.variance_msg)

      
            

        self.counter += 1

        if len(self.Edata) > self.max_effort_len:
            self.Edata.pop(0)
        if len(self.Pdata) > self.max_position_len:
            self.Pdata.pop(0)
            self.Vdata.pop(0)


    def analyze_effort_data(self):
        first, second, third, forth = 2, 5, 8, 11
        analysis_results = []
        

        for i in range(10, len(self.Edata)):
            dataMax = [0, 0, 0, 0]
            counters = [0, 0, 0, 0]
            for j in range(11):
                #print(dataMax)
                data = [self.Edata[i - j][first], self.Edata[i - j][second],
                        self.Edata[i - j][third], self.Edata[i - j][forth]]
                for k in range(4):
                    if data[k] > dataMax[k]:
                        counters[k] += 1
                        dataMax[k] = data[k]

            Q = sum(dataMax) / 4.0
            R, S, T, U = counters
            V = sum(counters)

            differences = [dataMax[k] - Q for k in range(4)]
            ratios = [differences[k] / Q for k in range(4)]
            abs_sum_ratios = sum([abs(ratio) for ratio in ratios])

            X, Y, Z, AA = differences
            AC, AD, AE, AF = ratios
            AG = abs_sum_ratios

            t1 = V
            t2, t3, t4, t5 = ratios
            t6 = abs_sum_ratios

            result = {'Q': Q, 'R': R, 'S': S, 'T': T, 'U': U, 'V': V,
                      'X': X, 'Y': Y, 'Z': Z, 'AA': AA, 'AC': AC, 'AD': AD, 'AE': AE, 'AF': AF, 'AG': AG}

            analysis_results.append(result)

        return analysis_results
    

    def analyze_position_data(self):
        analysis_results = []

        for i in range(self.pop_window - 1, len(self.Pdata)):
            diff_values = []
            for idx, (start, end) in enumerate([(0, 1), (3, 4), (6, 7), (9, 10)]):
                max_val = max([row[start] for row in self.Pdata[i - self.pop_window + 1:i + 1]])
                min_val = min([row[start] for row in self.Pdata[i - self.pop_window + 1:i + 1]])
                diff = max_val - min_val
                diff_values.append(diff)

            total_diff_sum = sum(diff_values)

            result = {'R': diff_values[0], 'U': diff_values[1], 'X': diff_values[2], 'AA': diff_values[3], 'AC': total_diff_sum}
            analysis_results.append(result)

        return analysis_results
    


if __name__ == '__main__':
    rospy.init_node('Prop_Function', anonymous=True)
    # df_prior = pd.read_csv('prior.csv')
    prior_path = '/home/kasun/offlineRL/CQL/CQL-SAC-w-prop/dataset_generation'
    config = JointData(prior_path)
    # sub = rospy.Subscriber("/joint_states", JointState, callback = config.joint_callback)

    sub2 = rospy.Subscriber("/spot/status/battery_states", BatteryStateArray)
    rospy.loginfo("Node has been started")  

    rospy.spin()
    
