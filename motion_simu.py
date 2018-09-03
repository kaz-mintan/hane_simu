# coding:utf-8
# http://neuro-educator.com/rl1/

import numpy as np
import sys

from neural_network import *

import time
from datetime import datetime

#for test
from dummy_modules import dummy_evaluator
from dummy_modules import hand_motion

from datetime import datetime

from action_convert import *

class Motion_gen:
    #define
    gamma = 0.9
    alpha = 0.5
    type_face = 5
    type_ir = 5

    possible_a = np.linspace(0,1,20)
    next_q=0

    #type_action = 1 #3(pwm,delay,num of array)
    #state_ir = 1 #number of argument of state(ir sensor)
    def __init__(self,epi_num,a_num,s_num,r_mode)
        self.num_episodes=epi_num
        self.type_action = a_num
        self.state_ir = s_num
        self.mode=r_mode

        #init arrays?
        self.state_mean = np.zeros((self.type_face+self.state_ir,self.num_episodes))
        self.state_before = np.zeros_like(self.state_mean) #for delta mode

        self.action = np.zeros((self.type_action,self.num_episodes))
        self.reward = np.zeros(self.num_episodes+1)
        self.random = np.zeros(self.num_episodes)

        #init action
        action[:,0] = np.array([np.random.uniform(0,1)]
        #action[:,0] = np.array([np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1)])

    def setNN(epsilon, mu, epoch):
        ## set qfunction as nn
        self.q_input_size = self.type_face + self.state_ir + self.type_action
        self.q_output_size = 1
        self.q_hidden_size = (self._input_size + self.q_output_size )/2
        self.q_teacher = np.zeros((self.q_output_size,self.num_episodes))
        self.Q_func = Neural(self.q_input_size, self.q_hidden_size, self.q_output_size, epsilon, mu, epoch, self.gamma, self.alpha)

    def set_state(self,state_data,episode):
        self.state_mean[:,episode]=state_data

    def gen_action(self,episode):
        self.random[episode], self.action[:,episode], self.next_q = self.Q_func.test_gen_action(self.possible_a, self.state_mean, episode, self.random_rate)

    def set_reward(self,calced_reward):
        self.reward[episode+1]=calced_reward

    def go_next(self):
        self.state_before = self.state

episode_num=50
action_num=1
state_num=1
mode="delta"

epsilon = 0.1
mu = 0.9
epoch = 1000

motion_gen = Motion_gen(episode_num,action_num,state_num,mode)
motion_gen.setNN(epsilon, mu, epoch)

# main loop
for episode in range(episode_num-1):  #repeat for number of trials
    # simulation mode
    motion_gen.set_state(dummy_state,episode)#TODO
    #seq2feature(state_mean[:,episode], state, ir_no,type_face)

    #save data of state_mean
    with open('test_state_mean.csv', 'a') as smean_handle:
        numpy.savetxt(smean_handle,tmp_log(np.hstack((np.array([episode]),state_mean[:,episode])),datetime.now()),fmt="%.3f",delimiter=",")

    ### calcurate a_{t} based on s_{t}
    motion_gen.gen_action(episode)
    #just show action image figure#TODO

    #save data of action
    with open('test_action_start.csv', 'a') as act_handle:
        numpy.savetxt(act_handle,tmp_log(np.hstack((np.array([episode]),random[episode],convert_action(action[:,episode],ir_no))),datetime.now()),fmt="%.3f",delimiter=",")

    ### calcurate r_{t}
    motion_gen.set_reward(calced_reward)#calc

    #save data of reward
    with open('test_reward.csv', 'a') as reward_handle:
        numpy.savetxt(reward_handle,tmp_log(np.hstack((np.array([episode+1]),reward[episode+1])),datetime.now()),fmt="%.3f",delimiter=",")

    #update q function
    motion_gen.q_teacher = motion_gen.Q_func.update(motion_gen.state_mean,motion_gen.action,episode-1,motion_gen.q_teacher,motion_gen.reward,motion_gen.next_q)

    motion_gen.go_next()
