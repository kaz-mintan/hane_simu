# coding:utf-8
# http://neuro-educator.com/rl1/

import numpy as np

import time
from datetime import datetime

#for test
from dummy_modules import dummy_evaluator
from dummy_modules import hand_motion

from motion_gen import *

#define for test
episode_num=50
action_num=1
state_num=1
mode="delta"

epsilon = 0.1
mu = 0.9
epoch = 1000

# class
motion_gen = Motion_gen(episode_num,action_num,state_num,mode)
motion_gen.setNN(epsilon, mu, epoch)

# main loop
for episode in range(episode_num-1):  #repeat for number of trials
    # simulation mode
    dummy_state
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
