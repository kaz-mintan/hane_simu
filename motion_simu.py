# coding:utf-8
# http://neuro-educator.com/rl1/

import numpy as np

import time
from datetime import datetime

#for test
from dummy_modules import dummy_evaluator
from dummy_modules import hand_motion

from learn_modules import motion_gen
from learn_modules import test_save_txt

#define for test
episode_num=50
action_num=1
state_num=1
mode="delta"

epsilon = 0.1
mu = 0.9
epoch = 1000

# class
motion_frame= motion_gen.Motion_gen(episode_num,action_num,state_num,mode)
motion_frame.setNN(epsilon, mu, epoch)

hand = hand_motion.dummy_Hand(0)

# main loop
for episode in range(episode_num-1):  #repeat for number of trials
    # simulation mode
    dummy_state=hand.get_ir()
    motion_frame.set_state(dummy_state,episode)#TODO
    #print("dummystate", motion_frame.state_mean)
    #seq2feature(state_mean[:,episode], state, ir_no,type_face)

    #save data of state_mean
    with open('test_state_mean.csv', 'a') as smean_handle:
        np.savetxt(smean_handle,test_save_txt.tmp_log(np.hstack((np.array([episode]),motion_frame.state_mean[:,episode])),datetime.now()),fmt="%.3f",delimiter=",")

    ### calcurate a_{t} based on s_{t}
    motion_frame.gen_action(episode)
    #just show action image figure#TODO

    #save data of action
    with open('test_action_start.csv', 'a') as act_handle:
        np.savetxt(act_handle,test_save_txt.tmp_log(np.hstack((np.array([episode]),motion_frame.random[episode],motion_frame.action[:,episode])),datetime.now()),fmt="%.3f",delimiter=",")

    ### calcurate r_{t}
    motion_frame.set_reward(calced_reward)#calc

    #save data of reward
    with open('test_reward.csv', 'a') as reward_handle:
        np.savetxt(reward_handle,test_save_txt.tmp_log(np.hstack((np.array([episode+1]),motion_frame.reward[episode+1])),datetime.now()),fmt="%.3f",delimiter=",")

    #update q function
    motion_frame.q_teacher = motion_frame.Q_func.update(motion_frame.state_mean,motion_frame.action,episode-1,motion_frame.q_teacher,motion_frame.reward,motion_frame.next_q)

    motion_frame.go_next()
