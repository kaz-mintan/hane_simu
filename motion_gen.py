import numpy as np
from neural_network import *

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
    def __init__(self,epi_num,a_num,s_num,r_mode):
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
        self.action[:,0] = np.array([np.random.uniform(0,1)])
        #action[:,0] = np.array([np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1)])

    def setNN(self,epsilon, mu, epoch):
        ## set qfunction as nn
        self.q_input_size = self.type_face + self.state_ir + self.type_action
        self.q_output_size = 1
        self.q_hidden_size = (self.q_input_size + self.q_output_size )/2
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


