# New iteration of our MDP which has the [temporal, scatterplot and carrier] x [sensemaking, question] as states
# [observation, generalization, explanation and steer] as actions
import os
import fnmatch
import pdb
from collections import defaultdict
import glob
import pandas as pd
from read_data import read_data
import numpy as np
# from Categorizing_v3 import utilities
from Reward_Generator import reward
from Categorizing_v4 import Categorizing

class environment5:
    def __init__(self):
        path = os.getcwd()
        self.datasets = ['birdstrikes1', 'weather1', 'faa1']
        self.tasks = ['t1', 't2', 't3', 't4']
        self.action_space = ['same', 'modify-1', 'modify-2', 'modify-3', 'modify-4']
        self.obj = read_data()
        
        self.steps = 0
        self.done = False  # Done exploring the current subtask
        self.mem_states = []
        self.mem_reward = []
        self.mem_action = []
        self.threshold = 0

    def reset(self, all=False, test=False):
        # Resetting the variables used for tracking position of the agents
        if test:
            self.steps = self.threshold
        else:
            self.steps = 0
        self.done = False
        if all:
            self.mem_reward = []
            self.mem_states = []
            self.mem_action = []
            return

        s, r, a = self.cur_inter(self.steps)
        return s


    def get_state(self, cat, attributes, algo, dataset):
        state_len = len(cat.states)
        state = np.zeros(state_len, dtype = np.int32)
        
        high_level_attrs = cat.get_category(attributes, dataset)        
        for attrs in high_level_attrs:
            if attrs != None:
                state[cat.states[attrs]] = 1
                
        if algo == 'SARSA' or algo == 'Qlearn' or algo == 'Greedy' or algo == 'WSLS':
            state_str = ''.join(map(str, state))
            return state_str
            # converts the 0/1 in the numpy array into a string and using that string as state rather than numpy.             
        return state

    def process_data(self, dataset, user, thres, algo):

        #Get interaction sequence from the user interaction log
        self.obj.create_connection(r"Tableau.db")
        data = self.obj.merge2(dataset, user)
        # self.obj.close()
        #use the following to generate state, action, reward sequence from raw data
        u = reward()
        cat = Categorizing(dataset)
        raw_states, self.mem_action, self.mem_reward = u.generate(data, dataset)
        for i, s in enumerate(raw_states):
            self.mem_states.append(self.get_state(cat, s, algo, dataset))
        itrs = len(self.mem_states)        
        self.threshold = int(itrs * thres)   
        # print(dataset, user, thres, len(raw_interactions), len(self.mem_states), len(self.mem_action), len(self.mem_reward))     
        
        # for idx in range(len(raw_interactions) - 1):
        #     print(idx, "->", raw_interactions[idx], self.mem_states[idx], raw_actions[idx])
        # return data

    def cur_inter(self, steps):
        return self.mem_states[steps], self.mem_reward[steps], self.mem_action[steps]

    def peek_next_step(self):
        if len(self.mem_states) > self.steps + 1:
            return False, self.steps + 1
        else:
            return True, 0

    def take_step_action(self, test = False):
        if test:
            if len(self.mem_states) > self.steps + 3:
                self.steps += 1
            else:
                self.done = True
                self.steps = 0
        else:
        # print(self.steps)
            if self.threshold > self.steps + 1:
                self.steps += 1
            else:
                self.done = True
                self.steps = 0

    # predicted_action = action argument refers to action number
    def step(self, cur_state, pred_action, test = False):
        _, cur_reward, cur_action = self.cur_inter(self.steps)
        
        _, temp_step = self.peek_next_step()
        next_state, next_reward, next_action = self.cur_inter(temp_step)
        # print(cur_action, pred_action)
        if cur_action == pred_action: #check if the current action matches with the predicted action 
            prediction = 1
        else:
            prediction = 0
            cur_reward = 0

        self.take_step_action(test)

        return next_state, cur_reward, self.done, prediction