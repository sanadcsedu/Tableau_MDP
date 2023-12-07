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
from Categorizing_v3 import utilities

class environment5:
    def __init__(self):
        path = os.getcwd()
        self.datasets = ['birdstrikes1', 'weather1', 'faa1']
        self.tasks = ['t1', 't2', 't3', 't4']
        self.obj = read_data()
        self.obj.create_connection(r"Tableau.db")
        # self.action_space = {'Modify': 0, 'Keep': 1}
        self.action_space = {'Add':0, 'Remove': 1, 'Keep': 2}
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


    def get_state(self, high_level_state, algo):
        if algo == 'Qlearn' or algo == 'SARSA':
            state = high_level_state 
        else: 
            # high_level_states = ['Hypothesis_Generation', 'Sensemaking']
            high_level_states = ['Sensemaking', 'Foraging', 'Navigation']
            state = np.zeros(len(high_level_states), dtype = np.int)
            idx = 0
            for idx, s in enumerate(high_level_states):
                if s == high_level_state:
                    state[idx] = 1
                    break
            return state

    def process_data(self, dataset, user, thres, algo):

        #Get interaction sequence from the user interaction log
        data = self.obj.merge2(dataset, user)
        #use the following to generate state, action, reward sequence from raw data
        u = utilities()
        raw_states, raw_actions, self.mem_reward = u.generate(data, dataset)
        for s in raw_states:
            self.mem_states.append(self.get_state(s, algo))
        for a in raw_actions:
            self.mem_action.append(self.action_space[a])
        itrs = len(self.mem_states)        
        self.threshold = int(itrs * thres)   
        # print(dataset, user, thres, len(self.mem_states), len(self.mem_action), len(self.mem_reward))     
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
            # cur_reward = 0

        self.take_step_action(test)

        return next_state, cur_reward, self.done, prediction