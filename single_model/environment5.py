# New iteration of our MDP which has the [temporal, scatterplot and carrier] x [sensemaking, question] as states
# [observation, generalization, explanation and steer] as actions
import os
import fnmatch
import pdb
from collections import defaultdict
import glob
from read_data import read_data
import numpy as np
from Reward_Generator import reward

class environment5:
    def __init__(self):
        path = os.getcwd()
        self.datasets = ['birdstrikes1', 'weather1', 'faa1']
        self.tasks = ['t1', 't2', 't3', 't4']
        self.action_space = ['same', 'modify-1', 'modify-2', 'modify-3', 'modify-4']

        self.birdstrikes = {'"dam_eng1"': 1, '"dam_eng2"': 2, '"dam_windshld"': 3, '"dam_wing_rot"': 4, '"damage"': 5, '"dam_eng3"': 6, '"dam_tail"': 7, '"dam_nose"': 8, '"dam_lghts"': 9, '"dam_lg"': 10, '"dam_fuse"': 11, '"dam_eng4"': 12, '"dam_other"': 13, '"cost_repairs"': 14, '"incident_date"': 15, '"time_of_day"': 16, '"faaregion"': 17, '"location"': 18, '"latitude (generated)"': 19, '"longitude (generated)"': 20, '"state"': 21, '"distance"': 22, '"ac_mass"': 23, '"ac_class"': 24, '"speed"': 25, '"height"': 26, '"phase_of_flt"': 27, '"precip"': 28, '"sky"': 29, '"birds_struck"': 30, '"birds_seen"': 31, '"size"': 32}
        self.weather = {'tmax_f': 1, 'tmin_f': 2, 'tmax': 3, 'tmin': 4, 'latitude (generated)': 5, 'longitude (generated)': 6, 'lat': 7, 'lng': 8, 'state': 9, 'name': 10, 'number of records': 11, 'date': 12, 'icepellets': 13, 'freezingrain': 14, 'blowingsnow': 15, 'blowingspray': 16, 'drizzle': 17, 'freezingdrizzle': 18, 'prcp': 19, 'rain': 20, 'snow': 21, 'snowgeneral': 22, 'snwd': 23, 'hail': 24, 'glaze': 25, 'heavyfog': 26, 'groundfog': 27, 'icefog': 28, 'fog': 29, 'mist': 30, 'thunder': 31, 'tornado': 32, 'dust': 33, 'highwinds': 34, 'smoke': 35}
        self.faa = {'"arrdelay"': 1, '"depdelay"': 2, '"airtime"': 3, '"securitydelay"': 4, '"uniquecarrier"': 5, '"flightdate"': 6, '"distance"': 7, '"origin"': 8, '"dest"': 9, '"latitude (generated)"': 10, '"longitude (generated)"': 11, '"origincityname"': 12, '"destcityname"': 13, '"cancelled"': 14, '"diverted"': 15, '"cancellationcode"': 16}
        self.encoding = None

        self.obj = read_data()
        
        self.steps = 0
        self.done = False  # Done exploring the current subtask
        self.mem_states = []
        self.mem_reward = []
        self.mem_action = []
        
    def reset(self, all=False):
        # Resetting the variables used for tracking position of the agents
        self.steps = 0
        self.done = False
        if all:
            self.mem_reward = []
            self.mem_states = []
            self.mem_action = []
            return

        s, r, a = self.cur_inter(self.steps)
        return s


    def set_dictionary(self, dataset):
        if dataset == 'birdstrikes1':
            return self.birdstrikes
        elif dataset == 'weather1':
            return self.weather
        else: # 'FAA1'
            return self.faa


    def process_data(self, dataset, user, algorithm):

        #Get interaction sequence from the user interaction log
        self.obj.create_connection(r"Tableau.db")
        data = self.obj.merge2(dataset, user)
        mask = self.set_dictionary(dataset)

        #use the following to generate state, action, reward sequence from raw data
        u = reward()
        raw_states, self.mem_action, self.mem_reward = u.generate(data, dataset)
        for i, s in enumerate(raw_states):
            
            #for NN based implementations
            state = []
            for s_prime in s:
                try:
                    state.append(mask[s_prime])
                except KeyError:
                    state.append(0)

            while len(state) < 10:
                state.append(0)
            state = sorted(state)
            if algorithm == 'SVM':
                self.mem_states.append(state[:10]) # online SVM
            elif algorithm == 'Qlearn' or algorithm == 'SARSA':
                self.mem_states.append(tuple(state[:10])) #tuple for Q-learning and SARSA
            elif algorithm == 'Actor-Critic' or algorithm == 'Reinforce':
                self.mem_states.append(np.array(state[:10]))

    def cur_inter(self, steps):
        return self.mem_states[steps], self.mem_reward[steps], self.mem_action[steps]

    def peek_next_step(self):
        if len(self.mem_states) > self.steps + 1:
            return False, self.steps + 1
        else:
            return True, 0

    def take_step_action(self):
        if len(self.mem_states) > self.steps + 2:
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

        self.take_step_action()

        return next_state, cur_reward, self.done, prediction, cur_action
