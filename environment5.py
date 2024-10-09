# New iteration of our MDP which has the [temporal, scatterplot and carrier] x [sensemaking, question] as states
# [observation, generalization, explanation and steer] as actions
import os
import fnmatch
import pdb
from collections import defaultdict
import glob
from read_data_old import read_data
import numpy as np
from Reward_Generator import reward

class environment5:
    def __init__(self):
        path = os.getcwd()
        self.datasets = ['birdstrikes1', 'weather1', 'faa1']
        self.tasks = ['t1', 't2', 't3', 't4']
        self.action_space = ['same', 'modify-1', 'modify-2', 'modify-3', 'modify-4']
        self.obj = read_data()

        self.birdstrikes = {'"dam_eng1"': 1, '"dam_eng2"': 2, '"dam_windshld"': 3, '"dam_wing_rot"': 4, '"damage"': 5, '"dam_eng3"': 6, '"dam_tail"': 7, '"dam_nose"': 8, '"dam_lghts"': 9, '"dam_lg"': 10, '"dam_fuse"': 11, '"dam_eng4"': 12, '"dam_other"': 13, '"cost_repairs"': 14, '"incident_date"': 15, '"time_of_day"': 16, '"faaregion"': 17, '"location"': 18, '"latitude (generated)"': 19, '"longitude (generated)"': 20, '"state"': 21, '"distance"': 22, '"ac_mass"': 23, '"ac_class"': 24, '"speed"': 25, '"height"': 26, '"phase_of_flt"': 27, '"precip"': 28, '"sky"': 29, '"birds_struck"': 30, '"birds_seen"': 31, '"size"': 32, '"number of records"' :33, '"operator"': 34}
        self.weather = {'"tmax_f"': 1, '"tmin_f"': 2, '"tmax"': 3, '"tmin"': 4, '"latitude (generated)"': 5, '"longitude (generated)"': 6, '"lat"': 7, '"lng"': 8, '"state"': 9, '"name"': 10, '"number of records"': 11, '"date"': 12, '"icepellets"': 13, '"freezingrain"': 14, '"blowingsnow"': 15, '"blowingspray"': 16, '"drizzle"': 17, '"freezingdrizzle"': 18, '"prcp"': 19, '"rain"': 20, '"snow"': 21, '"snowgeneral"': 22, '"snwd"': 23, '"hail"': 24, '"glaze"': 25, '"heavyfog"': 26, '"groundfog"': 27, '"icefog"': 28, '"fog"': 29, '"mist"': 30, '"thunder"': 31, '"tornado"': 32, '"dust"': 33, '"highwinds"': 34, '"smoke"': 35}
        self.faa = {'"arrdelay"': 1, '"depdelay"': 2, '"airtime"': 3, '"securitydelay"': 4, '"uniquecarrier"': 5, '"flightdate"': 6, '"distance"': 7, '"origin"': 8, '"dest"': 9, '"latitude (generated)"': 10, '"longitude (generated)"': 11, '"origincityname"': 12, '"destcityname"': 13, '"cancelled"': 14, '"diverted"': 15, '"cancellationcode"': 16, '"number of records"':17}

        # Includes all attributes
        # the numbering is based on descending frequency of appearance 
        # self.birdstrikes = {'"number of records"': 1, '"precip"': 2, '"incident_date"': 3, '"damage"': 4, '"dam_eng1"': 5, '"ac_class"': 6, '"sky"': 7, '"dam_eng2"': 8, '"dam_windshld"': 9, '"dam_wing_rot"': 10, '': 11, '"time_of_day"': 12, '"birds_struck"': 13, '"state"': 14, '"phase_of_flt"': 15, '"faaregion"': 16, '"height"': 17, '"dam_eng3"': 18, '"index_nr"': 19, '"latitude (generated)"': 20, '"longitude (generated)"': 21, '"operator"': 22, '"distance"': 23, '"dam_tail"': 24, '"calculation(#of damage)"': 25, '"location"': 26, '"indicated_damage"': 27, '"dam_lghts"': 28, '"dam_eng4"': 29, '"size"': 30, '"birds_seen"': 31, '"dam_nose"': 32, '"dam_lg"': 33, '"cost_repairs"': 34, '"speed"': 35, '"incident_month"': 36, '"ac_mass"': 37, '"dam_fuse"': 38, '"calculation(sky_clean)"': 39, '"dam_other"': 40, '"reported_date"': 41, '"dam_prop"': 42, '"dam_rad"': 43, '"type_eng"': 44, '"ama"': 45, '"amo"': 46, '"aos"': 47, '"cost_other"': 48, '"cost_other_infl_adj"': 49, '"cost_repairs_infl_adj"': 50, '"calculation(frequency of damage)"': 51, '"warned"': 52, '"airport_id"': 53, '"atype"': 54, '"airport"': 55, '"species"': 56, '"person"': 57, '"runway"': 58, '"nr_injuries"': 59, '"time"': 60, '"incident_year"': 61, '"nr_fatalities"': 62, '"calculation(phase of flt dedup)"': 63, '"ema"': 64, '"emo"': 65, '"eng_1_pos"': 66, '"eng_2_pos"': 67, '"eng_3_pos"': 68, '"eng_4_pos"': 69, '"flt"': 70, '"ingested"': 71, '"num_engs"': 72, '"remains_collected"': 73, '"remains_sent"': 74, '"str_eng1"': 75, '"str_eng2"': 76, '"str_eng3"': 77, '"str_eng4"': 78, '"str_fuse"': 79, '"str_lg"': 80, '"str_lghts"': 81, '"str_nose"': 82, '"str_other"': 83, '"str_prop"': 84, '"str_rad"': 85, '"str_tail"': 86, '"str_windshld"': 87, '"str_wing_rot"': 88, '"transfer"': 89, '"source"': 90, '"enroute"': 91, '"calculation(count([damage])/total(count([ac class])))"': 92, '"calculation(damaged_boolean)"': 93}
        # self.weather = {'"date"': 1, '"state"': 2, '"lat"': 3, '"heavyfog"': 4, '"tmax_f"': 5, '"lng"': 6, '"tmin_f"': 7, '"highwinds"': 8, '"mist"': 9, '"drizzle"': 10, '"groundfog"': 11, '"prcp"': 12, '"tmax"': 13, '"rain"': 14, '"tmin"': 15, '': 16, '"name"': 17, '"fog"': 18, '"latitude (generated)"': 19, '"longitude (generated)"': 20, '"number of records"': 21, '"blowingsnow"': 22, '"icefog"': 23, '"freezingrain"': 24, '"blowingspray"': 25, '"hail"': 26, '"icepellets"': 27, '"smoke"': 28, '"freezingdrizzle"': 29, '"dust"': 30, '"glaze"': 31, '"snow"': 32, '"snowgeneral"': 33, '"thunder"': 34, '"tornado"': 35, '"snwd"': 36, '"calculation(avg_max_temp)"': 37, '"calculation(avg_min_temp)"': 38, '"unknownprecipitation"': 39, '"elevation"': 40, '"calculation(heavy fog (is null))"': 41, '"station"': 42, '"id"': 43, '"calculation(high_wind_count)"': 44, '"calculation(ground fog (is null))"': 45, '"calculation(drizzle (is null))"': 46, '"calculation(mist is null)"': 47}
        # self.faa = {'"uniquecarrier"': 1, '"number of records"': 2, '"arrdelay"': 3, '"flightdate"': 4, '"cancelled"': 5, '"distance"': 6, '"diverted"': 7, '"depdelay"': 8, '"dest"': 9, '': 10, '"longitude (generated)"': 11, '"latitude (generated)"': 12, '"origincityname"': 13, '"origin"': 14, '"airtime"': 15, '"originstate"': 16, '"weatherdelay"': 17, '"lateaircraftdelay"': 18, '"nasdelay"': 19, '"deststate"': 20, '"securitydelay"': 21, '"carrierdelay"': 22, '"actualelapsedtime"': 23, '"arrtime"': 24, '"crsarrtime"': 25, '"crsdeptime"': 26, '"crselapsedtime"': 27, '"deptime"': 28, '"taxiin"': 29, '"taxiout"': 30, '"destcityname"': 31, '"calculation(<>_delta)"': 32, '"calculation(is delta flight)"': 33, '"calculation([arrdelay]+[depdelay])"': 34, '"calculation(arrival y/n)"': 35, '"calculation(delayed y/n)"': 36, '"cancellationcode"': 37, '"calculation(percent delta)"': 38, '"tailnum"': 39, '"calculation(delay?)"': 40, '"calculation([dest]+[origin])"': 41, '"calculation(arrdelayed)"': 42, '"calculation(total delays)"': 43, '"calculation(depdelayed)"': 44, '"wheelsoff"': 45}

        # self.birdstrikes = {'"number of records"': 1, '"precip"': 2, '"incident_date"': 3, '"damage"': 4, '"dam_eng1"': 5, '"ac_class"': 6, '"sky"': 7, '"dam_eng2"': 8, '"dam_windshld"': 9, '"dam_wing_rot"': 10, '': 11, '"time_of_day"': 12, '"birds_struck"': 13, '"state"': 14, '"phase_of_flt"': 15, '"faaregion"': 16, '"height"': 17, '"dam_eng3"': 18, '"index_nr"': 19, '"latitude (generated)"': 20, '"longitude (generated)"': 21, '"operator"': 22, '"distance"': 23, '"dam_tail"': 24, '"calculation(#of damage)"': 25, '"location"': 26, '"indicated_damage"': 27, '"dam_lghts"': 28, '"dam_eng4"': 29, '"size"': 30, '"birds_seen"': 31, '"dam_nose"': 32, '"dam_lg"': 33, '"cost_repairs"': 34, '"speed"': 35, '"incident_month"': 36, '"ac_mass"': 37, '"dam_fuse"': 38, '"calculation(sky_clean)"': 39, '"dam_other"': 40}
        # self.weather = {'"date"': 1, '"state"': 2, '"lat"': 3, '"heavyfog"': 4, '"tmax_f"': 5, '"lng"': 6, '"tmin_f"': 7, '"highwinds"': 8, '"mist"': 9, '"drizzle"': 10, '"groundfog"': 11, '"prcp"': 12, '"tmax"': 13, '"rain"': 14, '"tmin"': 15, '': 16, '"name"': 17, '"fog"': 18, '"latitude (generated)"': 19, '"longitude (generated)"': 20, '"number of records"': 21, '"blowingsnow"': 22, '"icefog"': 23, '"freezingrain"': 24, '"blowingspray"': 25, '"hail"': 26, '"icepellets"': 27, '"smoke"': 28, '"freezingdrizzle"': 29, '"dust"': 30, '"glaze"': 31, '"snow"': 32, '"snowgeneral"': 33, '"thunder"': 34, '"tornado"': 35, '"snwd"': 36, '"calculation(avg_max_temp)"': 37, '"calculation(avg_min_temp)"': 38, '"unknownprecipitation"': 39, '"elevation"': 40}
        # self.faa = {'"uniquecarrier"': 1, '"number of records"': 2, '"arrdelay"': 3, '"flightdate"': 4, '"cancelled"': 5, '"distance"': 6, '"diverted"': 7, '"depdelay"': 8, '"dest"': 9, '': 10, '"longitude (generated)"': 11, '"latitude (generated)"': 12, '"origincityname"': 13, '"origin"': 14, '"airtime"': 15, '"originstate"': 16, '"weatherdelay"': 17, '"lateaircraftdelay"': 18, '"nasdelay"': 19, '"deststate"': 20, '"securitydelay"': 21, '"carrierdelay"': 22, '"actualelapsedtime"': 23, '"arrtime"': 24, '"crsarrtime"': 25, '"crsdeptime"': 26, '"crselapsedtime"': 27, '"deptime"': 28, '"taxiin"': 29, '"taxiout"': 30, '"destcityname"': 31, '"calculation(<>_delta)"': 32, '"calculation(is delta flight)"': 33, '"calculation([arrdelay]+[depdelay])"': 34, '"calculation(arrival y/n)"': 35, '"calculation(delayed y/n)"': 36, '"cancellationcode"': 37, '"calculation(percent delta)"': 38, '"tailnum"': 39, '"calculation(delay?)"': 40}

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


    def set_dictionary(self, dataset):
        if dataset == 'birdstrikes1':
            return self.birdstrikes
        elif dataset == 'weather1':
            return self.weather
        else: # 'FAA1'
            return self.faa

    def process_data(self, dataset, user, thres, algorithm):

        #Get interaction sequence from the user interaction log
        self.obj.create_connection(r"Tableau.db")
        data = self.obj.merge2(dataset, user)
        mask = self.set_dictionary(dataset)
        self.obj.conn.close()
        #use the following to generate state, action, reward sequence from raw data
        u = reward()
        max_len = 10

        raw_states, self.mem_action, self.mem_reward = u.generate(data, dataset)

        for _, s in enumerate(raw_states):
            
            #for NN based implementations
            state = []
            for s_prime in s:
                try:
                    state.append(mask[s_prime])
                except KeyError:
                    state.append(0)
            state = sorted(state)
            # print(s, state)
            
            while len(state) < max_len:
                state.append(0)

            if algorithm == 'SVM':
                self.mem_states.append(state[:max_len]) # for online SVM
            elif algorithm == 'Actor-Critic' or algorithm == 'Reinforce':
                self.mem_states.append(np.array(state[:max_len])) #actor-critic 
            else: # Q-learning, SARSA, Q-learning_v2, SARSA_v2, Greedy, Bayesian
                self.mem_states.append(tuple(state[:max_len])) #tuple for Q-learning and SARSA
        
        itrs = len(self.mem_states)        
        self.threshold = int(itrs * thres)   
        
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

        return next_state, cur_reward, self.done, prediction, cur_action