import pandas as pd
import ast
import numpy as np 
import itertools
from Reward_Generator import reward
from read_data import read_data
import pdb
import random 
from Categorizing_v4 import Categorizing
from itertools import combinations
import random
import nltk
from nltk.corpus import treebank
from nltk.tag.hmm import HiddenMarkovModelTrainer
from nltk.tag import hmm
from nltk.probability import LidstoneProbDist
from collections import defaultdict

def get_state(cat, attributes, dataset):
    state_len = len(cat.states)
    state = np.zeros(state_len, dtype = np.int32)
    
    high_level_attrs = cat.get_category(attributes, dataset)        
    return '-'.join(map(str, high_level_attrs))
    # for attrs in high_level_attrs:
    #     if attrs != None:
    #         state[cat.states[attrs]] = 1
            
    # state_str = ''.join(map(str, state))
    # return state_str


obj = read_data()
obj.create_connection(r"Tableau.db")
r = reward()
final_results = np.zeros(9, dtype = float)
for d in r.datasets:
    users = obj.get_user_list_for_dataset(d)
    #getting all users data for model initialization
    cleaned_data = []
    cat = Categorizing(d)

    for user in users:
        u = user[0]
        data = obj.merge2(d, u)
        raw_states, raw_actions, mem_reward = r.generate(data, d)
        sequences = []
        dd =  defaultdict(int)
        cnt = 1
        for i, s in enumerate(raw_states):
            if len(s) < 3:
                continue
            states = get_state(cat, s, d)
            temp = []
            for items in states:
                if items in dd:
                    temp.append(dd[items])
                else:
                    dd[items] = cnt
                    cnt += 1
                    temp.append(dd[items])
            # states = '-'.join(map(str, states))
            if raw_actions[i] == 0:
                action = 'same'
            else:
                action = f'modify-{raw_actions[i]}'

            sequences.append((str(temp), action))
            # sequences.append((states, action))
            
        # pdb.set_trace()

        threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = []
        for t in threshold:
            split = int(len(sequences) * t)
            predicted_tags = []
            true_tags = []
            states=[]
            for i in range(split , len(sequences)):
                try:
                     trainer = nltk.HiddenMarkovModelTagger.train([sequences[:i]])
                     state, true_tag = sequences[i]
                     prediction = trainer.tag([state])
                     predicted_tag = prediction[0][1]
                    
                except ValueError:
                    # print('Value Error')
                    continue

                predicted_tags.append(predicted_tag)
                true_tags.append(true_tag)
                states.append(state)

            assert len(states) == len(true_tags) == len(predicted_tags)
            # print('States:', states)
            # print ('True Tags:', true_tags)
            # print ('Predicted Tags:', predicted_tags)
            # pdb.set_trace()
            #Calculate accuracy between predicted_tags and true_tags
            accuracy = np.mean(np.array(true_tags) == np.array(predicted_tags))
            results.append(accuracy)
            # print(accuracy)

        final_results = np.add(final_results, results)

    final_results /= len(users)
    print(d, ", ".join(f"{x:.2f}" for x in final_results))

# birdstrikes1 0.42, 0.40, 0.39, 0.38, 0.37, 0.37, 0.39, 0.41, 0.42
# weather1 0.51, 0.51, 0.50, 0.51, 0.52, 0.52, 0.50, 0.55, 0.51
# faa1 0.55, 0.54, 0.54, 0.55, 0.52, 0.54, 0.53, 0.55, 0.49