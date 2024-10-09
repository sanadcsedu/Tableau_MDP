import pandas as pd
import ast
import numpy as np 
import itertools
from Reward_Generator import reward
from read_data_old import read_data
import pdb
import random 
from itertools import combinations
import random
import nltk
from nltk.corpus import treebank
from nltk.tag.hmm import HiddenMarkovModelTrainer
from nltk.tag import hmm
from nltk.probability import LidstoneProbDist
from collections import defaultdict
import environment5 as environment5

obj = read_data()
obj.create_connection(r"Tableau.db")
r = reward()
final_results = np.zeros(9, dtype = float)
final_cnt = np.zeros((5, 9), dtype = float)
final_split_accu = np.zeros((5, 9), dtype = float)
# action_space = {'same':0, 'modify-1':1, 'modify-2':2, 'modify-3':3, 'modify-4':4}
for d in r.datasets:
    print("------", d, "-------")
    users = obj.get_user_list_for_dataset(d)
    #getting all users data for model initialization
    cleaned_data = []
    # cat = Categorizing(d)

    for user in users:
        u = user[0]
        data = obj.merge2(d, u)
        raw_states, raw_actions, mem_reward = r.generate(data, d)
            
        threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = []
        accu_split = [[] for _ in range(5)]
        cnt_split = [[] for _ in range(5)]
            
        env = environment5.environment5()
        env.process_data(d, u, 1, 'Greedy')
        sequences = []
        for ii in range(len(env.mem_action)):
            sequences.append((env.mem_states[ii], env.mem_action[ii]))
        
        for t in threshold:
            split = int(len(sequences) * t)
            predicted_tags = []
            true_tags = []
            states=[]
            split_accs = [[] for _ in range(5)]
            for i in range(split , len(sequences)):
                try:
                    trainer = nltk.HiddenMarkovModelTagger.train([sequences[:i]])
                    state, true_tag = sequences[i]
                    prediction = trainer.tag([state])
                    predicted_tag = prediction[0][1]
                    
                except ValueError:
                    continue

                predicted_tags.append(predicted_tag)
                true_tags.append(true_tag)
                states.append(state)

                if predicted_tag == true_tag:
                    split_accs[true_tag].append(1)
                else:
                    split_accs[true_tag].append(0)

            assert len(states) == len(true_tags) == len(predicted_tags)
            
            #Calculate accuracy between predicted_tags and true_tags
            accuracy = np.mean(np.array(true_tags) == np.array(predicted_tags))
            for ii in range(4):
                if len(split_accs[ii]) > 0:
                    accu_split[ii].append(np.mean(split_accs[ii]))
                    cnt_split[ii].append(len(split_accs[ii]))
                else:
                    accu_split[ii].append(0)
                    cnt_split[ii].append(0)

            results.append(accuracy)

        # print(u, ", ".join(f"{x:.2f}" for x in results))
        final_results = np.add(final_results, results)
        for ii in range(4):            
            final_split_accu[ii] = np.add(final_split_accu[ii], accu_split[ii])
            final_cnt[ii] = np.add(final_cnt[ii], cnt_split[ii])

    final_results /= len(users)
    for ii in range(5):            
        final_split_accu[ii] /= len(users)
        final_cnt[ii] /= len(users)
    
    print('HMM', ", ".join(f"{x:.2f}" for x in final_results))
    for ii in range(5):
        print("Action ", ii, ", ".join(f"{x:.2f}" for x in final_split_accu[ii]))

    # for ii in range(5):
    #     print("Action ", ii, ", ".join(f"{x:.2f}" for x in final_cnt[ii]))

# ------ birdstrikes1 -------
# 1 0.50, 0.44, 0.43, 0.33, 0.20, 0.25, 0.33, 0.25, 0.50
# 5 0.41, 0.38, 0.42, 0.36, 0.38, 0.42, 0.40, 0.38, 0.57
# 9 0.46, 0.42, 0.40, 0.33, 0.29, 0.17, 0.20, 0.33, 0.00
# 109 0.24, 0.21, 0.20, 0.13, 0.15, 0.19, 0.25, 0.31, 0.29
# 13 0.57, 0.52, 0.46, 0.45, 0.35, 0.21, 0.00, 0.00, 0.00
# 25 0.43, 0.45, 0.46, 0.51, 0.51, 0.52, 0.54, 0.50, 0.50
# 29 0.16, 0.11, 0.12, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00
# 33 0.55, 0.61, 0.68, 0.63, 0.55, 0.44, 0.29, 0.44, 0.60
# 37 0.51, 0.49, 0.56, 0.63, 0.66, 0.71, 0.72, 0.58, 0.50
# 53 0.72, 0.71, 0.70, 0.66, 0.67, 0.74, 0.80, 0.90, 1.00
# 57 0.31, 0.35, 0.29, 0.31, 0.37, 0.29, 0.33, 0.42, 0.50
# 61 0.16, 0.17, 0.19, 0.20, 0.21, 0.27, 0.13, 0.07, 0.12
# 73 0.55, 0.59, 0.53, 0.45, 0.36, 0.45, 0.41, 0.40, 0.25
# 77 0.12, 0.12, 0.14, 0.17, 0.20, 0.25, 0.33, 0.00, 0.00
# 81 0.22, 0.25, 0.29, 0.29, 0.35, 0.31, 0.42, 0.38, 0.50
# 85 0.32, 0.29, 0.27, 0.31, 0.27, 0.33, 0.43, 0.60, 1.00
# 97 0.30, 0.34, 0.39, 0.39, 0.46, 0.33, 0.38, 0.36, 0.50

# HMM 0.38, 0.38, 0.38, 0.37, 0.35, 0.35, 0.35, 0.35, 0.40

# Action  0 0.31, 0.32, 0.32, 0.30, 0.21, 0.15, 0.12, 0.10, 0.07
# Action  1 0.73, 0.72, 0.70, 0.67, 0.67, 0.68, 0.67, 0.61, 0.62
# Action  2 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
# Action  3 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
# Action  4 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00

# Action  0 26.24, 23.18, 20.12, 16.71, 13.12, 9.76, 7.24, 4.59, 2.29
# Action  1 12.06, 10.71, 9.47, 8.76, 8.18, 7.41, 6.00, 4.29, 2.35
# Action  2 0.76, 0.76, 0.76, 0.65, 0.59, 0.59, 0.41, 0.24, 0.06
# Action  3 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
# Action  4 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
# ------ weather1 -------
# 1 0.23, 0.20, 0.23, 0.26, 0.29, 0.36, 0.37, 0.54, 0.86
# 5 0.74, 0.77, 0.76, 0.74, 0.71, 0.68, 0.65, 0.62, 0.62
# 113 0.10, 0.08, 0.09, 0.11, 0.13, 0.11, 0.07, 0.11, 0.00
# 117 0.19, 0.17, 0.17, 0.14, 0.13, 0.08, 0.06, 0.08, 0.17
# 21 0.57, 0.60, 0.68, 0.77, 0.83, 0.79, 0.72, 0.67, 0.83
# 25 0.21, 0.16, 0.18, 0.21, 0.26, 0.32, 0.29, 0.00, 0.00
# 29 0.90, 0.89, 0.88, 0.90, 0.88, 0.86, 0.82, 1.00, 1.00
# 41 0.21, 0.14, 0.03, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
# 45 0.34, 0.32, 0.36, 0.38, 0.33, 0.36, 0.45, 0.43, 0.50
# 53 0.16, 0.12, 0.12, 0.13, 0.16, 0.20, 0.26, 0.27, 0.12
# 65 0.09, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
# 69 0.35, 0.35, 0.34, 0.41, 0.37, 0.36, 0.38, 0.36, 0.33
# 73 0.24, 0.19, 0.21, 0.19, 0.21, 0.26, 0.27, 0.28, 0.11
# 77 0.17, 0.19, 0.22, 0.25, 0.25, 0.31, 0.20, 0.00, 0.00
# 93 0.20, 0.19, 0.21, 0.20, 0.18, 0.21, 0.20, 0.29, 0.25
# 97 0.19, 0.22, 0.25, 0.29, 0.35, 0.43, 0.48, 0.71, 0.86

# HMM 0.33, 0.31, 0.32, 0.34, 0.34, 0.35, 0.35, 0.36, 0.38

# Action  0 0.14, 0.14, 0.15, 0.14, 0.14, 0.13, 0.13, 0.13, 0.13
# Action  1 0.80, 0.80, 0.79, 0.73, 0.73, 0.73, 0.73, 0.66, 0.60
# Action  2 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
# Action  3 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
# Action  4 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00

# Action  0 33.20, 30.32, 26.32, 21.79, 17.44, 13.55, 10.08, 6.66, 3.21
# Action  1 11.57, 9.48, 8.84, 8.17, 7.32, 6.28, 4.88, 3.52, 1.96
# Action  2 0.36, 0.36, 0.36, 0.35, 0.35, 0.35, 0.28, 0.08, 0.00
# Action  3 0.06, 0.06, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
# Action  4 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00

# ------ faa1 -------
# 9 0.73, 0.72, 0.68, 0.63, 0.55, 0.52, 0.39, 0.42, 0.33
# 109 0.33, 0.25, 0.26, 0.24, 0.26, 0.18, 0.24, 0.36, 0.33
# 13 0.69, 0.75, 0.80, 0.89, 0.86, 1.00, 1.00, 1.00, 1.00
# 21 0.93, 0.97, 0.97, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00
# 33 0.12, 0.14, 0.15, 0.15, 0.18, 0.23, 0.18, 0.18, 0.17
# 37 0.43, 0.37, 0.42, 0.46, 0.41, 0.38, 0.35, 0.23, 0.29
# 41 0.28, 0.25, 0.14, 0.17, 0.10, 0.12, 0.17, 0.25, 0.00
# 45 0.89, 0.88, 0.86, 0.83, 0.80, 1.00, 1.00, 1.00, 1.00
# 57 0.12, 0.14, 0.16, 0.19, 0.22, 0.27, 0.12, 0.09, 0.00
# 65 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
# 69 0.37, 0.36, 0.40, 0.41, 0.38, 0.36, 0.38, 0.36, 0.29
# 81 0.53, 0.50, 0.44, 0.48, 0.44, 0.50, 0.36, 0.29, 0.25
# 85 0.33, 0.27, 0.30, 0.38, 0.43, 0.50, 0.75, 1.00, 1.00
# 89 0.37, 0.39, 0.38, 0.37, 0.30, 0.22, 0.14, 0.11, 0.20
# 93 0.71, 0.73, 0.77, 0.73, 0.68, 0.64, 0.79, 1.00, 1.00

# HMM 0.48, 0.47, 0.47, 0.48, 0.46, 0.49, 0.48, 0.51, 0.48

# Action  0 0.41, 0.39, 0.36, 0.35, 0.34, 0.34, 0.35, 0.36, 0.36
# Action  1 0.57, 0.58, 0.58, 0.60, 0.58, 0.58, 0.57, 0.58, 0.44
# Action  2 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
# Action  3 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
# Action  4 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00

# Action  0 27.21, 24.35, 20.62, 17.12, 14.10, 11.37, 9.01, 6.38, 3.35
# Action  1 9.10, 8.03, 7.66, 6.94, 5.75, 4.69, 3.26, 1.83, 0.93
# Action  2 0.62, 0.62, 0.62, 0.62, 0.56, 0.49, 0.35, 0.27, 0.07
# Action  3 0.14, 0.14, 0.14, 0.13, 0.13, 0.13, 0.07, 0.00, 0.00
# Action  4 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00