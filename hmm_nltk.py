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

def encoding(dataset, file):
        all_attributes = []
        all_actions = set()

        data_dataframe = []
        cat = Categorizing(dataset)

        if dataset == 'birdstrikes1':
            states = {"Damage":0, "Incident":1, "Aircraft":2, "Environment":3, "Wildlife":4, "Misc":5}
            # self.states = {"Damage":0, "Incident":1, "Aircraft":2, "Environment":3}
        elif dataset == 'weather1':
            states = {"Temperature":0, "Location":1, "Metadata":2, "CommonPhenomena":3, "Fog":4, "Extreme":5, "Misc":6, "Misc2":7}
        else: # 'FAA1'
            states = {"Performance":0, "Airline":1, "Location":2, "Status":3, "Misc":4}

        unique_attributes = set(states.keys())
        # attribute_encoder = LabelEncoder().fit(list(unique_attributes))
        all_combinations = []
        for r in range(1, 6):  # From 1 attribute up to 5
            for combo in combinations(unique_attributes, r):
                sorted_combo = tuple(sorted(combo))  # Sort the combination to ensure consistent ordering
                all_combinations.append(sorted_combo)
                # print(sorted_combo)
        # pdb.set_trace()
        # Encode all possible combinations
        attribute_encoder = LabelEncoder().fit([''.join(combo) for combo in all_combinations])

        for line in file:
            parts = line.split(",")
            parts[1] = parts[1].replace(";", ",")
            list_part = ast.literal_eval(parts[1])
            if len(list_part) == 0:
                continue
            high_level_attrs = cat.get_category(list_part, dataset)
            # print(high_level_attrs)
            x = attribute_encoder.transform(high_level_attrs)
            encoded_interaction = ''.join(str(p) for p in sorted(x))
            data_dataframe.append([int(parts[0]), encoded_interaction])

        data = pd.DataFrame(data_dataframe, columns=['Action', 'Attribute'])
        
        data['Encoded_Attribute'] = attribute_encoder.fit_transform(data['Attribute'])
        action_encoder = LabelEncoder()
        data['Action'] = action_encoder.fit_transform(data['Action'])
        # pdb.set_trace()
        # print(data['Action'])
        return data

 
obj = read_data()
obj.create_connection(r"Tableau.db")
r = reward()
final_results = np.zeros(9, dtype = float)
for d in r.datasets:
    users = obj.get_user_list_for_dataset(d)
    #getting all users data for model initialization
    cleaned_data = []
    
    for user in users:
        u = user[0]
        data = obj.merge2(d, u)
        raw_states, raw_actions, mem_reward = r.generate(data, d)
        sequences = []
        for i in range(len(raw_states)):
            temp = "["
            for idx, s in enumerate(raw_states[i]):
                if idx == len(raw_states[i]) - 1:
                    temp += s
                else:
                    temp += s + ";"
            temp += "]"
            sequences.append((temp, raw_actions[i]))
            
        threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = []
        for t in threshold:
            split = int(len(sequences) * t)
            runs = []
            for k in range(10):
                trainer = nltk.HiddenMarkovModelTagger.train([sequences[:split]])
                accuracy = trainer.accuracy([sequences[split:]])
                runs.append(accuracy)
            results.append(np.mean(runs))
        final_results = np.add(final_results, results)

    final_results /= len(users)
    print(d, ", ".join(f"{x:.2f}" for x in final_results))
        
    #     # # Update the model with observations (you'd loop through your observations to update)
    #     threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #     results = []
    #     for t in threshold:
    #         accu = []
    #         split = int(len(data2) * t)
    #         for idx in range(split):
    #             # print(data2.iloc[idx])
    #             model.update(data2.iloc[idx])
    #         for idx in range(split, len(data2)):
    #             predicted_action = model.predict_next_action(data2.iloc[idx])
    #             if predicted_action == data2['Action'][idx]:
    #                 accu.append(1)
    #             else:
    #                 accu.append(0)
    #             model.update(data2.iloc[idx])
    #         results.append(np.mean(accu))
    #         # print(t, np.mean(results))
    #     final_results = np.add(final_results, results)
    #     # pdb.set_trace()
    #     # print(len(raw_states), len(raw_actions), len(mem_reward))
    # final_results /= len(users)
    # print(d, ", ".join(f"{x:.2f}" for x in final_results))
