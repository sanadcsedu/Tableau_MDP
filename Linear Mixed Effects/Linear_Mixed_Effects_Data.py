import os
import fnmatch
import pdb
from collections import defaultdict
import glob
import pandas as pd
import numpy as np
from Categorizing_v3 import utilities
from Categorizing_v4 import Categorizing
import environment5 as environment5
import misc 
from read_data_lme import read_data_lme
from datetime import datetime, timedelta
import pandas as pd

if __name__ == "__main__":
    datasets = ['birdstrikes1', 'weather1', 'faa1']
    new_data = []
    for d in datasets:
        obj = read_data_lme()
        obj.create_connection(r"Tableau.db")
        user_list = obj.get_user_list_for_dataset(d)
        for user in user_list:
            interaction_data, timestamps = obj.merge2(d, user[0])
            # print("-------- " + str(user[0]) + " ------- " + d + " ------- ")
            u = utilities()

            start_time = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S.%f")
            end_time = datetime.strptime(timestamps[-1], "%Y-%m-%d %H:%M:%S.%f")
            mid_time = start_time + (end_time - start_time) / 2
            timestamps_dt = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f") for ts in timestamps]
            for idx, ts in enumerate(timestamps_dt):
                if ts > mid_time:
                    mid_idx = idx 
                    break
            # print(str(mid_idx) + " - " + str(len(timestamps)))

            raw_interactions, _, raw_actions, _ = u.generate(interaction_data, d)
            # print(str(len(timestamps)) + " - " + str(len(raw_interactions)) + " - " + str(len(raw_actions)))
            # for idx, actions in enumerate(raw_actions):
            #     print(raw_interactions[idx], actions, timestamps[idx])
            denominator = 0
            numerator_add = 0
            numerator_remove = 0
            numerator_keep = 0
            for i in range(mid_idx):
                denominator += 1
                if raw_actions[i] == 'Add':
                    numerator_add += 1
                elif raw_actions[i] == 'Remove':
                    numerator_remove += 1
                else:
                    numerator_keep += 1
            prob_add = round(numerator_add / denominator, 2)
            prob_remove = round(numerator_remove / denominator, 2)    
            prob_keep = round(numerator_keep / denominator, 2)
            #First Phase Probabilities
            new_data.append([user[0], d, prob_add, prob_remove, prob_keep, 'First'])

            denominator = 0
            numerator_add = 0
            numerator_remove = 0
            numerator_keep = 0
            for i in range(mid_idx, len(raw_actions)):
                denominator += 1
                if raw_actions[i] == 'Add':
                    numerator_add += 1
                elif raw_actions[i] == 'Remove':
                    numerator_remove += 1
                else:
                    numerator_keep += 1
            prob_add = round(numerator_add / denominator, 2)
            prob_remove = round(numerator_remove / denominator, 2)    
            prob_keep = round(numerator_keep / denominator, 2)
            #Second Phase Probabilities
            new_data.append([user[0], d, prob_add, prob_remove, prob_keep, 'Second'])
        
    df = pd.DataFrame(new_data, columns = ['Users', 'Datasets', 'Add', 'Remove', 'Keep', 'Phase'])
    # print(df)
    df.to_csv('lme_data_tableau_v1.csv', index=False)
            # for idx, items in enumerate(interaction_data):
            #     print(items, timestamps[idx])
