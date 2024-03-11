import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        obj.create_connection(r"/nfs/hpc/share/sahasa/Tableau_MDP/Tableau.db")
        user_list = obj.get_user_list_for_dataset(d)
        for user in user_list:
            interaction_data, timestamps = obj.merge2(d, user[0])
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
            
            denominator = 0
            numerator = np.zeros(5, dtype=float)
            for i in range(1, mid_idx):
                denominator += 1
                diff = 0
                for items in raw_interactions[i-1]:
                    if items not in raw_interactions[i]:
                        diff += 1 
                diff = min(diff, 4)
                numerator[diff] += 1   
            numerator /= denominator
            numerator = np.round(numerator, 2)
            new_data.append([user[0], d, numerator[0], numerator[1], numerator[2], numerator[3], numerator[4], 'First'])

            denominator = 0
            numerator = np.zeros(5, dtype=float)
            for i in range(mid_idx, len(raw_actions)):
                denominator += 1
                diff = 0
                for items in raw_interactions[i-1]:
                    if items not in raw_interactions[i]:
                        diff += 1 
                diff = min(diff, 4)
                numerator[diff] += 1   
            numerator /= denominator
            new_data.append([user[0], d, numerator[0], numerator[1], numerator[2], numerator[3], numerator[4], 'Second'])
   
    df = pd.DataFrame(new_data, columns = ['Users', 'Datasets', 'Keep', 'Modify1', 'Modify2', 'Modify3', 'Modify4', 'Phase'])
    print(df)
    df.to_csv('lme_data_tableau_new_action_aggregated.csv', index=False)
            # for idx, items in enumerate(interaction_data):
            #     print(items, timestamps[idx])
