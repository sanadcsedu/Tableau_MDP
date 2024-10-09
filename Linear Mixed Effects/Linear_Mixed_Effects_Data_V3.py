import os
import fnmatch
import pdb
from collections import defaultdict
import glob
import pandas as pd
import numpy as np
from Categorizing_v3 import utilities
from Categorizing_v4 import Categorizing
import environment5_old as environment5_old
import misc 
from read_data_lme import read_data_lme
from datetime import datetime, timedelta
import pandas as pd

if __name__ == "__main__":
    datasets = ['birdstrikes1', 'weather1', 'faa1']
    tasks = ['t1', 't2', 't3', 't4']

    new_data = []
    for d in datasets:
        obj = read_data_lme()
        obj.create_connection(r"Tableau.db")
        user_list = obj.get_user_list_for_dataset(d)
        for user in user_list:
            for t in tasks:
                interaction_data, timestamps = obj.merge_v2(user[0], d, t)
                # print("-------- " + str(user[0]) + " ------- " + d + " ------- ")
                u = utilities()

                raw_interactions, _, raw_actions, _ = u.generate(interaction_data, d)
                
                denominator = 0
                numerator_add = 0
                numerator_remove = 0
                numerator_keep = 0
                for i in range(len(raw_actions)):
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
                new_data.append([user[0], d, prob_add, prob_remove, prob_keep, t])

                
    df = pd.DataFrame(new_data, columns = ['Users', 'Datasets', 'Add', 'Remove', 'Keep', 'Tasks'])
    df.to_csv('lme_data_tableau_v3.csv', index=False)
 