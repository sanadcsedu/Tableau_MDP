import environment5
import numpy as np
from collections import defaultdict
import json
import pandas as pd
import random
import multiprocessing

eps=1e-35
class Greedy:
    def __init__(self):
        """Initializes the Greedy model."""
        self.freq = defaultdict(lambda: defaultdict(float))
        # self.reward = defaultdict(lambda: defaultdict(float))

    def GreedyDriver(self, env, thres):
        length = len(env.mem_action)
        threshold = int(length * thres)
        for i in range(threshold):
            self.freq[env.mem_states[i]][env.mem_action[i]] += 1
            # self.reward[env.mem_states[i]][env.mem_action[i]] += env.mem_reward[i]+eps

        accuracy = 0
        denom = 0
        insight = defaultdict(list)

        for i in range(threshold, length):
            denom += 1

            actions = {'same':0, 'modify-1':1, 'modify-2':2, 'modify-3':3, 'modify-4':4}
            # frequencies = list(self.freq[env.mem_states[i]].values())

            try:    #Finding the most rewarding action in the current state
                # _max = random.choices(list(actions.values()), weights=frequencies, k=1)[0]
                freqs = []
                for a, v in actions.items():
                    if v in self.freq[env.mem_states[i]]:
                        freqs.append(self.freq[env.mem_states[i]][v])
                    else:
                        freqs.append(1)
                _max = random.choices(list(actions.values()), weights=freqs, k=1)[0]

            except ValueError: #Randomly picking an action if it was used previously in current state 
                _max= random.choice([0, 1, 2, 3, 4])
               
            if _max == env.mem_action[i]:
                accuracy += 1
                insight[env.mem_action[i]].append(1)
            else:
                insight[env.mem_action[i]].append(0)

            self.freq[env.mem_states[i]][env.mem_action[i]] += 1

        accuracy /= denom
        self.freq.clear()
        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.mean(values))

        return accuracy, granular_prediction

class run_Greedy:
    def __inti__(self):
        pass

    def run_experiment(self, user_list, dataset, hyperparam_file, result_queue, info, info_split_accu, info_split_cnt):
        # Load hyperparameters from JSON file
        with open(hyperparam_file) as f:
            hyperparams = json.load(f)
        threshold = hyperparams['threshold']

        # Create result DataFrame with columns for relevant statistics
        final_accu = np.zeros(9, dtype=float)
        final_cnt = np.zeros((5, 9), dtype = float)
        final_split_accu = np.zeros((5, 9), dtype = float)
        for u in user_list:
            accu = []
            accu_split = [[] for _ in range(5)]
            cnt_split = [[] for _ in range(5)]
            for thres in threshold:
                avg_accu = []
                split_accs = [[] for _ in range(5)]

                for _ in range(5):
                    env.process_data(dataset, u[0], thres, 'Greedy')
                    # print(env.mem_states)
                    obj = Greedy()
                    temp_accuracy, gp = obj.GreedyDriver(env, thres)
                    avg_accu.append(temp_accuracy)
                    env.reset(True, False)

                    for key, val in gp.items():
                        split_accs[key].append(val[1])
                    
                accu.append(np.mean(avg_accu))
                for ii in range(5):
                    if len(split_accs[ii]) > 0:
                        accu_split[ii].append(np.mean(split_accs[ii]))
                        cnt_split[ii].append(gp[ii][0])
                    else:
                        accu_split[ii].append(0)
                        cnt_split[ii].append(0)

            print(u[0], ", ".join(f"{x:.2f}" for x in accu))
            final_accu = np.add(final_accu, accu)
            for ii in range(5):            
                final_split_accu[ii] = np.add(final_split_accu[ii], accu_split[ii])
                final_cnt[ii] = np.add(final_cnt[ii], cnt_split[ii])

        final_accu /= len(user_list)
        for ii in range(5):            
            final_split_accu[ii] /= len(user_list)
            final_cnt[ii] /= len(user_list)
        
        # print(final_accu)
        
        result_queue.put(final_accu)
        info_split_accu.put(final_split_accu)
        info_split_cnt.put(final_cnt)


if __name__ == '__main__':
    env = environment5.environment5()
    datasets = env.datasets
    for d in datasets:
        print("------", d, "-------")
        env.obj.create_connection(r"Tableau.db")
        user_list = env.obj.get_user_list_for_dataset(d)

        obj2 = run_Greedy()

        result_queue = multiprocessing.Queue()
        info = multiprocessing.Queue()
        info_split = multiprocessing.Queue()
        info_split_cnt = multiprocessing.Queue() 
    
        p1 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[:4], d, 'sampled_hyper_params.json', result_queue, info, info_split, info_split_cnt))
        p2 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[4:8], d, 'sampled_hyper_params.json', result_queue, info, info_split, info_split_cnt))
        p3 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[8:12], d, 'sampled_hyper_params.json', result_queue, info, info_split, info_split_cnt))
        p4 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[12:], d, 'sampled_hyper_params.json', result_queue, info, info_split, info_split_cnt))
        
        split_final = np.zeros((5, 9), dtype = float)
        split_final_cnt = np.zeros((5, 9), dtype = float)

        p1.start()
        p2.start()
        p3.start()
        p4.start()
        final_result = np.zeros(9, dtype = float)
        p1.join()
        final_result = np.add(final_result, result_queue.get())
        split_final = np.add(split_final, info_split.get())
        split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())
        # print(split_final_cnt)
        p2.join()
        final_result = np.add(final_result, result_queue.get())
        split_final = np.add(split_final, info_split.get())
        split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())

        p3.join()
        final_result = np.add(final_result, result_queue.get())
        split_final = np.add(split_final, info_split.get())
        split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())

        p4.join()
        final_result = np.add(final_result, result_queue.get())
        split_final = np.add(split_final, info_split.get())
        split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())

        final_result /= 4
        split_final /= 4
        split_final_cnt /= 4

        print("Bayesian ", ", ".join(f"{x:.2f}" for x in final_result))

        for ii in range(5):
            print("Action ", ii, ", ".join(f"{x:.2f}" for x in split_final[ii]))

        for ii in range(5):
            print("Action ", ii, ", ".join(f"{x:.2f}" for x in split_final_cnt[ii]))

# ------ birdstrikes1 -------
# 1 0.25, 0.31, 0.27, 0.33, 0.30, 0.28, 0.31, 0.33, 0.40
# 13 0.36, 0.32, 0.29, 0.33, 0.26, 0.22, 0.25, 0.24, 0.20
# 37 0.26, 0.33, 0.33, 0.36, 0.36, 0.35, 0.30, 0.40, 0.47
# 73 0.36, 0.38, 0.38, 0.34, 0.29, 0.34, 0.18, 0.22, 0.27
# 5 0.32, 0.32, 0.29, 0.34, 0.28, 0.30, 0.30, 0.28, 0.20
# 25 0.28, 0.26, 0.28, 0.27, 0.27, 0.29, 0.29, 0.17, 0.23
# 53 0.30, 0.28, 0.33, 0.31, 0.36, 0.31, 0.24, 0.33, 0.40
# 77 0.42, 0.45, 0.38, 0.44, 0.44, 0.41, 0.46, 0.45, 0.29
# 29 0.38, 0.34, 0.41, 0.41, 0.26, 0.40, 0.39, 0.45, 0.40
# 9 0.28, 0.33, 0.31, 0.34, 0.36, 0.31, 0.33, 0.33, 0.47
# 57 0.29, 0.31, 0.25, 0.31, 0.27, 0.28, 0.28, 0.20, 0.18
# 81 0.30, 0.28, 0.26, 0.29, 0.26, 0.37, 0.32, 0.22, 0.20
# 33 0.47, 0.46, 0.42, 0.45, 0.44, 0.38, 0.38, 0.29, 0.40
# 109 0.36, 0.40, 0.44, 0.46, 0.48, 0.55, 0.50, 0.46, 0.24
# 85 0.26, 0.22, 0.28, 0.32, 0.31, 0.25, 0.36, 0.44, 0.44
# 61 0.49, 0.45, 0.49, 0.43, 0.37, 0.44, 0.42, 0.59, 0.60
# 97 0.38, 0.37, 0.35, 0.32, 0.30, 0.32, 0.24, 0.28, 0.20

# Bayesian  0.34, 0.34, 0.34, 0.36, 0.33, 0.34, 0.33, 0.33, 0.33

# Action  0 0.39, 0.41, 0.41, 0.42, 0.38, 0.40, 0.37, 0.34, 0.29
# Action  1 0.25, 0.22, 0.21, 0.24, 0.23, 0.25, 0.20, 0.23, 0.26
# Action  2 0.27, 0.33, 0.32, 0.33, 0.33, 0.27, 0.20, 0.14, 0.17
# Action  3 0.08, 0.06, 0.07, 0.07, 0.07, 0.06, 0.05, 0.04, 0.02
# Action  4 0.11, 0.08, 0.12, 0.13, 0.06, 0.06, 0.08, 0.05, 0.04
# Action  0 41.15, 36.29, 31.68, 27.31, 21.81, 17.39, 12.78, 8.65, 4.08
# Action  1 24.84, 22.02, 19.44, 16.64, 15.09, 12.43, 10.10, 7.12, 4.10
# Action  2 7.44, 6.80, 6.12, 5.04, 3.90, 3.08, 2.19, 1.21, 0.55
# Action  3 2.10, 1.93, 1.61, 1.49, 1.26, 0.96, 0.59, 0.41, 0.23
# Action  4 3.20, 2.85, 2.50, 2.04, 1.66, 1.32, 0.86, 0.40, 0.19

# ------ weather1 -------
# 45 0.31, 0.33, 0.31, 0.30, 0.32, 0.39, 0.38, 0.37, 0.34
# 21 0.39, 0.37, 0.39, 0.36, 0.40, 0.40, 0.54, 0.44, 0.45
# 1 0.36, 0.44, 0.42, 0.40, 0.43, 0.48, 0.46, 0.39, 0.27
# 73 0.47, 0.43, 0.47, 0.46, 0.52, 0.50, 0.48, 0.58, 0.51
# 53 0.43, 0.44, 0.47, 0.43, 0.41, 0.42, 0.43, 0.46, 0.37
# 77 0.48, 0.41, 0.38, 0.39, 0.40, 0.46, 0.41, 0.49, 0.50
# 25 0.35, 0.40, 0.39, 0.47, 0.39, 0.44, 0.39, 0.56, 0.43
# 5 0.48, 0.50, 0.42, 0.48, 0.46, 0.52, 0.42, 0.41, 0.38
# 65 0.36, 0.25, 0.41, 0.39, 0.35, 0.38, 0.30, 0.42, 0.40
# 93 0.36, 0.39, 0.35, 0.30, 0.31, 0.28, 0.31, 0.43, 0.37
# 29 0.54, 0.55, 0.57, 0.62, 0.68, 0.61, 0.62, 0.75, 0.74
# 113 0.35, 0.42, 0.35, 0.38, 0.39, 0.44, 0.37, 0.34, 0.40
# 69 0.35, 0.37, 0.34, 0.29, 0.31, 0.28, 0.31, 0.29, 0.33
# 97 0.45, 0.45, 0.43, 0.51, 0.50, 0.46, 0.39, 0.41, 0.22
# 41 0.39, 0.35, 0.39, 0.38, 0.45, 0.43, 0.47, 0.47, 0.38
# 117 0.35, 0.37, 0.41, 0.39, 0.38, 0.34, 0.37, 0.36, 0.62

# Bayesian  0.40, 0.40, 0.41, 0.41, 0.42, 0.43, 0.42, 0.45, 0.42

# Action  0 0.50, 0.51, 0.52, 0.51, 0.52, 0.53, 0.54, 0.57, 0.60
# Action  1 0.30, 0.27, 0.27, 0.27, 0.27, 0.25, 0.22, 0.28, 0.20
# Action  2 0.11, 0.10, 0.08, 0.10, 0.10, 0.04, 0.02, 0.03, 0.00
# Action  3 0.02, 0.02, 0.00, 0.02, 0.01, 0.00, 0.00, 0.00, 0.00
# Action  4 0.05, 0.08, 0.08, 0.07, 0.04, 0.06, 0.06, 0.04, 0.03
# Action  0 43.94, 40.69, 36.19, 30.81, 25.50, 20.19, 14.69, 10.25, 4.75
# Action  1 20.06, 16.50, 14.44, 12.50, 10.88, 9.56, 8.06, 5.12, 3.12
# Action  2 5.75, 4.94, 4.00, 3.38, 2.94, 2.12, 1.06, 0.56, 0.19
# Action  3 0.88, 0.81, 0.62, 0.62, 0.31, 0.25, 0.12, 0.00, 0.00
# Action  4 5.50, 4.75, 4.19, 3.62, 2.69, 1.94, 1.69, 1.38, 0.88

# ------ faa1 -------
# 33 0.39, 0.35, 0.38, 0.34, 0.30, 0.31, 0.34, 0.33, 0.42
# 57 0.34, 0.30, 0.26, 0.26, 0.25, 0.25, 0.31, 0.31, 0.29
# 9 0.41, 0.43, 0.41, 0.37, 0.26, 0.27, 0.29, 0.28, 0.28
# 85 0.32, 0.37, 0.39, 0.41, 0.38, 0.51, 0.42, 0.29, 0.16
# 65 0.36, 0.37, 0.39, 0.30, 0.20, 0.23, 0.35, 0.40, 0.30
# 109 0.36, 0.36, 0.38, 0.38, 0.38, 0.40, 0.36, 0.39, 0.42
# 37 0.30, 0.28, 0.31, 0.31, 0.35, 0.25, 0.26, 0.30, 0.31
# 89 0.37, 0.33, 0.35, 0.40, 0.35, 0.39, 0.47, 0.41, 0.36
# 13 0.32, 0.34, 0.32, 0.32, 0.34, 0.45, 0.31, 0.38, 0.80
# 41 0.29, 0.33, 0.30, 0.23, 0.27, 0.20, 0.29, 0.31, 0.20
# 69 0.32, 0.35, 0.36, 0.31, 0.33, 0.31, 0.31, 0.29, 0.39
# 93 0.40, 0.35, 0.39, 0.44, 0.40, 0.43, 0.49, 0.55, 0.71
# 45 0.36, 0.44, 0.34, 0.34, 0.40, 0.32, 0.33, 0.46, 0.60
# 21 0.57, 0.61, 0.62, 0.60, 0.63, 0.68, 0.87, 0.86, 0.87
# 81 0.34, 0.39, 0.38, 0.41, 0.32, 0.26, 0.22, 0.19, 0.20

# Bayesian  0.36, 0.37, 0.37, 0.37, 0.35, 0.36, 0.38, 0.38, 0.42

# Action  0 0.44, 0.46, 0.45, 0.44, 0.41, 0.41, 0.43, 0.47, 0.44
# Action  1 0.25, 0.23, 0.24, 0.25, 0.24, 0.29, 0.30, 0.31, 0.16
# Action  2 0.31, 0.31, 0.34, 0.36, 0.41, 0.29, 0.24, 0.11, 0.04
# Action  3 0.12, 0.08, 0.08, 0.10, 0.04, 0.05, 0.01, 0.03, 0.00
# Action  4 0.12, 0.08, 0.12, 0.09, 0.10, 0.11, 0.09, 0.07, 0.04
# Action  0 39.88, 35.60, 30.94, 25.77, 20.44, 15.88, 12.23, 9.31, 5.54
# Action  1 17.38, 15.85, 14.54, 12.88, 11.35, 9.90, 7.65, 4.12, 1.79
# Action  2 6.23, 5.33, 4.17, 3.69, 3.15, 2.38, 1.50, 0.94, 0.21
# Action  3 1.81, 1.40, 1.27, 1.12, 1.04, 0.65, 0.40, 0.21, 0.06
# Action  4 4.73, 4.00, 3.67, 3.15, 2.96, 2.50, 1.83, 1.12, 0.56