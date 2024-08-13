# import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import environment5
# import plotting
from collections import Counter,defaultdict
import json
from pathlib import Path
import glob
# from tqdm import tqdm 
import os 
import multiprocessing

eps=1e-35
class Policy(nn.Module):
    def __init__(self,learning_rate,gamma,tau, dataset):
        super(Policy, self).__init__()
        self.data = []
        if dataset == 'birdstrikes1':
            self.fc1 = nn.Linear(6, 128)
            self.fc2 = nn.Linear(128, 5)
        elif dataset == 'faa1':
            self.fc1 = nn.Linear(4, 128)
            self.fc2 = nn.Linear(128, 5)
        else:
            self.fc1 = nn.Linear(7, 128)
            self.fc2 = nn.Linear(128, 5)
    
        self.gamma=gamma
        self.temperature = tau
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x / self.temperature
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + self.gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []


class Reinforce():
    def __init__(self,env,learning_rate,gamma,tau, dataset):
        self.env = env
        self.learning_rate, self.gamma, self.temperature = learning_rate, gamma, tau
        self.pi = Policy(self.learning_rate, self.gamma,self.temperature, dataset)

    def train(self):
        
        all_predictions=[]
        for _ in range(5):
            s = self.env.reset(all = False, test = False)
            s=np.array(s)
            done = False
            actions =[]
            predictions=[]
            while not done:
                prob = self.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                actions.append(a)
                s_prime, r, done, info, _ = self.env.step(s,a,False)
                predictions.append(info)


                self.pi.put_data((r, prob[a]))

                s = s_prime

            self.pi.train_net()
            all_predictions.append(np.mean(predictions))
        # print("############ Train Accuracy :{},".format(np.mean(all_predictions)))
        return self.pi, (np.mean(predictions)) #return last train_accuracy


    def test(self,policy):
        test_accuracies = []
        
        for _ in range(1):
            s = self.env.reset(all=False, test=True)
            done = False
            predictions = []
            # actions = []
            insight = defaultdict(list)

            while not done:
                prob = policy(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                # actions.append(a)
                s_prime, r, done, info, ground_action  = self.env.step(s, a, True)
                predictions.append(info)
                
                insight[ground_action].append(info)

                policy.put_data((r, prob[a]))

                s = s_prime
                self.pi.train_net()

            test_accuracies.append(np.mean(predictions))
        
        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.mean(values))

        return np.mean(test_accuracies), granular_prediction
    
class run_reinforce:
    def __init__(self):
        pass


    def run_experiment(self, user_list,dataset,hyperparam_file, result_queue, info, info_split_accu, info_split_cnt):
        # Load hyperparameters from JSON file
        with open(hyperparam_file) as f:
            hyperparams = json.load(f)

        # Extract hyperparameters from JSON file
        learning_rates = hyperparams['learning_rates']
        gammas = hyperparams['gammas']
        temperatures = hyperparams['temperatures']

        # aggregate_plotter = plotting.plotter(None)
        final_accu = np.zeros(9, dtype=float)
        final_cnt = np.zeros((5, 9), dtype = float)
        final_split_accu = np.zeros((5, 9), dtype = float)
        for user in user_list:
            # Extract user-specific threshold values
            threshold_h = hyperparams['threshold']            
            accu = []
            accu_split = [[] for _ in range(5)]
            cnt_split = [[] for _ in range(5)]

            env = environment5.environment5()
            for thres in threshold_h:
                max_accu = -1
                best_learning_rate = 0
                best_gamma = 0
                best_agent=None
                best_policy=None
                best_temp=0
                env.process_data(dataset, user[0], thres, 'Reinforce')            
                for learning_rate in learning_rates:
                    for gamma in gammas:
                        for temp in temperatures:
                            agent = Reinforce(env,learning_rate,gamma,temp, dataset)
                            policy,accuracies = agent.train()

                            if accuracies > max_accu:
                                max_accu=accuracies
                                best_learning_rate=learning_rate
                                best_gamma=gamma
                                best_agent = agent
                                best_policy = policy
                                best_temp=temp
                
                test_accs = []
                split_accs = [[] for _ in range(5)]
                
                for _ in range(5):
                    test_agent = best_agent
                    test_model = best_policy
                    temp_accuracy, gp = test_agent.test(test_model)
                    test_accs.append(temp_accuracy)

                    for key, val in gp.items():
                        # print(key, val)
                        split_accs[key].append(val[1])
                
                test_accuracy = np.mean(test_accs)
                accu.append(test_accuracy)
                env.reset(True, False)

                for ii in range(5):
                    if len(split_accs[ii]) > 0:
                        # print("action: {}, count: {}, accuracy:{}".format(ii, gp[ii][0], np.mean(split_accs[ii])))
                        accu_split[ii].append(np.mean(split_accs[ii]))
                        cnt_split[ii].append(gp[ii][0])
                    else:
                        accu_split[ii].append(0)
                        cnt_split[ii].append(0)

            print(user[0], ", ".join(f"{x:.2f}" for x in accu))

            final_accu = np.add(final_accu, accu)
            for ii in range(5):            
                final_split_accu[ii] = np.add(final_split_accu[ii], accu_split[ii])
                final_cnt[ii] = np.add(final_cnt[ii], cnt_split[ii])

        final_accu /= len(user_list)
        for ii in range(5):            
            final_split_accu[ii] /= len(user_list)
            final_cnt[ii] /= len(user_list)
        
        result_queue.put(final_accu)
        info_split_accu.put(final_split_accu)
        info_split_cnt.put(final_cnt)

    def get_user_name(self, raw_fname):
        user = Path(raw_fname).stem.split('-')[0]
        return user


if __name__ == '__main__':
    env = environment5.environment5()
    datasets = env.datasets
    for d in datasets:
        print("------", d, "-------")
        env.obj.create_connection(r"Tableau.db")
        user_list = env.obj.get_user_list_for_dataset(d)

        obj2 = run_reinforce()

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

        print("Reinforce ", ", ".join(f"{x:.2f}" for x in final_result))

        for ii in range(5):
            print("Action ", ii, ", ".join(f"{x:.2f}" for x in split_final[ii]))

        for ii in range(5):
            print("Action ", ii, ", ".join(f"{x:.2f}" for x in split_final_cnt[ii]))

# ------ birdstrikes1 -------
# 1 0.21, 0.21, 0.17, 0.16, 0.19, 0.28, 0.40, 0.83, 1.00
# 13 0.33, 0.29, 0.17, 0.40, 0.42, 0.43, 0.36, 0.29, 0.33
# 37 0.25, 0.56, 0.31, 0.19, 0.69, 0.71, 0.76, 0.77, 0.83
# 73 0.13, 0.60, 0.56, 0.60, 0.41, 0.48, 0.38, 0.63, 0.40
# 5 0.26, 0.31, 0.54, 0.34, 0.54, 0.51, 0.59, 0.56, 0.50
# 53 0.23, 0.52, 0.48, 0.47, 0.40, 0.45, 0.67, 0.36, 1.00
# 25 0.39, 0.38, 0.44, 0.44, 0.44, 0.44, 0.50, 0.51, 0.50
# 77 0.68, 0.62, 0.59, 0.60, 0.58, 0.62, 0.65, 0.60, 0.33
# 57 0.18, 0.52, 0.32, 0.22, 0.60, 0.53, 0.54, 0.33, 0.29
# 29 0.04, 0.50, 0.03, 0.56, 0.43, 0.41, 0.44, 0.50, 0.50
# 9 0.30, 0.04, 0.23, 0.29, 0.31, 0.20, 0.43, 0.16, 0.22
# 81 0.46, 0.42, 0.41, 0.46, 0.42, 0.46, 0.33, 0.27, 0.20
# 61 0.69, 0.20, 0.37, 0.69, 0.65, 0.05, 0.80, 0.90, 0.89
# 109 0.58, 0.13, 0.00, 0.67, 0.17, 0.65, 0.67, 0.61, 0.62
# 85 0.23, 0.29, 0.25, 0.17, 0.11, 0.26, 0.15, 0.12, 1.00
# 33 0.17, 0.17, 0.66, 0.66, 0.61, 0.51, 0.46, 0.43, 0.36
# 97 0.22, 0.70, 0.68, 0.62, 0.67, 0.65, 0.52, 0.50, 0.43

# Reinforce  0.31, 0.37, 0.36, 0.44, 0.45, 0.45, 0.52, 0.50, 0.56

# Action  0 0.40, 0.44, 0.49, 0.70, 0.76, 0.74, 0.74, 0.71, 0.67
# Action  1 0.34, 0.39, 0.27, 0.30, 0.17, 0.21, 0.26, 0.27, 0.31
# Action  2 0.27, 0.12, 0.12, 0.01, 0.06, 0.01, 0.00, 0.00, 0.06
# Action  3 0.05, 0.05, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00
# Action  4 0.02, 0.03, 0.02, 0.01, 0.01, 0.01, 0.05, 0.00, 0.00

# Action  0 40.09, 35.23, 30.61, 26.25, 20.75, 16.32, 11.71, 7.59, 3.01
# Action  1 24.01, 21.20, 18.61, 15.81, 14.26, 11.60, 9.28, 6.30, 3.27
# Action  2 7.33, 6.69, 6.01, 4.92, 3.79, 2.96, 2.08, 1.10, 0.44
# Action  3 2.10, 1.93, 1.61, 1.49, 1.26, 0.96, 0.59, 0.41, 0.23
# Action  4 3.20, 2.85, 2.50, 2.04, 1.66, 1.32, 0.86, 0.40, 0.19

# ------ weather1 -------
# 21 0.32, 0.56, 0.26, 0.19, 0.17, 0.07, 0.10, 0.69, 1.00
# 45 0.09, 0.38, 0.40, 0.44, 0.45, 0.40, 0.57, 0.33, 0.36
# 73 0.64, 0.69, 0.70, 0.68, 0.71, 0.67, 0.66, 0.68, 0.89
# 1 0.22, 0.18, 0.64, 0.66, 0.73, 0.77, 0.72, 0.68, 0.33
# 25 0.59, 0.22, 0.18, 0.63, 0.65, 0.64, 0.50, 0.67, 0.40
# 77 0.43, 0.64, 0.59, 0.62, 0.62, 0.61, 0.64, 0.67, 1.00
# 53 0.64, 0.67, 0.71, 0.69, 0.63, 0.66, 0.60, 0.65, 0.73
# 29 0.19, 0.78, 0.84, 0.86, 0.83, 0.83, 0.88, 0.82, 1.00
# 5 0.66, 0.72, 0.70, 0.71, 0.68, 0.67, 0.62, 0.55, 0.50
# 93 0.51, 0.52, 0.49, 0.45, 0.42, 0.38, 0.47, 0.50, 0.20
# 65 0.15, 0.16, 0.41, 0.46, 0.28, 0.46, 0.15, 0.43, 0.33
# 113 0.59, 0.60, 0.54, 0.65, 0.63, 0.60, 0.68, 0.86, 0.83
# 41 0.23, 0.59, 0.09, 0.72, 0.71, 0.71, 0.71, 0.72, 0.50
# 97 0.37, 0.66, 0.64, 0.70, 0.63, 0.55, 0.54, 0.44, 0.14
# 69 0.41, 0.33, 0.37, 0.35, 0.35, 0.31, 0.41, 0.47, 0.78
# 117 0.46, 0.20, 0.63, 0.62, 0.66, 0.68, 0.59, 0.79, 1.00

# Reinforce  0.41, 0.49, 0.51, 0.59, 0.57, 0.56, 0.55, 0.62, 0.62

# Action  0 0.54, 0.61, 0.72, 0.87, 0.83, 0.83, 0.87, 0.75, 0.81
# Action  1 0.22, 0.29, 0.28, 0.12, 0.14, 0.16, 0.09, 0.25, 0.18
# Action  2 0.11, 0.03, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
# Action  3 0.10, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
# Action  4 0.07, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00

# Action  0 42.94, 39.69, 35.19, 29.81, 24.50, 19.19, 13.69, 9.25, 3.75
# Action  1 19.44, 15.88, 13.81, 11.88, 10.25, 8.94, 7.44, 4.50, 2.50
# Action  2 5.75, 4.94, 4.00, 3.38, 2.94, 2.12, 1.06, 0.56, 0.19
# Action  3 0.88, 0.81, 0.62, 0.62, 0.31, 0.25, 0.12, 0.00, 0.00
# Action  4 5.12, 4.38, 3.81, 3.25, 2.31, 1.56, 1.31, 1.00, 0.50

# ------ faa1 -------
# 85 0.34, 0.21, 0.35, 0.58, 0.65, 0.69, 0.62, 0.29, 0.00
# 33 0.10, 0.62, 0.61, 0.62, 0.55, 0.53, 0.57, 0.60, 0.86
# 57 0.24, 0.48, 0.46, 0.40, 0.38, 0.32, 0.33, 0.53, 0.71
# 9 0.64, 0.24, 0.61, 0.55, 0.46, 0.39, 0.32, 0.39, 0.12
# 65 0.48, 0.66, 0.54, 0.48, 0.35, 0.36, 0.40, 0.33, 0.90
# 89 0.22, 0.54, 0.53, 0.57, 0.51, 0.56, 0.42, 0.20, 0.14
# 37 0.41, 0.39, 0.41, 0.39, 0.37, 0.26, 0.35, 0.21, 0.42
# 109 0.59, 0.58, 0.65, 0.62, 0.59, 0.56, 0.56, 0.59, 0.50
# 13 0.31, 0.16, 0.10, 0.46, 0.39, 0.44, 0.62, 0.00, 1.00
# 41 0.25, 0.52, 0.51, 0.26, 0.50, 0.45, 0.52, 0.56, 0.75
# 93 0.22, 0.60, 0.31, 0.66, 0.60, 0.56, 0.58, 0.80, 1.00
# 69 0.37, 0.38, 0.40, 0.26, 0.34, 0.44, 0.40, 0.53, 0.75
# 45 0.26, 0.18, 0.09, 0.71, 0.65, 0.09, 0.69, 0.75, 1.00
# 21 0.81, 0.86, 0.83, 0.80, 0.83, 0.88, 1.00, 1.00, 1.00
# 81 0.22, 0.52, 0.60, 0.60, 0.51, 0.32, 0.33, 0.31, 0.33

# Reinforce  0.36, 0.46, 0.46, 0.53, 0.52, 0.47, 0.52, 0.47, 0.62

# Action  0 0.33, 0.61, 0.71, 0.81, 0.84, 0.93, 0.77, 0.74, 0.76
# Action  1 0.45, 0.22, 0.14, 0.15, 0.14, 0.01, 0.21, 0.12, 0.15
# Action  2 0.19, 0.15, 0.15, 0.07, 0.13, 0.06, 0.10, 0.00, 0.00
# Action  3 0.03, 0.02, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
# Action  4 0.02, 0.03, 0.02, 0.00, 0.00, 0.00, 0.05, 0.06, 0.00

# Action  0 38.48, 34.21, 29.54, 24.38, 19.04, 14.48, 10.83, 7.92, 4.15
# Action  1 17.02, 15.50, 14.19, 12.52, 11.00, 9.54, 7.29, 3.77, 1.44
# Action  2 6.23, 5.33, 4.17, 3.69, 3.15, 2.38, 1.50, 0.94, 0.21
# Action  3 1.75, 1.33, 1.21, 1.06, 0.98, 0.58, 0.33, 0.15, 0.00
# Action  4 4.54, 3.81, 3.48, 2.96, 2.77, 2.31, 1.65, 0.94, 0.38
