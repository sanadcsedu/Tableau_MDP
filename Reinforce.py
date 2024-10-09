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

        self.fc1 = nn.Linear(10, 128)
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
                s_prime, r, done, info, ground_action = self.env.step(s,a,False)
                predictions.append(info)


                self.pi.put_data((r * info, prob[a]))

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

                self.pi.put_data((r * info, prob[a]))

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
