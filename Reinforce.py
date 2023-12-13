import pandas as pd
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
    def __init__(self,learning_rate,gamma,tau):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 4)
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
    def __init__(self,env,learning_rate,gamma,tau):
        self.env = env
        self.learning_rate, self.gamma, self.temperature = learning_rate, gamma, tau
        self.pi = Policy(self.learning_rate, self.gamma,self.temperature)

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
                s_prime, r, done, info = self.env.step(s,a,False)
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
            actions = []

            while not done:
                prob = policy(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                actions.append(a)
                s_prime, r, done, info  = self.env.step(s, a, True)
                predictions.append(info)
                
                policy.put_data((r, prob[a]))

                s = s_prime
                self.pi.train_net()

            test_accuracies.append(np.mean(predictions))
        return np.mean(test_accuracies)

class run_reinforce:
    def __init__(self):
        pass


    def run_experiment(self, user_list,dataset,hyperparam_file, result_queue):
        # Load hyperparameters from JSON file
        with open(hyperparam_file) as f:
            hyperparams = json.load(f)

        # Create result DataFrame with columns for relevant statistics
        result_dataframe = pd.DataFrame(
            columns=['Algorithm', 'User', 'Threshold', 'LearningRate', 'Discount', 'Temperature', 'Accuracy',
                    'StateAccuracy', 'Reward'])

        # Extract hyperparameters from JSON file
        learning_rates = hyperparams['learning_rates']
        gammas = hyperparams['gammas']
        temperatures = hyperparams['temperatures']

        # aggregate_plotter = plotting.plotter(None)
        final_accu = np.zeros(9, dtype=float)
        for user in user_list:
            # Extract user-specific threshold values
            threshold_h = hyperparams['threshold']            
            accu = []
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
                            agent = Reinforce(env,learning_rate,gamma,temp)
                            policy,accuracies = agent.train()

                            if accuracies > max_accu:
                                max_accu=accuracies
                                best_learning_rate=learning_rate
                                best_gamma=gamma
                                best_agent = agent
                                best_policy = policy
                                best_temp=temp

                # print("#TRAINING: User :{}, Threshold : {:.1f}, Accuracy: {}, LR: {} ,Discount: {}, Temperature:{}".format(user_name, thres,max_accu,best_learning_rate,best_gamma,best_temp))
                test_accuracy = best_agent.test(best_policy)
                # print("User :{}, Threshold : {:.1f}, Accuracy: {}".format(user_name, thres, test_accuracy))
                # print("#TESTING User :{}, Threshold : {:.1f}, Accuracy: {}, LR: {} ,Discount: {}, Temperature: {}".format(user_name, thres, max_accu,best_learning_rate,best_gamma,best_temp))
                accu.append(test_accuracy)
                env.reset(True, False) 
            # print(user[0], accu)
            print(user[0], ", ".join(f"{x:.2f}" for x in accu))

            final_accu = np.add(final_accu, accu)
        final_accu /= len(user_list)
        # print("Reinforce: ")
        # print(np.round(final_accu, decimals=2))
        result_queue.put(final_accu)

if __name__ == '__main__':
    env = environment5.environment5()
    datasets = env.datasets
    for d in datasets:
        print("------", d, "-------")
        user_list = env.obj.get_user_list_for_dataset(d)
        obj2 = run_reinforce()

        result_queue = multiprocessing.Queue()
        p1 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[:4], d, 'sampled_hyper_params.json', result_queue,))
        p2 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[4:8], d, 'sampled_hyper_params.json', result_queue,))
        p3 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[8:12], d, 'sampled_hyper_params.json', result_queue,))
        p4 = multiprocessing.Process(target=obj2.run_experiment, args=(user_list[12:], d, 'sampled_hyper_params.json', result_queue,))
        
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        final_result = np.zeros(9, dtype = float)
        p1.join()
        # temp = result_queue.get()
        final_result = np.add(final_result, result_queue.get())
        p2.join()
        # print(result_queue.get())
        final_result = np.add(final_result, result_queue.get())
        p3.join()
        # print(result_queue.get())
        final_result = np.add(final_result, result_queue.get())
        p4.join()
        # print(result_queue.get())
        final_result = np.add(final_result, result_queue.get())
        final_result /= 4
        print("Reinforce")
        print(np.round(final_result, decimals=2))
