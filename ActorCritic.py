import environment5
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# import plotting
from collections import Counter
# import pandas as pd
import json
import os
from collections import defaultdict
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from pathlib import Path
import glob
# from tqdm import tqdm 
import multiprocessing


#Class definition for the Actor-Critic model
class ActorCritic(nn.Module):
    def __init__(self,learning_rate,gamma, dataset):
        super(ActorCritic, self).__init__()
        # Class attributes
        self.data = []
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Neural network architecture
        if dataset == 'birdstrikes1':
            self.fc1 = nn.Linear(6, 64)
            self.fc_pi = nn.Linear(64, 5)#actor
            self.fc_v = nn.Linear(64, 1)#critic
        elif dataset == 'weather1':
            self.fc1 = nn.Linear(7, 64)
            self.fc_pi = nn.Linear(64, 5)#actor
            self.fc_v = nn.Linear(64, 1)#critic
        else: #dataset is FAA
            self.fc1 = nn.Linear(4, 64)
            self.fc_pi = nn.Linear(64, 5)#actor
            self.fc_v = nn.Linear(64, 1)#critic
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    #The critic network (called self.fc_v in the code) estimates the state value and is trained using the TD error to minimize the difference between the predicted and actual return.
    def pi(self, x, softmax_dim=0):
        """
        Compute the action probabilities using the policy network.

        Args:
            x (torch.Tensor): State tensor.
            softmax_dim (int): Dimension along which to apply the softmax function (default=0).

        Returns:
            prob (torch.Tensor): Tensor with the action probabilities.
        """

        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    #The actor network (called self.fc_pi ) outputs the action probabilities and is trained using the policy gradient method to maximize the expected return.
    def v(self, x):
        """
        Compute the state value using the value network.

        Args:
            x (torch.Tensor): State tensor.

        Returns:
            v (torch.Tensor): Tensor with the estimated state value.
        """

        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        """
        Add a transition tuple to the data buffer.

        Args:
            transition (tuple): Tuple with the transition data (s, a, r, s_prime, done).
        """

        self.data.append(transition)

    def make_batch(self):
        """
        Generate a batch of training data from the data buffer.

        Returns:
            s_batch, a_batch, r_batch, s_prime_batch, done_batch (torch.Tensor): Tensors with the
                states, actions, rewards, next states, and done flags for the transitions in the batch.
        """

        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

            s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(np.array(s_lst), dtype=torch.float), torch.tensor(np.array(a_lst)), \
                                                               torch.tensor(np.array(r_lst), dtype=torch.float), torch.tensor(np.array(s_prime_lst), dtype=torch.float), \
                                                               torch.tensor(np.array(done_lst), dtype=torch.float)

        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        """
           Train the Actor-Critic model using a batch of training data.
           """
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + self.gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)

        #The first term is the policy loss, which is computed as the negative log probability of the action taken multiplied by the advantage
        # (i.e., the difference between the estimated value and the target value).
        # The second term is the value loss, which is computed as the mean squared error between the estimated value and the target value
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

class Agent():
    def __init__(self, env,learning_rate,gamma,num_rollouts=10):
        self.env = env
        self.learning_rate, self.gamma, self.n_rollout=learning_rate,gamma,num_rollouts

    def train(self, dataset):
        model = ActorCritic(self.learning_rate, self.gamma, dataset)
        score = 0.0
        all_predictions = []
        for _ in range(5):
            done = False
            s = self.env.reset(all = False, test = False)

            predictions = []
            actions = []
            while not done:
                for t in range(self.n_rollout):
                    prob = model.pi(torch.from_numpy(s).float())
                    m = Categorical(prob)
                    a = m.sample().item()
                    actions.append(a)
                    s_prime, r, done, info, _ = self.env.step(s, a, False)
                    predictions.append(info)
                   
                    model.put_data((s, a, r, s_prime, done))

                    s = s_prime

                    score += r

                    if done:
                        break
                #train at the end of the episode: batch will contain all the transitions from the n-steps
                model.train_net()

            score = 0.0
            all_predictions.append(np.mean(predictions))
        # print("############ Train Accuracy :{:.2f},".format(np.mean(all_predictions)))
        return model, np.mean(predictions)  # return last episodes accuracyas training accuracy


    def test(self,model):

        test_predictions = []
        for _ in range(1):
            done = False
            s = self.env.reset(all=False, test=True)
            predictions = []
            score=0
            insight = defaultdict(list)

            while not done:
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, pred, ground_action = self.env.step(s, a, True)
                predictions.append(pred)
                
                insight[ground_action].append(pred)
                model.put_data((s, a, r, s_prime, done))

                s = s_prime
                
                score += r

                if done:
                    break
                model.train_net()

            test_predictions.append(np.mean(predictions))
        
        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.mean(values))

        return np.mean(test_predictions), granular_prediction

class run_ac:
    def __init__(self):
        pass

    def run_experiment(self, user_list, dataset, hyperparam_file, result_queue, info, info_split_accu, info_split_cnt):
        # Load hyperparameters from JSON file
        with open(hyperparam_file) as f:
            hyperparams = json.load(f)

        # Extract hyperparameters from JSON file
        learning_rates = hyperparams['learning_rates']
        gammas = hyperparams['gammas']

        final_accu = np.zeros(9, dtype=float)
        final_cnt = np.zeros((5, 9), dtype = float)
        final_split_accu = np.zeros((5, 9), dtype = float)
        
        # Loop over all users
        for user in user_list:
            # Extract user-specific threshold values
            threshold_h = hyperparams['threshold']
            accu = []
            accu_split = [[] for _ in range(5)]
            cnt_split = [[] for _ in range(5)]

            env = environment5.environment5()
            # Loop over all threshold values
            for thres in threshold_h:
                max_accu = -1
                best_learning_rate = 0
                best_gamma = 0
                best_agent = None
                best_model = None

                env.process_data(dataset, user[0], thres, 'Actor-Critic')
                # Loop over all combinations of hyperparameters
                for learning_rate in learning_rates:
                    for gamma in gammas:
                        # env = environment5.environment5()
                        # env.process_data(dataset, user[0], thres, 'Actor-Critic')
                        agent = Agent(env, learning_rate, gamma)
                        model, accuracies = agent.train(dataset)

                        # Keep track of best combination of hyperparameters
                        if accuracies > max_accu:
                            max_accu = accuracies
                            best_learning_rate = learning_rate
                            best_gamma = gamma
                            best_agent = agent
                            best_model = model

                # Test the best agent and store results in DataFrame
                # test_accuracy = best_agent.test(best_model)
                #running them 5 times and taking the average test accuracy to reduce fluctuations
                test_accs = []
                split_accs = [[] for _ in range(5)]
                
                for _ in range(5):
                    test_agent = best_agent
                    test_model = best_model
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

        obj2 = run_ac()

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

        print("Actor-Critic ", ", ".join(f"{x:.2f}" for x in final_result))

        for ii in range(5):
            print("Action ", ii, ", ".join(f"{x:.2f}" for x in split_final[ii]))

        for ii in range(5):
            print("Action ", ii, ", ".join(f"{x:.2f}" for x in split_final_cnt[ii]))

# ------ birdstrikes1 -------
# 1 0.23, 0.21, 0.19, 0.03, 0.67, 0.23, 0.77, 0.86, 0.80
# 37 0.19, 0.29, 0.45, 0.63, 0.64, 0.71, 0.76, 0.75, 0.80
# 13 0.25, 0.32, 0.16, 0.23, 0.38, 0.41, 0.33, 0.24, 0.43
# 73 0.30, 0.60, 0.63, 0.55, 0.49, 0.50, 0.56, 0.35, 0.60
# 5 0.45, 0.54, 0.34, 0.57, 0.53, 0.51, 0.59, 0.56, 0.50
# 53 0.20, 0.52, 0.47, 0.47, 0.49, 0.34, 0.64, 0.33, 0.27
# 25 0.38, 0.45, 0.47, 0.42, 0.49, 0.51, 0.49, 0.50, 0.50
# 77 0.56, 0.62, 0.58, 0.60, 0.57, 0.17, 0.63, 0.60, 0.33
# 57 0.20, 0.46, 0.49, 0.48, 0.25, 0.53, 0.53, 0.32, 0.29
# 9 0.08, 0.18, 0.34, 0.21, 0.44, 0.35, 0.49, 0.53, 0.38
# 29 0.16, 0.20, 0.47, 0.51, 0.43, 0.41, 0.44, 0.50, 0.25
# 81 0.48, 0.42, 0.41, 0.46, 0.37, 0.43, 0.39, 0.27, 0.20
# 85 0.16, 0.18, 0.25, 0.34, 0.22, 0.54, 0.51, 0.72, 0.87
# 61 0.21, 0.75, 0.38, 0.69, 0.64, 0.27, 0.69, 0.89, 0.89
# 109 0.23, 0.50, 0.67, 0.68, 0.73, 0.76, 0.67, 0.62, 0.62
# 33 0.52, 0.61, 0.62, 0.65, 0.60, 0.48, 0.45, 0.43, 0.36
# 97 0.71, 0.68, 0.25, 0.65, 0.65, 0.65, 0.52, 0.50, 0.43
# Actor-Critic  0.30, 0.44, 0.42, 0.48, 0.51, 0.46, 0.56, 0.53, 0.50
# Action  0 0.31, 0.70, 0.56, 0.60, 0.62, 0.69, 0.72, 0.79, 0.73
# Action  1 0.36, 0.16, 0.41, 0.34, 0.33, 0.30, 0.29, 0.19, 0.20
# Action  2 0.05, 0.09, 0.01, 0.02, 0.05, 0.00, 0.01, 0.01, 0.00
# Action  3 0.08, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00
# Action  4 0.12, 0.08, 0.00, 0.03, 0.01, 0.00, 0.00, 0.00, 0.00
# Action  0 40.09, 35.23, 30.61, 26.25, 20.75, 16.32, 11.71, 7.59, 3.01
# Action  1 24.01, 21.20, 18.61, 15.81, 14.26, 11.60, 9.28, 6.30, 3.27
# Action  2 7.33, 6.69, 6.01, 4.92, 3.79, 2.96, 2.08, 1.10, 0.44
# Action  3 2.10, 1.93, 1.61, 1.49, 1.26, 0.96, 0.59, 0.41, 0.23
# Action  4 3.20, 2.85, 2.50, 2.04, 1.66, 1.32, 0.86, 0.40, 0.19
# ------ weather1 -------
# 45 0.40, 0.39, 0.40, 0.42, 0.54, 0.40, 0.38, 0.33, 0.52
# 21 0.18, 0.31, 0.26, 0.50, 0.55, 0.07, 0.10, 0.69, 0.90
# 1 0.50, 0.18, 0.63, 0.66, 0.73, 0.77, 0.72, 0.68, 0.33
# 73 0.66, 0.68, 0.70, 0.68, 0.72, 0.66, 0.65, 0.68, 0.89
# 25 0.24, 0.55, 0.48, 0.65, 0.64, 0.63, 0.50, 0.65, 0.40
# 77 0.26, 0.64, 0.59, 0.62, 0.62, 0.63, 0.64, 0.67, 0.95
# 53 0.52, 0.63, 0.70, 0.69, 0.63, 0.66, 0.60, 0.65, 0.73
# 29 0.55, 0.77, 0.83, 0.86, 0.83, 0.83, 0.88, 0.82, 1.00
# 5 0.66, 0.71, 0.70, 0.71, 0.68, 0.67, 0.62, 0.55, 0.50
# 93 0.50, 0.52, 0.48, 0.45, 0.42, 0.38, 0.46, 0.50, 0.20
# 65 0.25, 0.41, 0.18, 0.26, 0.20, 0.34, 0.42, 0.60, 0.27
# 113 0.54, 0.23, 0.67, 0.62, 0.63, 0.60, 0.68, 0.86, 0.83
# 41 0.18, 0.15, 0.48, 0.71, 0.70, 0.71, 0.71, 0.72, 0.50
# 97 0.63, 0.66, 0.64, 0.70, 0.63, 0.55, 0.54, 0.44, 0.20
# 69 0.41, 0.33, 0.37, 0.34, 0.40, 0.30, 0.42, 0.46, 0.71
# 117 0.57, 0.61, 0.63, 0.62, 0.66, 0.68, 0.59, 0.79, 1.00
# Actor-Critic  0.44, 0.49, 0.55, 0.59, 0.60, 0.55, 0.56, 0.63, 0.62
# Action  0 0.54, 0.62, 0.77, 0.77, 0.90, 0.82, 0.87, 0.81, 0.86
# Action  1 0.30, 0.35, 0.17, 0.19, 0.06, 0.15, 0.12, 0.21, 0.13
# Action  2 0.06, 0.00, 0.02, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00
# Action  3 0.01, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
# Action  4 0.06, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
# Action  0 42.94, 39.69, 35.19, 29.81, 24.50, 19.19, 13.69, 9.25, 3.75
# Action  1 19.44, 15.88, 13.81, 11.88, 10.25, 8.94, 7.44, 4.50, 2.50
# Action  2 5.75, 4.94, 4.00, 3.38, 2.94, 2.12, 1.06, 0.56, 0.19
# Action  3 0.88, 0.81, 0.62, 0.62, 0.31, 0.25, 0.12, 0.00, 0.00
# Action  4 5.12, 4.38, 3.81, 3.25, 2.31, 1.56, 1.31, 1.00, 0.50
# ------ faa1 -------
# 85 0.23, 0.52, 0.12, 0.59, 0.65, 0.68, 0.55, 0.29, 0.00
# 57 0.54, 0.47, 0.46, 0.40, 0.40, 0.32, 0.32, 0.25, 0.60
# 33 0.46, 0.61, 0.61, 0.62, 0.54, 0.53, 0.57, 0.60, 0.86
# 9 0.63, 0.62, 0.61, 0.56, 0.45, 0.39, 0.32, 0.42, 0.17
# 65 0.42, 0.61, 0.58, 0.48, 0.35, 0.34, 0.40, 0.33, 0.00
# 89 0.52, 0.55, 0.53, 0.57, 0.51, 0.55, 0.57, 0.67, 0.86
# 37 0.38, 0.39, 0.39, 0.31, 0.42, 0.42, 0.40, 0.31, 0.53
# 109 0.60, 0.64, 0.64, 0.61, 0.58, 0.56, 0.55, 0.59, 0.50
# 93 0.37, 0.52, 0.61, 0.66, 0.61, 0.56, 0.59, 0.81, 1.00
# 41 0.25, 0.57, 0.51, 0.53, 0.49, 0.45, 0.40, 0.56, 0.75
# 13 0.19, 0.43, 0.19, 0.49, 0.43, 0.47, 0.48, 0.75, 1.00
# 69 0.36, 0.44, 0.42, 0.39, 0.43, 0.34, 0.41, 0.40, 0.23
# 45 0.58, 0.61, 0.43, 0.71, 0.74, 0.67, 0.62, 0.75, 1.00
# 81 0.24, 0.62, 0.62, 0.56, 0.53, 0.36, 0.30, 0.37, 0.27
# 21 0.82, 0.85, 0.83, 0.80, 0.83, 0.88, 1.00, 1.00, 1.00

# Actor-Critic  0.44, 0.56, 0.50, 0.56, 0.54, 0.51, 0.50, 0.54, 0.59

# Action  0 0.53, 0.88, 0.76, 0.96, 0.89, 0.86, 0.75, 0.72, 0.70
# Action  1 0.37, 0.12, 0.22, 0.06, 0.13, 0.13, 0.21, 0.29, 0.16
# Action  2 0.04, 0.02, 0.07, 0.02, 0.02, 0.02, 0.03, 0.01, 0.00
# Action  3 0.04, 0.00, 0.00, 0.00, 0.01, 0.00, 0.01, 0.00, 0.00
# Action  4 0.06, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00
# Action  0 38.48, 34.21, 29.54, 24.38, 19.04, 14.48, 10.83, 7.92, 4.15
# Action  1 17.02, 15.50, 14.19, 12.52, 11.00, 9.54, 7.29, 3.77, 1.44
# Action  2 6.23, 5.33, 4.17, 3.69, 3.15, 2.38, 1.50, 0.94, 0.21
# Action  3 1.75, 1.33, 1.21, 1.06, 0.98, 0.58, 0.33, 0.15, 0.00
# Action  4 4.54, 3.81, 3.48, 2.96, 2.77, 2.31, 1.65, 0.94, 0.38
