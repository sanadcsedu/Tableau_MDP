import pdb
import numpy as np
from collections import defaultdict
import pandas as pd
import itertools
# import matplotlib.pyplot as plt
import sys
# import plotting
import environment5_old as environment5_old
import random
import multiprocessing
import time
from pathlib import Path
import glob
import json 

class SARSA:
    def __init__(self):
        pass

    def epsilon_greedy_policy(self, Q, epsilon, nA):
        def policy_fnc(state):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[state])
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fnc

    def sarsa(self, Q, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5):

        for _ in range(num_episodes):
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.action_space))

            state = env.reset()
            action = policy(state)

            training_accuracy = []

            # One step in the environment
            for t in itertools.count():
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, prediction, ground_action = env.step(state, action, False)
                training_accuracy.append(prediction)

                best_next_action_probs = policy(next_state)
                best_next_action = np.random.choice(np.arange(len(best_next_action_probs)), p=best_next_action_probs)

                td_target = reward * prediction + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta

                # giving the reward for the ground action 
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][ground_action]
                Q[state][ground_action] += alpha * td_delta

                state = next_state
                action = best_next_action
                if done:
                    break

        return Q, np.mean(training_accuracy)

    def test(self, env, Q, discount_factor, alpha, epsilon, num_episodes=1):
        epsilon = epsilon

        for _ in range(1):
            state = env.reset()
            stats = []
            insight = defaultdict(list)
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.action_space))

            for t in itertools.count():
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, prediction, ground_action = env.step(state, action, True)

                stats.append(prediction)
                insight[ground_action].append(prediction)

                best_next_action_probs = policy(next_state)
                best_next_action = np.random.choice(np.arange(len(best_next_action_probs)), p=best_next_action_probs)

                td_target = reward * prediction + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta

                # giving the reward for the ground action 
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][ground_action]
                Q[state][ground_action] += alpha * td_delta

                state = next_state
                if done:
                    break

        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.sum(values))

        return np.mean(stats), granular_prediction
        
def training(train_files, env, dataset, algorithm, epoch):
    #loading the hyper-parameters 
    hyperparam_file='sampled_hyper_params.json'
    with open(hyperparam_file) as f:
            hyperparams = json.load(f)
    # Extract hyperparameters from JSON file
    discount_h =hyperparams['gammas']
    alpha_h = hyperparams['learning_rates']
    epsilon_h = hyperparams['epsilon']
    
    best_discount = best_alpha = best_eps = max_accu = -1
    for eps in epsilon_h:
        for alp in alpha_h:
            for dis in discount_h:
                accu = []
                model = SARSA()
                Q = defaultdict(lambda: np.zeros(len(env.action_space)))
                for user in train_files:
                    env.reset(True)
                    env.process_data(dataset, user[0], algorithm)             
                    #updates the Q value after each user trajectory
                    # print(user[0])
                    Q, accu_user = model.sarsa(Q, env, epoch, dis, alp, eps)
                    # print(user[0], eps, alp, dis, accu_user)
                    accu.append(accu_user)
                   
                #accuracy of the model learned over training data
                accu_model = np.mean(accu)
                if accu_model > max_accu:
                    max_accu = accu_model
                    best_eps = eps
                    best_alpha = alp
                    best_discount = dis
                    best_q=Q
    # print("Training Accuracy", max_accu)
    return best_q, best_alpha, best_eps, best_discount, max_accu

def testing(test_files, env, trained_Q, alpha, eps, discount, dataset, algorithm):
    Q = trained_Q
    final_accu = []
    for user in test_files:
        env.reset(True)
        env.process_data(dataset, user[0], algorithm)
        model = SARSA()
        accu, gp = model.test(env, Q, discount, alpha, eps)
        # pdb.set_trace()
        # print("testing", accu)
        final_accu.append(accu)
    return np.mean(final_accu), gp

if __name__ == "__main__":
    env = environment5_old.environment5()
    datasets = env.datasets
    # num_iterations = 5
    for d in datasets:
        final_output = []
        print("# ", d, " Dataset")
        print("# ---------------------------------------------")
        split_accs = [[] for _ in range(5)]
        split_cnt = [[] for _ in range(5)]              
        env.obj.create_connection(r"Tableau.db")
        user_list = env.obj.get_user_list_for_dataset(d)

        accuracies = []
        X_train = []
        X_test = []
        # for _ in range(num_iterations):
        # Leave-One-Out Cross-Validation
        for i, test_user_log in enumerate((user_list)):
            train_files = user_list[:i] + user_list[i+1:]  # All users except the ith one
            # train_files, test_files = train_test_split(user_list, test_size=0.3, random_state=42)
            trained_Q, best_alpha, best_eps, best_discount, training_accuracy = training(train_files, env, d, 'SARSA', 10)
            X_train.append(training_accuracy)
            # test user
            test_files = [test_user_log]
            testing_accu, gp = testing(test_files, env, trained_Q, best_alpha, best_eps, best_discount, d, 'SARSA')
            # print("Testing Accuracy ", accu)
            for key, val in gp.items():
                split_accs[key].append(val[1])
                split_cnt[key].append(val[0])

            X_test.append(testing_accu)
            # accuracies.append(accu)

        # train_accu = np.mean(X_train)
        test_accu = np.mean(X_test)
        # print("SARSA Training {:.2f}".format(train_accu))
        print("SARSA Testing {:.2f}".format(test_accu))

        for i in range(5):
            accu = round(np.sum(split_accs[i]) / np.sum(split_cnt[i]), 2)
            print("Action {} Accuracy {}".format(i, accu)) 
        print()