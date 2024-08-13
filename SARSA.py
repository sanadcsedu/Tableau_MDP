import pdb
import misc
import numpy as np
from collections import defaultdict
import pandas as pd
import itertools
# import matplotlib.pyplot as plt
import sys
# import plotting
import environment5 as environment5
import random
import multiprocessing
import time
from pathlib import Path
import glob


class TD_SARSA:
    def __init__(self):
        pass

    # @jit(target ="cuda")
    def epsilon_greedy_policy(self, Q, epsilon, nA):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.
        Args:
            Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA (see below)
            epsilon: The probability to select a random action. Float between 0 and 1.
            nA: Number of actions in the environment.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.
        """

        def policy_fnc(state):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[state])
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fnc

    def sarsa(
        self, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5
    ):
        """
               SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

               Args:
                   env: OpenAI environment.
                   num_episodes: Number of episodes to run for.
                   discount_factor: Gamma discount factor.
                   alpha: TD learning rate.
                   epsilon: Chance the sample a random action. Float betwen 0 and 1.

               Returns:
                   A tuple (Q, stats).
                   Q is the optimal action-value function, a dictionary mapping state -> action values.
                   stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
               """
        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).

        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        # Define the valid actions for each state

        # Q = defaultdict(lambda: np.zeros(len(env.action_space)))
        Q = defaultdict(lambda: np.zeros(5))
        
        # The policy we're following
        # policy = self.epsilon_greedy_policy(Q, epsilon, len(env.action_space))
        policy = self.epsilon_greedy_policy(Q, epsilon, 5)
        
        for i_episode in range(num_episodes):
            state = env.reset(all = False, test = False)
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            
            training_accuracy=[]

            # One step in the environment
            for t in itertools.count():
                # Take a step
                next_state, reward, done, pred, _ = env.step(state, action, False)
                training_accuracy.append(pred)

                # Pick the next action
                next_action_probs = policy(next_state)
                next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
                
                # TD Update
                td_target = reward + discount_factor * Q[next_state][next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta
        
                if done:
                    break
                    
                action = next_action
                state = next_state        
        
        return Q, np.mean(training_accuracy)

    def test(self, env, Q, discount_factor, alpha, epsilon, num_episodes=1):
        epsilon = epsilon

        for i_episode in range(1):
            # Reset the environment and pick the first action
            state = env.reset(all=False, test=True)

            stats = []
            insight = defaultdict(list)
            # policy = self.epsilon_greedy_policy(Q, epsilon, len(env.action_space))
            policy = self.epsilon_greedy_policy(Q, epsilon, 5)

            for t in itertools.count():
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, prediction, ground_action = env.step(state, action, True)
                
                stats.append(prediction)
                insight[ground_action].append(prediction)

                # Pick the next action
                next_action_probs = policy(next_state)
                next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            
                # TD Update
                td_target = reward + discount_factor * Q[next_state][next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta

                state = next_state
                if done:
                    break

            granular_prediction = defaultdict()
            for keys, values in insight.items():
                granular_prediction[keys] = (len(values), np.mean(values))

            return np.mean(stats), granular_prediction

if __name__ == "__main__":
    env = environment5.environment5()
    datasets = env.datasets
    for d in datasets:
        final_output = []
        print("# ", d, " Dataset")
        print("# ---------------------------------------------")
        print()
        env.obj.create_connection(r"Tableau.db")
        user_list = env.obj.get_user_list_for_dataset(d)
        
        obj2 = misc.misc(len(user_list))
        result_queue = multiprocessing.Queue()
        info = multiprocessing.Queue()
        info_split = multiprocessing.Queue()
        info_split_cnt = multiprocessing.Queue() 
        
        p1 = multiprocessing.Process(target=obj2.hyper_param, args=(user_list[:4], d, 'SARSA',10, result_queue, info, info_split, info_split_cnt))
        p2 = multiprocessing.Process(target=obj2.hyper_param, args=(user_list[4:8], d, 'SARSA',10, result_queue, info, info_split, info_split_cnt))
        p3 = multiprocessing.Process(target=obj2.hyper_param, args=(user_list[8:12], d, 'SARSA',10, result_queue, info, info_split, info_split_cnt))
        p4 = multiprocessing.Process(target=obj2.hyper_param, args=(user_list[12:], d, 'SARSA',10, result_queue, info, info_split, info_split_cnt))
        
        split_final = np.zeros((5, 9), dtype = float)
        split_final_cnt = np.zeros((5, 9), dtype = float)

        p1.start()
        p2.start()
        p3.start()
        p4.start()
        final_result = np.zeros(9, dtype=float)
        p1.join()
        final_output.extend(info.get())
        final_result = np.add(final_result, result_queue.get())
        split_final = np.add(split_final, info_split.get())
        split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())
        
        p2.join()
        final_output.extend(info.get())
        final_result = np.add(final_result, result_queue.get())
        split_final = np.add(split_final, info_split.get())
        split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())

        p3.join()
        final_output.extend(info.get())
        final_result = np.add(final_result, result_queue.get())
        split_final = np.add(split_final, info_split.get())
        split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())

        p4.join()
        final_output.extend(info.get())
        final_result = np.add(final_result, result_queue.get())
        split_final = np.add(split_final, info_split.get())
        split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())

        final_result /= 4
        split_final /= 4
        split_final_cnt /= 4

        print()
        print("# SARSA ", ", ".join(f"{x:.2f}" for x in final_result))
        print()
        print("# Accuracy of actions over different thresholds")
        for ii in range(5):
            print("# Action ", ii, ", ".join(f"{x:.2f}" for x in split_final[ii]))
        print()
        print("# Average Count of Actions over different thresholds")
        for ii in range(5):
            print("# Action ", ii, ", ".join(f"{x:.2f}" for x in split_final_cnt[ii]))

#  birdstrikes1  Dataset
# ---------------------------------------------

#  1 0.28, 0.15, 0.21, 0.08, 0.19, 0.15, 0.22, 0.57, 0.67
#  37 0.27, 0.24, 0.24, 0.35, 0.14, 0.55, 0.50, 0.49, 0.53
#  13 0.57, 0.51, 0.53, 0.48, 0.44, 0.42, 0.35, 0.31, 0.50
#  73 0.59, 0.40, 0.57, 0.50, 0.35, 0.37, 0.41, 0.42, 0.50
#  5 0.54, 0.56, 0.56, 0.54, 0.57, 0.52, 0.51, 0.49, 0.50
#  53 0.36, 0.35, 0.42, 0.44, 0.38, 0.30, 0.28, 0.36, 0.87
#  25 0.32, 0.33, 0.38, 0.40, 0.46, 0.44, 0.41, 0.44, 0.44
#  57 0.30, 0.50, 0.59, 0.58, 0.56, 0.53, 0.53, 0.33, 0.29
#  9 0.36, 0.31, 0.24, 0.56, 0.44, 0.42, 0.37, 0.46, 0.67
#  29 0.43, 0.28, 0.45, 0.56, 0.46, 0.53, 0.41, 0.48, 0.25
#  77 0.61, 0.62, 0.63, 0.52, 0.62, 0.62, 0.64, 0.65, 0.36
#  109 0.60, 0.63, 0.62, 0.63, 0.71, 0.66, 0.64, 0.64, 0.57
#  61 0.47, 0.38, 0.60, 0.41, 0.64, 0.68, 0.79, 0.79, 0.89
#  81 0.47, 0.40, 0.50, 0.54, 0.37, 0.56, 0.50, 0.44, 0.40
#  33 0.67, 0.63, 0.60, 0.55, 0.38, 0.51, 0.45, 0.42, 0.36
#  85 0.35, 0.31, 0.32, 0.39, 0.39, 0.38, 0.58, 0.33, 0.47
#  97 0.69, 0.62, 0.65, 0.58, 0.66, 0.61, 0.48, 0.50, 0.34

# SARSA  0.46, 0.42, 0.47, 0.48, 0.46, 0.48, 0.47, 0.48, 0.51

# Accuracy of actions over different thresholds
# Action  0 0.79, 0.73, 0.82, 0.76, 0.77, 0.83, 0.76, 0.75, 0.75
# Action  1 0.13, 0.08, 0.10, 0.13, 0.09, 0.09, 0.12, 0.13, 0.17
# Action  2 0.20, 0.17, 0.27, 0.18, 0.35, 0.18, 0.10, 0.18, 0.13
# Action  3 0.08, 0.04, 0.01, 0.03, 0.00, 0.04, 0.05, 0.00, 0.06
# Action  4 0.03, 0.05, 0.03, 0.01, 0.06, 0.04, 0.02, 0.01, 0.00

# Average Count of Actions over different thresholds
# Action  0 40.09, 35.23, 30.61, 26.25, 20.75, 16.32, 11.71, 7.59, 3.01
# Action  1 24.01, 21.20, 18.61, 15.81, 14.26, 11.60, 9.28, 6.30, 3.27
# Action  2 7.33, 6.69, 6.01, 4.92, 3.79, 2.96, 2.08, 1.10, 0.44
# Action  3 2.10, 1.93, 1.61, 1.49, 1.26, 0.96, 0.59, 0.41, 0.23
# Action  4 3.20, 2.85, 2.50, 2.04, 1.66, 1.32, 0.86, 0.40, 0.19

#  weather1  Dataset
# ---------------------------------------------

#  45 0.38, 0.29, 0.29, 0.25, 0.43, 0.26, 0.39, 0.30, 0.56
#  21 0.32, 0.32, 0.26, 0.14, 0.17, 0.07, 0.73, 0.66, 0.93
#  1 0.21, 0.35, 0.59, 0.63, 0.74, 0.76, 0.71, 0.68, 0.33
#  73 0.46, 0.69, 0.67, 0.66, 0.65, 0.64, 0.65, 0.66, 0.80
#  25 0.38, 0.24, 0.60, 0.42, 0.59, 0.59, 0.50, 0.62, 0.40
#  77 0.36, 0.44, 0.58, 0.62, 0.60, 0.61, 0.64, 0.67, 0.95
#  53 0.64, 0.66, 0.66, 0.69, 0.60, 0.62, 0.59, 0.60, 0.73
#  29 0.46, 0.71, 0.78, 0.86, 0.75, 0.79, 0.84, 0.82, 1.00
#  5 0.62, 0.60, 0.65, 0.69, 0.67, 0.60, 0.61, 0.55, 0.48
#  93 0.51, 0.50, 0.45, 0.45, 0.42, 0.38, 0.47, 0.48, 0.28
#  65 0.21, 0.28, 0.30, 0.38, 0.34, 0.20, 0.42, 0.43, 0.27
#  113 0.53, 0.58, 0.56, 0.37, 0.42, 0.24, 0.65, 0.79, 0.83
#  41 0.52, 0.32, 0.41, 0.66, 0.65, 0.71, 0.70, 0.71, 0.50
#  97 0.43, 0.63, 0.64, 0.69, 0.60, 0.55, 0.54, 0.44, 0.14
#  69 0.36, 0.32, 0.37, 0.32, 0.42, 0.38, 0.41, 0.25, 0.40
#  117 0.17, 0.58, 0.29, 0.61, 0.64, 0.68, 0.57, 0.77, 0.96

# SARSA  0.41, 0.47, 0.51, 0.53, 0.54, 0.51, 0.59, 0.59, 0.60

# Accuracy of actions over different thresholds
# Action  0 0.50, 0.66, 0.79, 0.79, 0.85, 0.80, 0.81, 0.81, 0.83
# Action  1 0.30, 0.21, 0.10, 0.10, 0.09, 0.09, 0.19, 0.14, 0.11
# Action  2 0.10, 0.04, 0.05, 0.05, 0.01, 0.03, 0.01, 0.00, 0.00
# Action  3 0.04, 0.04, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00
# Action  4 0.08, 0.05, 0.05, 0.02, 0.02, 0.02, 0.01, 0.02, 0.00

# Average Count of Actions over different thresholds
# Action  0 42.94, 39.69, 35.19, 29.81, 24.50, 19.19, 13.69, 9.25, 3.75
# Action  1 19.44, 15.88, 13.81, 11.88, 10.25, 8.94, 7.44, 4.50, 2.50
# Action  2 5.75, 4.94, 4.00, 3.38, 2.94, 2.12, 1.06, 0.56, 0.19
# Action  3 0.88, 0.81, 0.62, 0.62, 0.31, 0.25, 0.12, 0.00, 0.00
# Action  4 5.12, 4.38, 3.81, 3.25, 2.31, 1.56, 1.31, 1.00, 0.50

#  faa1  Dataset
# ---------------------------------------------

#  85 0.35, 0.45, 0.30, 0.53, 0.66, 0.62, 0.53, 0.26, 0.00
#  33 0.37, 0.55, 0.57, 0.62, 0.51, 0.66, 0.58, 0.55, 0.77
#  57 0.26, 0.47, 0.48, 0.45, 0.38, 0.45, 0.33, 0.23, 0.49
#  9 0.43, 0.62, 0.58, 0.60, 0.40, 0.46, 0.35, 0.47, 0.23
#  65 0.69, 0.65, 0.54, 0.43, 0.35, 0.49, 0.40, 0.43, 1.00
#  89 0.53, 0.50, 0.53, 0.54, 0.52, 0.56, 0.55, 0.16, 0.86
#  37 0.42, 0.33, 0.24, 0.33, 0.29, 0.28, 0.29, 0.41, 0.58
#  109 0.30, 0.38, 0.57, 0.62, 0.45, 0.52, 0.52, 0.58, 0.28
#  41 0.38, 0.50, 0.51, 0.55, 0.50, 0.50, 0.45, 0.47, 0.75
#  13 0.42, 0.29, 0.31, 0.40, 0.50, 0.42, 0.54, 0.68, 0.93
#  93 0.54, 0.43, 0.60, 0.60, 0.56, 0.56, 0.59, 0.92, 0.91
#  69 0.41, 0.27, 0.46, 0.40, 0.39, 0.63, 0.44, 0.54, 0.73
#  45 0.45, 0.65, 0.71, 0.46, 0.77, 0.61, 0.68, 0.72, 0.93
#  81 0.32, 0.33, 0.54, 0.57, 0.47, 0.37, 0.05, 0.35, 0.43
#  21 0.51, 0.81, 0.80, 0.77, 0.82, 0.84, 0.98, 0.91, 0.97

# SARSA  0.43, 0.48, 0.51, 0.53, 0.51, 0.53, 0.49, 0.51, 0.65

# Accuracy of actions over different thresholds
# Action  0 0.63, 0.71, 0.80, 0.88, 0.89, 0.90, 0.86, 0.76, 0.78
# Action  1 0.16, 0.16, 0.09, 0.03, 0.08, 0.11, 0.05, 0.28, 0.15
# Action  2 0.30, 0.25, 0.21, 0.42, 0.33, 0.37, 0.10, 0.16, 0.00
# Action  3 0.09, 0.01, 0.01, 0.00, 0.11, 0.03, 0.00, 0.00, 0.00
# Action  4 0.07, 0.08, 0.02, 0.04, 0.01, 0.03, 0.03, 0.01, 0.01

# Average Count of Actions over different thresholds
# Action  0 38.48, 34.21, 29.54, 24.38, 19.04, 14.48, 10.83, 7.92, 4.15
# Action  1 17.02, 15.50, 14.19, 12.52, 11.00, 9.54, 7.29, 3.77, 1.44
# Action  2 6.23, 5.33, 4.17, 3.69, 3.15, 2.38, 1.50, 0.94, 0.21
# Action  3 1.75, 1.33, 1.21, 1.06, 0.98, 0.58, 0.33, 0.15, 0.00
# Action  4 4.54, 3.81, 3.48, 2.96, 2.77, 2.31, 1.65, 0.94, 0.38