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
        for ii in range(4):
            print("# Action ", ii, ", ".join(f"{x:.2f}" for x in split_final[ii]))
        print()
        print("# Average Count of Actions over different thresholds")
        for ii in range(4):
            print("# Action ", ii, ", ".join(f"{x:.2f}" for x in split_final_cnt[ii]))


#  birdstrikes1  Dataset
# ---------------------------------------------

#  1 0.24, 0.21, 0.13, 0.18, 0.20, 0.14, 0.32, 0.57, 0.40
#  37 0.34, 0.24, 0.30, 0.52, 0.18, 0.14, 0.37, 0.48, 0.73
#  13 0.54, 0.55, 0.45, 0.47, 0.47, 0.39, 0.40, 0.26, 0.47
#  73 0.37, 0.62, 0.56, 0.50, 0.40, 0.43, 0.34, 0.57, 0.40
#  5 0.53, 0.49, 0.52, 0.59, 0.53, 0.50, 0.58, 0.56, 0.55
#  53 0.34, 0.45, 0.41, 0.41, 0.41, 0.28, 0.36, 0.53, 0.00
#  25 0.44, 0.43, 0.41, 0.42, 0.45, 0.44, 0.52, 0.51, 0.46
#  9 0.46, 0.27, 0.24, 0.32, 0.47, 0.48, 0.60, 0.38, 0.11
#  77 0.63, 0.42, 0.65, 0.61, 0.62, 0.60, 0.45, 0.62, 0.31
#  57 0.41, 0.55, 0.55, 0.56, 0.59, 0.51, 0.51, 0.32, 0.31
#  29 0.38, 0.45, 0.47, 0.57, 0.49, 0.52, 0.44, 0.50, 0.50
#  81 0.27, 0.38, 0.39, 0.37, 0.41, 0.41, 0.41, 0.29, 0.40
#  109 0.60, 0.57, 0.63, 0.67, 0.69, 0.76, 0.67, 0.62, 0.60
#  61 0.52, 0.60, 0.70, 0.69, 0.65, 0.62, 0.79, 0.89, 0.89
#  33 0.66, 0.31, 0.64, 0.60, 0.58, 0.40, 0.56, 0.50, 0.47
#  85 0.31, 0.47, 0.34, 0.17, 0.21, 0.61, 0.49, 0.12, 0.87
#  97 0.51, 0.65, 0.66, 0.66, 0.67, 0.62, 0.55, 0.46, 0.31

# SARSA  0.45, 0.45, 0.47, 0.49, 0.47, 0.46, 0.49, 0.49, 0.46

# Accuracy of actions over different thresholds
# Action  0 0.73, 0.80, 0.76, 0.80, 0.88, 0.89, 0.79, 0.66, 0.59
# Action  1 0.14, 0.11, 0.13, 0.13, 0.04, 0.06, 0.10, 0.22, 0.19
# Action  2 0.18, 0.19, 0.15, 0.17, 0.23, 0.19, 0.17, 0.11, 0.09
# Action  3 0.01, 0.04, 0.03, 0.04, 0.03, 0.05, 0.01, 0.01, 0.05

# Average Count of Actions over different thresholds
# Action  0 40.09, 35.23, 30.61, 26.25, 20.75, 16.32, 11.71, 7.59, 3.01
# Action  1 24.01, 21.20, 18.61, 15.81, 14.26, 11.60, 9.28, 6.30, 3.27
# Action  2 7.33, 6.69, 6.01, 4.92, 3.79, 2.96, 2.08, 1.10, 0.44
# Action  3 2.10, 1.93, 1.61, 1.49, 1.26, 0.96, 0.59, 0.41, 0.23

#  weather1  Dataset
# ---------------------------------------------

#  45 0.28, 0.29, 0.30, 0.28, 0.30, 0.50, 0.40, 0.32, 0.60
#  21 0.27, 0.31, 0.24, 0.19, 0.72, 0.09, 0.80, 0.65, 0.00
#  73 0.62, 0.66, 0.64, 0.66, 0.66, 0.62, 0.62, 0.65, 0.80
#  1 0.38, 0.64, 0.63, 0.65, 0.69, 0.75, 0.67, 0.68, 0.33
#  25 0.28, 0.18, 0.54, 0.63, 0.61, 0.63, 0.50, 0.65, 0.40
#  77 0.21, 0.56, 0.53, 0.61, 0.54, 0.56, 0.64, 0.64, 1.00
#  53 0.42, 0.62, 0.35, 0.69, 0.58, 0.57, 0.59, 0.64, 0.73
#  29 0.75, 0.73, 0.81, 0.86, 0.77, 0.74, 0.88, 0.78, 0.96
#  5 0.66, 0.70, 0.66, 0.68, 0.64, 0.49, 0.59, 0.53, 0.42
#  93 0.48, 0.51, 0.42, 0.40, 0.39, 0.29, 0.03, 0.48, 0.20
#  65 0.29, 0.27, 0.21, 0.34, 0.47, 0.49, 0.37, 0.37, 0.33
#  41 0.59, 0.41, 0.71, 0.58, 0.57, 0.71, 0.69, 0.72, 0.50
#  113 0.42, 0.23, 0.50, 0.48, 0.57, 0.45, 0.54, 0.83, 0.80
#  97 0.59, 0.62, 0.64, 0.70, 0.63, 0.50, 0.51, 0.40, 0.14
#  69 0.35, 0.21, 0.35, 0.38, 0.37, 0.27, 0.10, 0.42, 0.40
#  117 0.53, 0.30, 0.58, 0.59, 0.45, 0.60, 0.58, 0.76, 0.98

# SARSA  0.45, 0.45, 0.51, 0.54, 0.56, 0.52, 0.53, 0.60, 0.54

# Accuracy of actions over different thresholds
# Action  0 0.65, 0.62, 0.72, 0.87, 0.79, 0.81, 0.71, 0.75, 0.86
# Action  1 0.15, 0.19, 0.14, 0.05, 0.11, 0.10, 0.13, 0.18, 0.04
# Action  2 0.10, 0.06, 0.03, 0.02, 0.02, 0.03, 0.08, 0.03, 0.00
# Action  3 0.01, 0.03, 0.03, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00

# Average Count of Actions over different thresholds
# Action  0 42.94, 39.69, 35.19, 29.81, 24.50, 19.19, 13.69, 9.25, 3.75
# Action  1 19.44, 15.88, 13.81, 11.88, 10.25, 8.94, 7.44, 4.50, 2.50
# Action  2 5.75, 4.94, 4.00, 3.38, 2.94, 2.12, 1.06, 0.56, 0.19
# Action  3 0.88, 0.81, 0.62, 0.62, 0.31, 0.25, 0.12, 0.00, 0.00

#  faa1  Dataset
# ---------------------------------------------

#  85 0.50, 0.47, 0.57, 0.58, 0.63, 0.65, 0.36, 0.29, 0.00
#  57 0.23, 0.44, 0.45, 0.45, 0.33, 0.27, 0.33, 0.28, 0.37
#  33 0.41, 0.63, 0.60, 0.63, 0.57, 0.51, 0.52, 0.57, 0.83
#  65 0.47, 0.32, 0.67, 0.56, 0.36, 0.47, 0.56, 0.33, 1.00
#  9 0.60, 0.59, 0.64, 0.60, 0.49, 0.41, 0.39, 0.42, 0.12
#  89 0.52, 0.53, 0.51, 0.56, 0.52, 0.58, 0.55, 0.75, 0.83
#  37 0.36, 0.37, 0.33, 0.38, 0.33, 0.27, 0.28, 0.29, 0.24
#  93 0.38, 0.61, 0.52, 0.64, 0.61, 0.58, 0.66, 0.80, 0.91
#  109 0.48, 0.58, 0.66, 0.56, 0.57, 0.52, 0.55, 0.58, 0.50
#  41 0.38, 0.33, 0.48, 0.55, 0.57, 0.51, 0.51, 0.69, 0.75
#  69 0.30, 0.19, 0.28, 0.53, 0.46, 0.53, 0.50, 0.54, 0.57
#  45 0.39, 0.53, 0.75, 0.71, 0.77, 0.54, 0.66, 0.70, 1.00
#  13 0.53, 0.34, 0.45, 0.56, 0.42, 0.44, 0.43, 0.75, 0.93
#  81 0.35, 0.45, 0.55, 0.62, 0.49, 0.39, 0.25, 0.31, 0.37
#  21 0.53, 0.79, 0.83, 0.79, 0.85, 0.88, 0.94, 1.00, 1.00

# SARSA  0.43, 0.48, 0.55, 0.58, 0.53, 0.51, 0.50, 0.56, 0.63

# Accuracy of actions over different thresholds
# Action  0 0.65, 0.74, 0.84, 0.91, 0.91, 0.89, 0.82, 0.83, 0.82
# Action  1 0.16, 0.10, 0.09, 0.08, 0.07, 0.09, 0.10, 0.18, 0.03
# Action  2 0.24, 0.22, 0.38, 0.62, 0.45, 0.33, 0.15, 0.14, 0.00
# Action  3 0.08, 0.11, 0.11, 0.01, 0.03, 0.00, 0.00, 0.00, 0.00

# Average Count of Actions over different thresholds
# Action  0 38.48, 34.21, 29.54, 24.38, 19.04, 14.48, 10.83, 7.92, 4.15
# Action  1 17.02, 15.50, 14.19, 12.52, 11.00, 9.54, 7.29, 3.77, 1.44
# Action  2 6.23, 5.33, 4.17, 3.69, 3.15, 2.38, 1.50, 0.94, 0.21
# Action  3 1.75, 1.33, 1.21, 1.06, 0.98, 0.58, 0.33, 0.15, 0.00