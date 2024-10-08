import misc
import numpy as np
from collections import defaultdict
import itertools
import environment5 as environment5
import multiprocessing
from multiprocessing import Pool
import time
import random
from pathlib import Path
import glob
# from tqdm import tqdm 
import os 

class Qlearning:
    def __init__(self):
        pass


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
        # def policy_fnc(state):
        #     coin = random.random()
        #     if coin < epsilon:
        #             best_action = random.randint(0, 3)
        #     else:
        #         best_action = np.argmax(Q[state])
        #     return best_action

        # return policy_fnc
    
        def policy_fnc(state):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[state])
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fnc



    def q_learning(self, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5):
        """
        Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
        while following an epsilon-greedy policy

        Args:
            env: setting the environment as local fnc by importing env earlier
            num_episodes: Number of episodes to run for.
            discount_factor: Gamma discount factor.
            alpha: TD learning rate.
            epsilon: Chance to sample a random action. Float between 0 and 1.

        Returns:
            A tuple (Q, episode_lengths).
            Q is the optimal action-value function, a dictionary mapping state -> action values.
            stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        """
        Q = defaultdict(lambda: np.zeros(len(env.action_space)))

        for i_episode in range(num_episodes):
            # The policy we're following
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.action_space))

            # Reset the environment and pick the first state
            state = env.reset(all = False, test = False)
            training_accuracy=[]
            for t in itertools.count():
                # Take a step
                action_probs = policy(state)
                # print(action_probs)

                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, prediction, ground_action = env.step(state, action, False)

                training_accuracy.append(prediction)

                # TD Update
                best_next_action = np.argmax(Q[next_state])
                td_target = reward * prediction + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * (td_delta)

                state = next_state
                if done:
                    break


        return Q, np.mean(training_accuracy)


    def test(self, env, Q, discount_factor, alpha, epsilon, num_episodes=1):
        epsilon = epsilon

        for _ in range(1):

            state = env.reset(all=False, test=True)
            stats = []
            
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.action_space))
            insight = defaultdict(list)
            for t in itertools.count():
            
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

                next_state, reward, done, prediction, ground_action = env.step(state, action, True)
            
                stats.append(prediction)
                insight[ground_action].append(prediction)

                # Turning off the Q-Learning update when testing, the prediction is based on the Learned model from first x% interactions
                best_next_action = np.argmax(Q[next_state])
                td_target = reward * prediction + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * (td_delta)
                
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
        
        p1 = multiprocessing.Process(target=obj2.hyper_param, args=(user_list[:4], d, 'Qlearn',10, result_queue, info, info_split, info_split_cnt))
        p2 = multiprocessing.Process(target=obj2.hyper_param, args=(user_list[4:8], d, 'Qlearn',10, result_queue, info, info_split, info_split_cnt))
        p3 = multiprocessing.Process(target=obj2.hyper_param, args=(user_list[8:12], d, 'Qlearn',10, result_queue, info, info_split, info_split_cnt))
        p4 = multiprocessing.Process(target=obj2.hyper_param, args=(user_list[12:], d, 'Qlearn',10, result_queue, info, info_split, info_split_cnt))
        
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
        print("# Q-Learning ", ", ".join(f"{x:.2f}" for x in final_result))
        print()
        print("# Accuracy of actions over different thresholds")
        for ii in range(5):
            print("# Action ", ii, ", ".join(f"{x:.2f}" for x in split_final[ii]))
        # print()
        # print("# Average Count of Actions over different thresholds")
        # for ii in range(5):
        #     print("# Action ", ii, ", ".join(f"{x:.2f}" for x in split_final_cnt[ii]))


#  birdstrikes1  Dataset
# ---------------------------------------------

# 1 0.23, 0.15, 0.31, 0.08, 0.30, 0.35, 0.20, 0.66, 0.33
# 13 0.50, 0.30, 0.54, 0.43, 0.34, 0.43, 0.34, 0.30, 0.47
# 37 0.25, 0.32, 0.33, 0.19, 0.17, 0.57, 0.34, 0.48, 0.77
# 73 0.59, 0.60, 0.57, 0.49, 0.39, 0.44, 0.30, 0.38, 0.48
# 5 0.54, 0.52, 0.36, 0.57, 0.52, 0.49, 0.59, 0.47, 0.50
# 53 0.39, 0.44, 0.43, 0.51, 0.43, 0.34, 0.28, 0.36, 0.80
# 25 0.31, 0.31, 0.41, 0.51, 0.43, 0.48, 0.46, 0.41, 0.32
# 77 0.66, 0.66, 0.56, 0.61, 0.63, 0.64, 0.60, 0.59, 0.31
# 9 0.46, 0.25, 0.26, 0.47, 0.50, 0.45, 0.42, 0.68, 0.60
# 29 0.40, 0.50, 0.41, 0.55, 0.47, 0.54, 0.45, 0.56, 0.30
# 81 0.33, 0.40, 0.41, 0.32, 0.21, 0.46, 0.34, 0.44, 0.36
# 57 0.40, 0.48, 0.57, 0.61, 0.54, 0.49, 0.54, 0.28, 0.29
# 85 0.43, 0.36, 0.27, 0.39, 0.52, 0.39, 0.40, 0.53, 0.53
# 109 0.61, 0.63, 0.63, 0.71, 0.50, 0.72, 0.63, 0.67, 0.57
# 33 0.67, 0.68, 0.61, 0.59, 0.61, 0.57, 0.44, 0.43, 0.55
# 97 0.50, 0.65, 0.68, 0.62, 0.65, 0.65, 0.54, 0.50, 0.43
# 61 0.64, 0.47, 0.69, 0.68, 0.65, 0.67, 0.72, 0.71, 0.87

# Q-Learning  0.46, 0.45, 0.47, 0.49, 0.46, 0.51, 0.47, 0.50, 0.54

# Accuracy of actions over different thresholds
# Action  0 0.75, 0.74, 0.73, 0.82, 0.84, 0.77, 0.78, 0.77, 0.63
# Action  1 0.13, 0.14, 0.16, 0.10, 0.06, 0.13, 0.10, 0.12, 0.22
# Action  2 0.28, 0.16, 0.12, 0.26, 0.20, 0.28, 0.11, 0.20, 0.14
# Action  3 0.03, 0.02, 0.04, 0.02, 0.01, 0.00, 0.01, 0.00, 0.00
# Action  4 0.08, 0.05, 0.07, 0.02, 0.02, 0.04, 0.01, 0.01, 0.00

# Average Count of Actions over different thresholds
# Action  0 40.09, 35.23, 30.61, 26.25, 20.75, 16.32, 11.71, 7.59, 3.01
# Action  1 24.01, 21.20, 18.61, 15.81, 14.26, 11.60, 9.28, 6.30, 3.27
# Action  2 7.33, 6.69, 6.01, 4.92, 3.79, 2.96, 2.08, 1.10, 0.44
# Action  3 2.10, 1.93, 1.61, 1.49, 1.26, 0.96, 0.59, 0.41, 0.23
# Action  4 3.20, 2.85, 2.50, 2.04, 1.66, 1.32, 0.86, 0.40, 0.19

#  weather1  Dataset
# ---------------------------------------------

#  45 0.29, 0.26, 0.34, 0.47, 0.43, 0.38, 0.46, 0.30, 0.60
#  21 0.45, 0.39, 0.26, 0.19, 0.73, 0.77, 0.69, 0.57, 0.97
#  73 0.61, 0.63, 0.66, 0.63, 0.71, 0.66, 0.64, 0.67, 0.89
#  1 0.60, 0.58, 0.61, 0.63, 0.73, 0.76, 0.67, 0.68, 0.33
#  25 0.40, 0.19, 0.53, 0.58, 0.60, 0.63, 0.46, 0.60, 0.40
#  77 0.36, 0.59, 0.54, 0.61, 0.58, 0.62, 0.64, 0.67, 1.00
#  53 0.61, 0.66, 0.72, 0.65, 0.60, 0.65, 0.60, 0.65, 0.69
#  29 0.45, 0.68, 0.79, 0.80, 0.74, 0.82, 0.88, 0.76, 1.00
#  5 0.61, 0.64, 0.64, 0.66, 0.68, 0.63, 0.59, 0.52, 0.50
#  65 0.18, 0.24, 0.37, 0.27, 0.39, 0.40, 0.42, 0.37, 0.33
#  93 0.50, 0.49, 0.49, 0.44, 0.42, 0.35, 0.47, 0.48, 0.36
#  113 0.27, 0.52, 0.50, 0.50, 0.47, 0.53, 0.59, 0.80, 0.77
#  41 0.19, 0.32, 0.15, 0.59, 0.67, 0.55, 0.69, 0.71, 0.50
#  69 0.31, 0.40, 0.41, 0.32, 0.35, 0.23, 0.42, 0.40, 0.42
#  97 0.60, 0.65, 0.62, 0.66, 0.63, 0.55, 0.54, 0.46, 0.14
#  117 0.41, 0.44, 0.58, 0.60, 0.66, 0.68, 0.59, 0.79, 0.91

# Q-Learning  0.43, 0.49, 0.58, 0.54, 0.59, 0.58, 0.58, 0.59, 0.63

# Accuracy of actions over different thresholds
# Action  0 0.58, 0.67, 0.82, 0.82, 0.83, 0.76, 0.86, 0.72, 0.82
# Action  1 0.22, 0.14, 0.09, 0.08, 0.11, 0.18, 0.11, 0.24, 0.12
# Action  2 0.09, 0.07, 0.04, 0.06, 0.01, 0.02, 0.01, 0.03, 0.00
# Action  3 0.06, 0.06, 0.04, 0.03, 0.00, 0.00, 0.00, 0.00, 0.00
# Action  4 0.08, 0.06, 0.05, 0.05, 0.01, 0.01, 0.00, 0.01, 0.01

# Average Count of Actions over different thresholds
# Action  0 42.94, 39.69, 35.19, 29.81, 24.50, 19.19, 13.69, 9.25, 3.75
# Action  1 19.44, 15.88, 13.81, 11.88, 10.25, 8.94, 7.44, 4.50, 2.50
# Action  2 5.75, 4.94, 4.00, 3.38, 2.94, 2.12, 1.06, 0.56, 0.19
# Action  3 0.88, 0.81, 0.62, 0.62, 0.31, 0.25, 0.12, 0.00, 0.00
# Action  4 5.12, 4.38, 3.81, 3.25, 2.31, 1.56, 1.31, 1.00, 0.50

#  faa1  Dataset
# ---------------------------------------------

#  85 0.31, 0.35, 0.59, 0.61, 0.66, 0.68, 0.47, 0.43, 0.13
#  33 0.52, 0.62, 0.64, 0.61, 0.55, 0.54, 0.57, 0.60, 0.86
#  57 0.54, 0.48, 0.45, 0.21, 0.39, 0.41, 0.25, 0.37, 0.11
#  65 0.22, 0.34, 0.63, 0.41, 0.45, 0.33, 0.66, 0.63, 1.00
#  9 0.47, 0.58, 0.63, 0.56, 0.45, 0.45, 0.36, 0.37, 0.33
#  89 0.30, 0.35, 0.52, 0.56, 0.50, 0.55, 0.56, 0.73, 0.77
#  37 0.48, 0.36, 0.32, 0.30, 0.39, 0.28, 0.30, 0.31, 0.22
#  93 0.35, 0.66, 0.46, 0.68, 0.73, 0.61, 0.57, 0.87, 0.97
#  109 0.52, 0.29, 0.61, 0.62, 0.58, 0.51, 0.30, 0.52, 0.50
#  41 0.37, 0.53, 0.52, 0.35, 0.55, 0.50, 0.55, 0.49, 0.70
#  13 0.47, 0.45, 0.41, 0.56, 0.28, 0.50, 0.48, 0.68, 0.93
#  69 0.44, 0.42, 0.41, 0.42, 0.50, 0.50, 0.54, 0.65, 0.73
#  45 0.54, 0.66, 0.70, 0.66, 0.60, 0.72, 0.69, 0.70, 0.93
#  81 0.39, 0.51, 0.48, 0.63, 0.43, 0.31, 0.25, 0.23, 0.43
#  21 0.75, 0.78, 0.83, 0.80, 0.83, 0.89, 0.93, 0.95, 0.91

# Q-Learning  0.44, 0.49, 0.54, 0.54, 0.53, 0.52, 0.52, 0.57, 0.64

# Accuracy of actions over different thresholds
# Action  0 0.62, 0.72, 0.86, 0.84, 0.85, 0.83, 0.76, 0.86, 0.76
# Action  1 0.16, 0.13, 0.09, 0.10, 0.17, 0.13, 0.19, 0.22, 0.10
# Action  2 0.26, 0.37, 0.32, 0.42, 0.46, 0.47, 0.21, 0.15, 0.04
# Action  3 0.07, 0.07, 0.09, 0.11, 0.05, 0.04, 0.00, 0.00, 0.00
# Action  4 0.07, 0.07, 0.04, 0.01, 0.03, 0.06, 0.05, 0.00, 0.03

# Average Count of Actions over different thresholds
# Action  0 38.48, 34.21, 29.54, 24.38, 19.04, 14.48, 10.83, 7.92, 4.15
# Action  1 17.02, 15.50, 14.19, 12.52, 11.00, 9.54, 7.29, 3.77, 1.44
# Action  2 6.23, 5.33, 4.17, 3.69, 3.15, 2.38, 1.50, 0.94, 0.21
# Action  3 1.75, 1.33, 1.21, 1.06, 0.98, 0.58, 0.33, 0.15, 0.00
# Action  4 4.54, 3.81, 3.48, 2.96, 2.77, 2.31, 1.65, 0.94, 0.38