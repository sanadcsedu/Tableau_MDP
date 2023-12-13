#contains all the miscellaneous functions for running 
import pandas as pd
import SARSA
import numpy as np
# import matplotlib.pyplot as plt 
import json
import Qlearning
from collections import Counter
from pathlib import Path
import glob
# from tqdm import tqdm 
import os 
import multiprocessing
import environment5

class misc:
    def __init__(self, users,hyperparam_file='sampled_hyper_params.json'):
        """
        Initializes the misc class.
        Parameters:
    - users: List of users
    - hyperparam_file: File path to the hyperparameters JSON file
    """
        # Load hyperparameters from JSON file
        with open(hyperparam_file) as f:
            hyperparams = json.load(f)
        # Extract hyperparameters from JSON file
        self.discount_h =hyperparams['gammas']
        self.alpha_h = hyperparams['learning_rates']
        self.epsilon_h = hyperparams['epsilon']
        self.threshold_h = hyperparams['threshold']


    def hyper_param(self, users_hyper, dataset, algorithm, epoch, result_queue):
        """
            Performs hyperparameter optimization.

            Parameters:
            - env: Environment object
            - users_hyper: List of user data
            - algorithm: Algorithm name ('QLearn' or 'SARSA')
            - epoch: Number of epochs

            Returns:
            None
            """
        best_discount = best_alpha = best_eps = -1
        pp = 5
        final_accu = np.zeros(9, dtype=float)
        for user in users_hyper:
            accu = []
            env = environment5.environment5()
            for thres in self.threshold_h:
                max_accu_thres = -1
                env.process_data(dataset, user[0], thres, algorithm) 
                for eps in self.epsilon_h:
                    for alp in self.alpha_h:
                        for dis in self.discount_h:
                            for _ in range(pp):
                                if algorithm == 'Qlearn':
                                    obj = Qlearning.Qlearning()
                                    Q, train_accuracy = obj.q_learning(env, epoch, dis, alp, eps)
                                    # print(train_accuracy)
                                else:
                                    obj = SARSA.TD_SARSA()
                                    Q, train_accuracy = obj.sarsa(env, epoch, dis, alp, eps)
                                    # print(train_accuracy)
                                if max_accu_thres < train_accuracy:
                                    max_accu_thres = train_accuracy
                                    best_eps = eps
                                    best_alpha = alp
                                    best_discount = dis
                                    best_q=Q
                                    best_obj=obj
                                max_accu_thres = max(max_accu_thres, train_accuracy)
                # print("Top Training Accuracy: {}, Threshold: {}".format(max_accu_thres, thres))
                test_accuracy = best_obj.test(env, best_q, best_discount, best_alpha, best_eps)
                accu.append(test_accuracy)
                env.reset(True, False)
            # print(user[0], accu)
            print(user[0], ", ".join(f"{x:.2f}" for x in accu))
            final_accu = np.add(final_accu, accu)
        final_accu /= len(users_hyper)
        # print(algorithm)
        # print(np.round(final_accu, decimals=2))
        result_queue.put(final_accu)