#contains all the miscellaneous functions for running 
import pandas as pd
import SARSA
import SARSA_v2
import numpy as np
# import matplotlib.pyplot as plt 
import json
import Qlearning
import Qlearning_v2
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

    def hyper_param(self, users_hyper, dataset, algorithm, epoch, result_queue, info, info_split_accu, info_split_cnt):
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
        output_list = []
        final_accu = np.zeros(9, dtype=float)
        final_cnt = np.zeros((5, 9), dtype = float)
        final_split_accu = np.zeros((5, 9), dtype = float)        
        for user in users_hyper:
            accu = []
            accu_split = [[] for _ in range(5)]
            cnt_split = [[] for _ in range(5)]
            env = environment5.environment5()
            for thres in self.threshold_h:
                max_accu_thres = -1
                env.process_data(dataset, user[0], thres, algorithm) 
                for eps in self.epsilon_h:
                    for alp in self.alpha_h:
                        for dis in self.discount_h:
                            # env = environment5.environment5()
                            # env.process_data(dataset, user[0], thres, algorithm) 
                            # for _ in range(pp):
                            if algorithm == 'Qlearn':
                                obj = Qlearning.Qlearning()
                                Q, train_accuracy = obj.q_learning(env, epoch, dis, alp, eps)
                            elif algorithm == 'Qlearn_v2':
                                obj = Qlearning_v2.Qlearning()
                                Q, train_accuracy = obj.q_learning(env, epoch, dis, alp, eps)
                            elif algorithm == 'SARSA':
                                obj = SARSA.TD_SARSA()
                                Q, train_accuracy = obj.sarsa(env, epoch, dis, alp, eps)
                            else: # algorithm == 'SARSA_v2'
                                obj = SARSA_v2.TD_SARSA()
                                Q, train_accuracy = obj.sarsa(env, epoch, dis, alp, eps)
                                
                            if max_accu_thres < train_accuracy:
                                max_accu_thres = train_accuracy
                                best_eps = eps
                                best_alpha = alp
                                best_discount = dis
                                best_q=Q
                                best_obj=obj
                            max_accu_thres = max(max_accu_thres, train_accuracy)
                
                test_accs = []
                # test_env = env
                split_accs = [[] for _ in range(5)]
                for _ in range(5):
                    test_model = best_obj
                    test_q, test_discount, test_alpha, test_eps = best_q, best_discount, best_alpha, best_eps
                    temp_accuracy, gp = test_model.test(env, test_q, test_discount, test_alpha, test_eps)
                    for key, val in gp.items():
                        split_accs[key].append(val[1])
                    test_accs.append(temp_accuracy)
                    # env.reset(True, False)

                
                test_accuracy = np.mean(test_accs)
                # test_accuracy = best_obj.test(env, best_q, best_discount, best_alpha, best_eps)
                accu.append(test_accuracy)
                env.reset(True, False)

                # printing accuracy for each action:
                # print("Threshold: {}".format(thres))
                for ii in range(5):
                    if len(split_accs[ii]) > 0:
                        # print("action: {}, count: {}, accuracy:{}".format(ii, gp[ii][0], np.mean(split_accs[ii])))
                        accu_split[ii].append(np.mean(split_accs[ii]))
                        cnt_split[ii].append(gp[ii][0])
                    else:
                        # print("{} Not Present".format(ii))
                        accu_split[ii].append(0)
                        cnt_split[ii].append(0)
            
            # print("# ", user[0], ", ".join(f"{x:.2f}" for x in accu))
            
            final_accu = np.add(final_accu, accu)
            for ii in range(5):            
                final_split_accu[ii] = np.add(final_split_accu[ii], accu_split[ii])
                final_cnt[ii] = np.add(final_cnt[ii], cnt_split[ii])

        final_accu /= len(users_hyper)
        for ii in range(5):            
            final_split_accu[ii] /= len(users_hyper)
            final_cnt[ii] /= len(users_hyper)
        
        result_queue.put(final_accu)
        info.put(output_list)
        info_split_accu.put(final_split_accu)
        info_split_cnt.put(final_cnt)