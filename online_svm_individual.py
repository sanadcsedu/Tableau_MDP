import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os
import ast
from read_data import read_data
# import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import environment5 as environment5
from tqdm import tqdm
import pdb 
import multiprocessing
from pathlib import Path
import json 
from collections import defaultdict


class OnlineSVM:
    def __init__(self, max_iter=10):
        """
        Initializes the Online SVM model using SGDClassifier.
        """
        from sklearn.linear_model import SGDClassifier
        self.model = SGDClassifier(loss='hinge', max_iter=max_iter, tol=1e-6)

    def train(self, X_train, y_train):
        """
        Train the model on the provided training data.
        """
        all_classes = np.array([0, 1, 2, 3, 4])

        # self.model.fit(X_train, y_train)
        self.model.partial_fit(X_train, y_train, classes=all_classes)


    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test data and return the accuracy.
        """
        # do online prediction predict , partial_fit, predict, partial_fit
        all_accuracies = []
        all_classes = np.array([0, 1, 2, 3, 4])
        insight = defaultdict(list)

        for i in range(len(X_test)):
            y_pred = self.model.predict([X_test[i]])
            accuracy = accuracy_score([y_test[i]], y_pred)
            insight[y_test[i]].append(accuracy)
            # pdb.set_trace()

            self.model.partial_fit([X_test[i]], [y_test[i]], classes=all_classes)
            all_accuracies.append(accuracy)

        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.mean(values))

        return np.mean(all_accuracies), granular_prediction

    def evaluate2(self, X_test, y_test):
        """
        Evaluate the model on the test data and return the accuracy.
        """
        #one shot prediction
        y_pred = self.model.predict(X_test)
        all_accuracies = accuracy_score(y_test, y_pred)

        return all_accuracies, y_pred

class run_online_svm:
    def __init__(self):
        pass

    def run_experiment(self, user_list, dataset, hyperparam_file, result_queue, info, info_split_accu, info_split_cnt):
        # print("here")
        # Load hyperparameters from JSON file
        with open(hyperparam_file) as f:
            hyperparams = json.load(f)

        final_accu = np.zeros(9, dtype=float)
        final_cnt = np.zeros((5, 9), dtype = float)
        final_split_accu = np.zeros((5, 9), dtype = float)
        
        # Loop over all users
        for user in user_list:
            # print(user)
            # Extract user-specific threshold values
            threshold_h = hyperparams['threshold']
            accu = []
            accu_split = [[] for _ in range(5)]
            cnt_split = [[] for _ in range(5)]

            env = environment5.environment5()
            # Loop over all threshold values
            for thres in threshold_h:
                # max_accu = -1
                # best_agent = None
                # best_model = None

                env.process_data(dataset, user[0], thres, 'Actor-Critic')
                
                X_train = np.array(env.mem_states[:env.threshold])
                y_train = np.array(env.mem_action[:env.threshold])
                # pdb.set_trace()
                # X_train = np.array(env.mem_states[:1])
                # y_train = np.array(env.mem_action[:1])
                
                all_classes = np.array([0, 1, 2, 3, 4])
                
                model = OnlineSVM()
                model.train(X_train, y_train)
            
                # Test the best agent and store results in DataFrame
                # test_accuracy = best_agent.test(best_model)
                #running them 5 times and taking the average test accuracy to reduce fluctuations
                test_accs = []
                split_accs = [[] for _ in range(5)]
                # print(X_train)
                for _ in range(5):
                    X_test = np.array(env.mem_states[env.threshold:])
                    y_test = np.array(env.mem_action[env.threshold:])
                    
                    # pdb.set_trace()
                    # Evaluate the model on the test data for this user
                    accuracy, gp = model.evaluate(X_test, y_test) 
                    for key, val in gp.items():
                        split_accs[key].append(val[1])
                        # split_cnt[key].append(val[0])
                    test_accs.append(accuracy)

                test_accuracy = np.mean(test_accs)
                accu.append(test_accuracy)
                env.reset(True, False)

                for ii in range(5):
                    if len(split_accs[ii]) > 0:
                        # print("action: {}, count: {}, accuracy:{}".format(ii, gp[ii][0], np.mean(split_accs[ii])))
                        accu_split[ii].append(np.mean(split_accs[ii]))
                        # cnt_split[ii].append(gp[ii][0])
                    else:
                        accu_split[ii].append(0)
                        # cnt_split[ii].append(0)

            # print(user[0], ", ".join(f"{x:.2f}" for x in accu))

            final_accu = np.add(final_accu, accu)
            for ii in range(5):            
                final_split_accu[ii] = np.add(final_split_accu[ii], accu_split[ii])
                # final_cnt[ii] = np.add(final_cnt[ii], cnt_split[ii])

        final_accu /= len(user_list)
        for ii in range(5):            
            final_split_accu[ii] /= len(user_list)
            # final_cnt[ii] /= len(user_list)
        # print(final_split_accu)
        result_queue.put(final_accu)
        info_split_accu.put(final_split_accu)
        # info_split_cnt.put(final_cnt)
    

if __name__ == '__main__':
    env = environment5.environment5()
    datasets = env.datasets
    for d in datasets:
        print("------", d, "-------")
        env.obj.create_connection(r"Tableau.db")
        user_list = env.obj.get_user_list_for_dataset(d)

        obj2 = run_online_svm()

        result_queue = multiprocessing.Queue()
        info = multiprocessing.Queue()
        info_split = multiprocessing.Queue()
        info_split_cnt = multiprocessing.Queue() 
        # obj2.run_experiment(user_list[:4], d, 'sampled_hyper_params.json', result_queue, info, info_split, info_split_cnt)
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
        # split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())
        # print(split_final)
        p2.join()
        final_result = np.add(final_result, result_queue.get())
        split_final = np.add(split_final, info_split.get())
        # split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())

        p3.join()
        final_result = np.add(final_result, result_queue.get())
        split_final = np.add(split_final, info_split.get())
        # split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())

        p4.join()
        final_result = np.add(final_result, result_queue.get())
        split_final = np.add(split_final, info_split.get())
        # split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())

        final_result /= 4
        split_final /= 4
        # split_final_cnt /= 4

        print("Online SVM ", ", ".join(f"{x:.2f}" for x in final_result))

        for ii in range(5):
            print("Action ", ii, ", ".join(f"{x:.2f}" for x in split_final[ii]))

        # for ii in range(5):
        #     print("Action ", ii, ", ".join(f"{x:.2f}" for x in split_final_cnt[ii]))