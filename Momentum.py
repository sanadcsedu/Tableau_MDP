import environment5
import numpy as np
from collections import defaultdict
import json
import pandas as pd
import random
import multiprocessing

eps=1e-35
class Momentum:
    def __init__(self):
        """Initializes the Greedy model."""
        self.last_action = defaultdict()

    def MomentumDriver(self, env, thres):
        length = len(env.mem_action)
        threshold = int(length * thres)
        for i in range(threshold):
            self.last_action[env.mem_states[i]] = env.mem_action[i]

        # Checking accuracy on the remaining data:
        accuracy = 0
        denom = 0
        insight = defaultdict(list)

        for i in range(threshold, length):
            denom += 1
            try: #Finding the last action in the current state
                candidate = self.last_action[env.mem_states[i]]
            except KeyError: #Randomly picking an action if the current state is new 
                candidate = random.choice([0, 1, 2, 3, 4])
            
            if candidate == env.mem_action[i]:
                accuracy += 1
                insight[env.mem_action[i]].append(1)
            else:
                insight[env.mem_action[i]].append(0)

            self.last_action[env.mem_states[i]] = env.mem_action[i]

        accuracy /= denom
        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.mean(values))

        return accuracy, granular_prediction


class run_Momentum:
    def __inti__(self):
        pass

    def run_experiment(self, user_list, dataset, hyperparam_file, result_queue, info, info_split_accu, info_split_cnt):
        # Load hyperparameters from JSON file
        with open(hyperparam_file) as f:
            hyperparams = json.load(f)
        threshold = hyperparams['threshold']

        # Create result DataFrame with columns for relevant statistics
        final_accu = np.zeros(9, dtype=float)
        final_cnt = np.zeros((5, 9), dtype = float)
        final_split_accu = np.zeros((5, 9), dtype = float)
        
        for u in user_list:
            accu = []
            accu_split = [[] for _ in range(5)]
            cnt_split = [[] for _ in range(5)]
            
            for thres in threshold:
                avg_accu = []
                split_accs = [[] for _ in range(5)]

                for _ in range(5):
                    env.process_data(dataset, u[0], thres, 'Greedy')
                    # print(env.mem_states)
                    obj = Momentum()
                    temp_accuracy, gp = obj.MomentumDriver(env, thres)
                    avg_accu.append(temp_accuracy)
                    env.reset(True, False)
                    
                    for key, val in gp.items():
                        split_accs[key].append(val[1])
                    
                accu.append(np.mean(avg_accu))
                for ii in range(5):
                    if len(split_accs[ii]) > 0:
                        accu_split[ii].append(np.mean(split_accs[ii]))
                        cnt_split[ii].append(gp[ii][0])
                    else:
                        accu_split[ii].append(0)
                        cnt_split[ii].append(0)
    
            print(u[0],",", ", ".join(f"{x:.2f}" for x in accu))
            
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


if __name__ == "__main__":
    env = environment5.environment5()
    datasets = env.datasets
    for d in datasets:
        print("------", d, "-------")
        env.obj.create_connection(r"Tableau.db")
        user_list = env.obj.get_user_list_for_dataset(d)
        # env.obj.close()

        obj2 = run_Momentum()

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

        print("Momentum ", ", ".join(f"{x:.2f}" for x in final_result))
        for ii in range(5):
            print("Action ", ii, ", ".join(f"{x:.2f}" for x in split_final[ii]))
