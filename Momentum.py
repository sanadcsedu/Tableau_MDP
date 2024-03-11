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
        for i in range(threshold, length):
            denom += 1
            try: #Finding the last action in the current state
                candidate = self.last_action[env.mem_states[i]]
            except KeyError: #Randomly picking an action if the current state is new 
                candidate = random.choice([0, 1, 2, 3, 4])
            
            if candidate == env.mem_action[i]:
                 accuracy += 1

        accuracy /= denom
        return accuracy

class run_Momentum:
    def __inti__(self):
        pass

    def run_experiment(self, user_list, dataset, hyperparam_file, result_queue):
        # Load hyperparameters from JSON file
        with open(hyperparam_file) as f:
            hyperparams = json.load(f)
        threshold = hyperparams['threshold']

        # Create result DataFrame with columns for relevant statistics
        final_accu = np.zeros(9, dtype=float)
        for u in user_list:
            accu = []
            for thres in threshold:
                avg_accu = []
                for _ in range(5):
                    env.process_data(dataset, u[0], thres, 'Greedy')
                    # print(env.mem_states)
                    obj = Momentum()
                    avg_accu.append(obj.MomentumDriver(env, thres))
                    env.reset(True, False)
                accu.append(np.mean(avg_accu))
            final_accu = np.add(final_accu, accu)
        
        final_accu /= len(user_list)
        result_queue.put(final_accu)


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
        print("Momentum ", ", ".join(f"{x:.2f}" for x in final_result))