import environment5
import numpy as np
from collections import defaultdict
import json
import pandas as pd
import random
import multiprocessing

eps=1e-35
class Greedy:
    def __init__(self):
        """Initializes the Greedy model."""
        self.freq = defaultdict(lambda: defaultdict(float))
        self.reward = defaultdict(lambda: defaultdict(float))

    def GreedyDriver(self, env, thres):
        length = len(env.mem_action)
        threshold = int(length * thres)
        for i in range(threshold):
            self.freq[env.mem_states[i]][env.mem_action[i]] += 1

        # Checking accuracy on the remaining data:
        accuracy = 0
        denom = 0
        for i in range(threshold, length):
            denom += 1
            actions = list(self.freq[env.mem_states[i]].keys())
            frequencies = list(self.freq[env.mem_states[i]].values())
            try:
                # Sample uniformly based on the probability of all actions
                _max = random.choices(actions, weights=frequencies, k=1)[0]

            #if state is not observed in training data then take a random action
            except IndexError:
                _max = random.choice([0, 1, 2, 3, 4])
                
            if _max == env.mem_action[i]:
                 accuracy += 1

            self.freq[env.mem_states[i]][env.mem_action[i]] += 1

        accuracy /= denom
        return accuracy

class run_Greedy:
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
                    obj = Greedy()
                    avg_accu.append(obj.GreedyDriver(env, thres))
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

        obj2 = run_Greedy()

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
        print("Greedy ", ", ".join(f"{x:.2f}" for x in final_result))
