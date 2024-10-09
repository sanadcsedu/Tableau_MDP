from read_data_old import read_data
import pdb
from collections import defaultdict
from Reward_Generator import reward 
import ast
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np 
import numpy as np 
import pymc as pm
import os
# os.environ["THEANO_FLAGS"] = "blas__ldflags=,mkl__ldflags=,blas__check_openmp=False"


class bayesian:
    def __init__(self) -> None:
        pass
    
    def encoding(self, file):
        all_attributes = []
        all_actions = set()

        data_dataframe = []

        for line in file:
            parts = line.split(",")
            parts[1] = parts[1].replace(";", ",")
            all_actions.add(int(parts[0]))
            list_part = ast.literal_eval(parts[1])
            for attrs in list_part:
                all_attributes.append(attrs)
            if len(list_part) == 0:
                continue
            # data_dataframe.append([int(parts[0]), list_part])

        unique_attributes = set(all_attributes)
        attribute_encoder = LabelEncoder().fit(list(unique_attributes))

        for line in file:
            parts = line.split(",")
            parts[1] = parts[1].replace(";", ",")
            list_part = ast.literal_eval(parts[1])
            if len(list_part) == 0:
                continue
            x = attribute_encoder.transform(list_part)
            encoded_interaction = ''.join(str(p) for p in sorted(x))
            data_dataframe.append([int(parts[0]), encoded_interaction])
            
        # print(attribute_encoder.classes_)
        data = pd.DataFrame(data_dataframe, columns=['Action', 'Attribute'])

        interaction_encoder = LabelEncoder().fit(data['Attribute'])
        print(interaction_encoder.classes_)

        action_encoder = LabelEncoder().fit(data['Action'])
        print(action_encoder.classes_)

        data['Encoded_Attribute'] = attribute_encoder.fit_transform(data['Attribute'])
        data['Action'] = action_encoder.fit_transform(data['Action'])

        return data

    def run_bayesian(self, file):
        
        data = self.encoding(file)

        threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = []
        for t in threshold:
            # Split the dataset into training and testing sets
            split = int(len(data) * t)
            train_data = data[:split]
            test_data = data[split:]

            sequence = []
            # encoded_actions = action_encoder.transform(data['Action'])
            # print(encoded_actions)
            for aa, bb in zip(data['Action'], data['Encoded_Attribute']):
                sequence.append([bb, aa])
            # sequences = np.array(sequence[:split])
            sequences = np.array(sequence)
            states = sequences[:, 0]
            actions = sequences[:, 1]

            n_states = len(np.unique(states))
            n_actions = len(np.unique(actions))

            states = sequences[:split, 0]
            actions = sequences[:split, 1]

            with pm.Model() as model:
                # Priors for emission probabilities for each state
                emissions_p = pm.Dirichlet('emissions_p', a=np.ones(n_actions), shape=(n_states, n_actions))
                
                # Model each action as being drawn from a categorical distribution with probabilities
                # determined by the state it's in
                action_obs = pm.Categorical('action_obs', p=emissions_p[states], observed=actions)
                
                # Perform inference
                trace = pm.sample(1000, return_inferencedata=True, cores=2)
            
            accu = []
            for idx, rows in test_data.iterrows():
                current_state = rows['Encoded_Attribute']
                # Calculate the posterior mean emission probabilities for the current state
                posterior_means = trace.posterior['emissions_p'].mean(dim=['chain', 'draw']).values

                # The predicted action is the one with the highest mean probability for the current state
                predicted_action = np.argmax(posterior_means[current_state])

                # print(f"Predicted action for state {current_state}: {predicted_action}")
                if(predicted_action == rows['Action']):
                    accu.append(1)
                else:
                    accu.append(0)
            results.append(np.mean(accu))
        # print(results)
        return results

if __name__ == '__main__':
    obj = read_data()
    obj.create_connection(r"Tableau.db")
    r = reward()
    for d in r.datasets:
        users = obj.get_user_list_for_dataset(d)
        for user in users:
            u = user[0]
            data = obj.merge2(d, u)    
            raw_states, raw_actions, mem_reward = r.generate(data, d)
            cleaned_data = []
            for i in range(len(raw_states)):
                # print(f"{i}, {raw_actions[i]}, {raw_states[i]}, {mem_reward[i]}")
                temp = str(raw_actions[i]) + ",["
                # print(raw_actions[i], end = ",[")
                for idx, s in enumerate(raw_states[i]):
                    if idx == len(raw_states[i]) - 1:
                        # print(s, end="")
                        temp += s
                    else:
                        # print(s, end=";")
                        temp += s + ";"
                # print("]",)
                temp += "]"
                cleaned_data.append(temp)
                print(temp)
            obj = bayesian()
            print(obj.run_bayesian(cleaned_data)) 
            pdb.set_trace()
            # print(len(raw_states), len(raw_actions), len(mem_reward))