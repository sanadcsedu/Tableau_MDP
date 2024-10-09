from read_data import read_data
import pdb
from collections import defaultdict
from Categorizing_v4 import Categorizing

class reward:
    def __init__(self) -> None:
        self.obj = read_data()
        self.obj.create_connection(r"Tableau.db")
        self.datasets = ['birdstrikes1', 'weather1', 'faa1']
        self.tasks = ['t1', 't2', 't3', 't4']
        self.attributes = defaultdict(lambda: defaultdict(float))
    
        self.rewards = defaultdict(lambda: defaultdict(float))
        self.rewards['birdstrikes1'] = {'"dam_eng1"': 1.0, '"dam_eng2"': 1.0, '"dam_windshld"': 1.0, '"dam_wing_rot"': 1.0, '"damage"': 0.94, '"ac_class"': 1.0, '"incident_date"': 0.94, '"precip"': 1.0, '"sky"': 1.0, '"atype"': 0.18, '"phase_of_flt"': 0.35, '"operator"': 0.18, '"ac_mass"': 0.24, '"state"': 0.35, '"size"': 0.35, '"birds_struck"': 0.47, '"time_of_day"': 0.47, '"type_eng"': 0.18, '"birds_seen"': 0.24, '"distance"': 0.29, '"height"': 0.24, '"dam_eng3"': 0.35, '"indicated_damage"': 0.18, '"dam_tail"': 0.29, '"dam_nose"': 0.24, '"dam_lghts"': 0.29, '"dam_lg"': 0.24, '"dam_fuse"': 0.24, '"dam_eng4"': 0.29, '"dam_other"': 0.24, '"warned"': 0.24, '"cost_repairs"': 0.24, '"dam_prop"': 0.18, '"dam_rad"': 0.18, '"index_nr"': 0.18, '"speed"': 0.24, '"incident_month"': 0.18, '"airport"': 0.18, '"species"': 0.18, '"faaregion"': 0.24, '"location"': 0.29, '"latitude (generated)"': 0.24, '"longitude (generated)"': 0.24}
        self.rewards['weather1'] = {'"heavyfog"': 1.0, '"date"': 1.0, '"tmax_f"': 0.81, '"tmin_f"': 0.75, '"latitude (generated)"': 0.69, '"longitude (generated)"': 0.69, '"lat"': 0.75, '"lng"': 0.75, '"state"': 0.94, '"freezingrain"': 0.5, '"blowingsnow"': 0.56, '"blowingspray"': 0.56, '"drizzle"': 1.0, '"dust"': 0.5, '"fog"': 0.56, '"mist"': 0.94, '"groundfog"': 0.94, '"elevation"': 0.19, '"freezingdrizzle"': 0.44, '"glaze"': 0.44, '"hail"': 0.5, '"highwinds"': 0.94, '"icefog"': 0.56, '"icepellets"': 0.5, '"prcp"': 0.81, '"rain"': 0.75, '"smoke"': 0.44, '"tmax"': 0.75, '"tmin"': 0.62, '"name"': 0.38, '"snow"': 0.5, '"snowgeneral"': 0.38, '"snwd"': 0.25, '"thunder"': 0.38, '"tornado"': 0.31}
        self.rewards['faa1'] = {'"cancelled"': 1.0, '"diverted"': 1.0, '"arrdelay"': 1.0, '"depdelay"': 0.87, '"flightdate"': 1.0, '"airtime"': 0.27, '"uniquecarrier"': 1.0, '"distance"': 1.0, '"origin"': 0.47, '"dest"': 0.47, '"cancellationcode"': 0.27, '"latitude (generated)"': 0.53, '"longitude (generated)"': 0.53, '"origincityname"': 0.47, '"carrierdelay"': 0.2, '"lateaircraftdelay"': 0.2, '"nasdelay"': 0.2, '"securitydelay"': 0.27, '"weatherdelay"': 0.2, '"destcityname"': 0.27}
        # print(len(self.rewards['birdstrikes1'], len(self.rewards['weather1'], len(self.rewards['faa1']))))
    
    def generate_reward(self):
        for d in self.datasets:
            users = self.obj.get_user_list_for_dataset(d)
            # print(users)
            for u in users:
                user = u[0]
                interactions = self.obj.merge2(d, user)
                
                temp_attrs = defaultdict(int)
                for itrs in interactions:
                    itrs = itrs.strip('[]')
                    states = itrs.split(', ')
                    for attrs in states:
                        temp_attrs[attrs] = 1
                for keys, values in temp_attrs.items():
                    self.attributes[d][keys] += 1
            
            #Normalize the Rewards / #Users 
            keys_to_delete = []
            for keys, values in self.attributes[d].items():
                if values < 3 or 'calculation' in keys: #At least 3 users need to use this attribute
                    keys_to_delete.append(keys)
                else:
                    self.attributes[d][keys] = round(self.attributes[d][keys] / len(users), 2)
            
            for keys in keys_to_delete:
                del self.attributes[d][keys]

            self.attributes[d].pop('', None)
            self.attributes[d].pop('"number of records"', None)

    def generate(self, data, dataset):
        interactions = []
        #Converting string interactions into python lists
        for itrs in data:
            # if len(itrs) <= 2:
            #     continue
            itrs = itrs.strip('[]')
            states = itrs.split(', ')
            interactions.append(states)

        mem_states = []
        mem_action = []
        mem_rewards = []
        #new one  
        for i in range(1, len(interactions)):
            current_interaction = set(interactions[i-1])
            next_interaction = set(interactions[i])
            diff = len(current_interaction.symmetric_difference(next_interaction))

            total_reward = 0
            for items in interactions[i-1]:
                if items in self.rewards[dataset]:
                    total_reward += self.rewards[dataset][items]

            # if diff > 3 and interactions[i] != ['']:
            if diff > 3:
                mem_action.append(4)
            else:
                mem_action.append(diff)
            
            mem_states.append(interactions[i-1])
            mem_rewards.append(total_reward)            

        return mem_states, mem_action, mem_rewards
    
        # for i in range(1, len(interactions)):
        #     diff = 0
        #     for items in interactions[i-1]:
        #         if items not in interactions[i]:
        #             diff += 1
        #     total_reward = 0
        #     for items in interactions[i-1]:
        #         if items in self.rewards[dataset]:
        #             total_reward += self.rewards[dataset][items]

        #     if diff > 3 and interactions[i] != ['']:
        #         mem_action.append(4)
        #     else:
        #         mem_action.append(diff)
            
        #     mem_states.append(interactions[i-1])
        #     mem_rewards.append(total_reward)            

        # return mem_states, mem_action, mem_rewards

if __name__ == '__main__':
    obj = read_data()
    obj.create_connection(r"Tableau.db")
    r = reward()
    for d in r.datasets:
        users = obj.get_user_list_for_dataset(d)
        cat = Categorizing(d)
        for user in users:
            u = user[0]
            data = obj.merge2(d, u)    
            raw_states, raw_actions, mem_reward = r.generate(data, d)
            for idx, states in enumerate(raw_states):
                high_level_attrs = cat.get_category(states, d)
                print("{};{}".format(high_level_attrs, raw_actions[idx]))
            pdb.set_trace()
            
            
# if __name__ == '__main__':
#     obj = read_data()
#     obj.create_connection(r"Tableau.db")
#     r = reward()
#     for d in r.datasets:
#         users = obj.get_user_list_for_dataset(d)
#         for user in users:
#             u = user[0]
#             data = obj.merge2(d, u)    
#             raw_states, raw_actions, mem_reward = r.generate(data, d)
#             for i in range(len(raw_states)):
#                 # print(f"{i}, {raw_actions[i]}, {raw_states[i]}, {mem_reward[i]}")
#                 print(raw_actions[i], end = ",[")
#                 for idx, s in enumerate(raw_states[i]):
#                     if idx == len(raw_states[i]) - 1:
#                         print(s, end="")
#                     else:
#                         print(s, end=";")
#                 print("]",)
#             pdb.set_trace()
#             # print(len(raw_states), len(raw_actions), len(mem_reward))