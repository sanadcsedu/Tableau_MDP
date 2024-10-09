# from environment5 import environment5
from read_data_old import read_data
from collections import defaultdict
import pandas as pd 
import numpy as np
from tabulate import tabulate
import math 

class rewards_v2:
    def __init__(self):
        self.birdstrikes_relevant = {'"dam_eng1"': 1.0, '"dam_eng2"': 1.0, '"dam_windshld"': 1.0, '"dam_wing_rot"': 1.0, '"number of records"': 0.88, '"damage"': 0.94, '"ac_class"': 1.0, '"incident_date"': 0.94, '"precip"': 1.0, '"sky"': 1.0, '"phase_of_flt"': 0.35, '"ac_mass"': 0.24, '"state"': 0.35, '"size"': 0.35, '"birds_struck"': 0.47, '"time_of_day"': 0.47, '"birds_seen"': 0.24, '"distance"': 0.29, '"height"': 0.24, '"dam_eng3"': 0.35, '"dam_tail"': 0.29, '"dam_nose"': 0.24, '"dam_lghts"': 0.29, '"dam_lg"': 0.24, '"dam_fuse"': 0.24, '"dam_eng4"': 0.29, '"dam_other"': 0.24, '"warned"': 0.24, '"cost_repairs"': 0.24, '"speed"': 0.24, '"faaregion"': 0.24, '"location"': 0.29, '"latitude (generated)"': 0.24, '"longitude (generated)"': 0.24}
        self.weather_relevant = {'"heavyfog"': 1.0, '"number of records"': 0.38, '"date"': 1.0, '"tmax_f"': 0.81, '"tmin_f"': 0.75, '"latitude (generated)"': 0.69, '"longitude (generated)"': 0.69, '"lat"': 0.75, '"lng"': 0.75, '"state"': 0.94, '"freezingrain"': 0.5, '"blowingsnow"': 0.56, '"blowingspray"': 0.56, '"drizzle"': 1.0, '"dust"': 0.5, '"fog"': 0.56, '"mist"': 0.94, '"groundfog"': 0.94, '"freezingdrizzle"': 0.44, '"glaze"': 0.44, '"hail"': 0.5, '"highwinds"': 0.94, '"icefog"': 0.56, '"icepellets"': 0.5, '"prcp"': 0.81, '"rain"': 0.75, '"smoke"': 0.44, '"tmax"': 0.75, '"tmin"': 0.62, '"name"': 0.38, '"snow"': 0.5, '"snowgeneral"': 0.38, '"snwd"': 0.25, '"thunder"': 0.38, '"tornado"': 0.31}
        self.faa_relevant = {'"cancelled"': 1.0, '"diverted"': 1.0, '"arrdelay"': 1.0, '"depdelay"': 0.87, '"flightdate"': 1.0, '"airtime"': 0.27, '"uniquecarrier"': 1.0, '"distance"': 1.0, '"origin"': 0.47, '"number of records"': 1.0, '"dest"': 0.47, '"cancellationcode"': 0.27, '"latitude (generated)"': 0.53, '"longitude (generated)"': 0.53, '"origincityname"': 0.47, '"securitydelay"': 0.27, '"destcityname"': 0.27}

    def get_reward(self, dataset, attribute):
        if dataset == 'birdstrikes1':
            if attribute in self.birdstrikes_relevant:
                return self.birdstrikes_relevant[attribute]
            return 0
        elif dataset == 'weather1':
            if attribute in self.weather_relevant:
                return self.weather_relevant[attribute]
            return 0
        else:
            if attribute in self.faa_relevant:
                return self.faa_relevant[attribute]
            return 0
        
class rewards_v1:
    def __init__(self):
        self.weather_relevant = ['"heavyfog"', '"date"', '"tmax_f"', '"tmin_f"', '"latitude (generated)"', '"longitude (generated)"', '"lat"', '"lng"', '"state"', '"freezingrain"', '"blowingsnow"', '"blowingspray"', '"drizzle"', '"dust"', '"fog"', '"mist"', '"groundfog"', '"freezingdrizzle"', '"glaze"', '"hail"', '"highwinds"', '"icefog"', '"icepellets"', '"prcp"', '"rain"', '"smoke"', '"tmax"', '"tmin"', '"name"', '"snow"', '"snowgeneral"', '"snwd"', '"thunder"', '"tornado"']
        self.birdstrikes_relevant = ['"dam_eng1"', '"dam_eng2"', '"dam_windshld"', '"dam_wing_rot"', '"number of records"', '"damage"', '"ac_class"', '"incident_date"', '"precip"', '"sky"', '"phase_of_flt"', '"ac_mass"', '"state"', '"size"', '"birds_struck"', '"time_of_day"', '"birds_seen"', '"distance"', '"height"', '"dam_eng3"', '"dam_tail"', '"dam_nose"', '"dam_lghts"', '"dam_lg"', '"dam_fuse"', '"dam_eng4"', '"dam_other"', '"warned"', '"cost_repairs"', '"speed"', '"faaregion"', '"location"', '"latitude (generated)"', '"longitude (generated)"']
        self.faa_relevant = ['"cancelled"', '"diverted"', '"arrdelay"', '"depdelay"', '"flightdate"', '"airtime"', '"uniquecarrier"', '"distance"', '"origin"', '"number of records"', '"dest"', '"cancellationcode"', '"latitude (generated)"', '"longitude (generated)"', '"origincityname"', '"securitydelay"', '"destcityname"']

    def get_reward(self, dataset, attribute):
        if dataset == 'birdstrikes1':
            if attribute in self.birdstrikes_relevant:
                return 1
            return 0
        elif dataset == 'weather1':
            if attribute in self.weather_relevant:
                return 1
            return 0
        else:
            if attribute in self.faa_relevant:
                return 1
            return 0   


# class nlp:
#     def __init__(self):
#         pass
#     def tf_idf(self, dataset):
#         pass 
#     def get_tf_idf_values(self, df):
#         #term frequency
#         f =  open('output.txt', 'w')
#         f.write(tabulate(df, headers='keys', tablefmt='psql'))
        
#         row, col = df.shape
#         total_word_count = df.sum(axis = 1)
#         df = df.div(total_word_count, axis=0)
#         print(total_word_count)
        
#         f.write(tabulate(df, headers='keys', tablefmt='psql'))
    
#     def create_idf(self, d):
#         env = environment5()
#         user_list = env.obj.get_user_list_for_dataset(d)
#         obj = read_data()
#         obj.create_connection(r"Tableau.db")

#         af = defaultdict(int) # af contains document frequency, i.e., number of users using this attribute at least once
#         for user in user_list:
#             data = obj.merge2(d, user[0])
#             attribute_user = defaultdict()
#             for itrs in data:
#                 itrs = itrs.strip('[]')
#                 states = itrs.split(', ')
#                 for s in states:
#                     if len(s) > 0:
#                         attribute_user[s] = 1
#             for items, values in attribute_user.items():
#                 af[items] += 1
#         inverse_af = defaultdict(int)
#         N = len(user_list)
#         for items, values in af.items():
#             inverse_af[items] = math.log10(N / values) # log(N/df_t) for each term t
#             if inverse_af[items] >= 1:
#                 del inverse_af[items]

#         # items_to_remove = [items for items, values in attribute_dict.items() if values < 4]
#         # for items in items_to_remove:
#         #     del attribute_dict[items]
#         # print(attribute_dict)
#         for items, values in inverse_af.items():
#             print(items, values)
#             # print(items, end=",")

#     def create_df(self, d):
#         env = environment5()
#         user_list = env.obj.get_user_list_for_dataset(d)

#         #Finding all possible attributes a user can navigate to: [columns]
#         columns = ['users']
#         list_of_all_attrs = set()
#         obj = read_data()
#         obj.create_connection(r"Tableau.db")
#         for user in user_list:
#             data = obj.merge2(d, user[0])
#             for itrs in data:
#                 itrs = itrs.strip('[]')
#                 states = itrs.split(', ')
#                 for s in states:
#                     if len(s) > 0:
#                         list_of_all_attrs.add(s)
#         # all_attributes = list(list_of_all_attrs)
#         for attrs in list_of_all_attrs:
#             columns.append(attrs)
#         print(len(columns))
#         print(columns)

#         df = pd.DataFrame(columns=columns)

#         for rowidx, user in enumerate(user_list):
#             df.loc[rowidx, 'users'] = user[0]
#             data = obj.merge2(d, user[0])        
#             attribute_freq = defaultdict(int)
#             for itrs in data:
#                 itrs = itrs.strip('[]')
#                 states = itrs.split(', ')
#                 for s in states:
#                     if len(s) > 0:
#                         attribute_freq[s] += 1
#             for items, values in attribute_freq.items():
#                 df.loc[rowidx, items] = values
#         df = df.replace(np.nan, 0)
#         # print(df.head())
#         # with open('output.txt', 'w') as f:
#         #    f.write(tabulate(df, headers='keys', tablefmt='psql'))
#         self.get_tf_idf_values(df)
    
#     def frequency(self, d):
#         env = environment5()
#         user_list = env.obj.get_user_list_for_dataset(d)
#         obj = read_data()
#         obj.create_connection(r"Tableau.db")

#         af = defaultdict(int) # af contains document frequency, i.e., number of users using this attribute at least once
#         for user in user_list:
#             data = obj.merge2(d, user[0])
#             attribute_user = defaultdict()
#             for itrs in data:
#                 itrs = itrs.strip('[]')
#                 states = itrs.split(', ')
#                 for s in states:
#                     if len(s) > 0:
#                         attribute_user[s] = 1
#             for items, values in attribute_user.items():
#                 af[items] += 1
        
#         items_to_remove = [items for items, values in af.items() if values < 4]
#         for items in items_to_remove:
#             del af[items]
#         denom = sum(af.values())
#         for items, values in af.items():
#             print(items, values)
#         print("-------")
#         for items, values in af.items():
#             af[items] = round(values/len(user_list), 2)
#             # print("{} {}".format(items, round(values/len(user_list), 2)))
#         # for items, values in af.items():
#         #     if values >= 4:
#         #         print("'", end="")
#         #         print(items, end = "', ")
#         print(af)

# if __name__ == "__main__":    
#     obj = nlp()
#     env = environment5()
#     # env.process_data('birdstrikes1', 81, 0.1, "A")
#     datasets = env.datasets
#     for d in datasets:
#         print("------", d, "-------")
#         obj.frequency(d)


# for user in user_list:
#             data = obj.merge2(d, user[0])
#             attribute_user = defaultdict()
#             for itrs in data:
#                 itrs = itrs.strip('[]')
#                 states = itrs.split(', ')
#                 for s in states:
#                     attribute_user[s] = 1
#             for items, values in attribute_user.items():
#                 attribute_dict[items] += 1
        
#         items_to_remove = [items for items, values in attribute_dict.items() if values < 4]
#         for items in items_to_remove:
#             del attribute_dict[items]
#         # print(attribute_dict)
#         for items, values in attribute_dict.items():
#             print(items, end=",")