# import Categorizing
import sqlite3
import pdb
import math
from Categorizing_v4 import Categorizing
# from Reward_Generator import reward

class read_data:

    def __init__(self):
        # self.conn = self.create_connection(r"Tableau.db")
        self.conn = None
        self.tasks = ['t1', 't2', 't3', 't4']

    # Creating a connection with the database using sqlite3
    def create_connection(self, db_file):
        try:
            self.conn = sqlite3.connect(db_file)
            # print("Connection Created")
        except sqlite3.Error as e:
            print(e)

    # Read all the information form the database and store important ones inside the class for current user
    def read_cur_data(self, userid, dataset, task):
        c = self.conn.cursor()
        # query = 'SELECT userid, task, seqId, state, timestamp FROM master_file where dataset = ' + '\'' + str(dataset) + '\'' + ' and userid = ' + str(userid) + ' and task = \'' + str(task) + '\''
        query = 'SELECT state FROM master_file where dataset = ' + '\'' + str(dataset) + '\'' + ' and userid = ' + str(userid) + ' and task = \'' + str(task) + '\'' + ' order by seqId'
        # print(query)
        c.execute(query)
        cur_data = c.fetchall()
        # print(cur_data)
        return cur_data

    def get_user_list_for_dataset(self, dataset):
        # self.conn = self.create_connection(r"Tableau.db")
        c = self.conn.cursor()
        query = 'select distinct(userid)  from master_file where dataset = ' + '\'' + str(dataset) + '\''
        # print(query)
        c.execute(query)
        cur_data = c.fetchall()
        return cur_data

    def merge2(self, dataset, user):
        # merging interactions of all tasks, contains attrubutes used in each interaction 
        interactions = []
        for t in self.tasks:
            data = self.read_cur_data(user, dataset, t)
            for itrs in data:
                interactions.append(itrs[0])
        return interactions 
    
if __name__ == '__main__':
    obj = read_data()
    obj.create_connection(r"Tableau.db")
    datasets = ['birdstrikes1', 'weather1', 'faa1']
    tasks = ['t1', 't2', 't3', 't4']
    for d in datasets:
        users = obj.get_user_list_for_dataset(d)
        cat = Categorizing(d)
        # r = reward()
        for u in users:
            user = u[0]
            data = obj.merge2(d, user)
            # raw_states, raw_actions, mem_reward = r.generate(data, d)
            # for idx, states in enumerate(raw_states):
            #     high_level_attrs = cat.get_category(states, d)
            #     print(high_level_attrs, raw_actions[idx])
            # print(len(interactions))
            # print(" - ", len(interactions))
            pdb.set_trace()