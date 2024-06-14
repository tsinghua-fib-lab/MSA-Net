import numpy as np
import pickle as pkl
import datetime
from tqdm import tqdm

file_path = './higgs-activity_time.txt'
infected_record = {'original': []}
infected_users = []


def retweet(info):
    timestamp = int(info['time'])
    timestamp = datetime.datetime.fromtimestamp(timestamp)
    timestamp = timestamp.strftime("%Y-%m-%d-%H")
    if not timestamp in infected_record.keys():
        infected_record[str(timestamp)] = []
    infected_record[timestamp].append(info['user1'])
    infected_users.append(info['user1'])
    if not info['user2'] in infected_users:
        infected_record['original'].append(info['user2'])
        infected_users.append(info['user2'])


def mention(info):
    timestamp = int(info['time'])
    timestamp = datetime.datetime.fromtimestamp(timestamp)
    timestamp = timestamp.strftime("%Y-%m-%d-%H")
    if not timestamp in infected_record.keys():
        infected_record[str(timestamp)] = []
    infected_record[timestamp].append(info['user2'])
    infected_users.append(info['user2'])
    if not info['user1'] in infected_users:
        infected_record['original'].append(info['user1'])
        infected_users.append(info['user1'])


def reply(info):
    pass


with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in tqdm(lines):
        line = line.strip()  # 去除行尾的换行符和空格
        elements = line.split()
        info = {
            'user1': elements[0],
            'user2': elements[1],
            'time': elements[2],
            'act': elements[3]
        }
        if info['act'] == 'RT':
            retweet(info)
        if info['act'] == 'MT':
            mention(info)
        if info['act'] == 'RE':
            reply(info)
np.save('./infected_users.npy', np.array(infected_users))
with open('./infected_records.pkl', 'wb') as f:
    pkl.dump(infected_record, f)
