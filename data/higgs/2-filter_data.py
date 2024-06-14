import pandas as pd
import csv
from tqdm import tqdm

story = '714'
file_path = './digg_votes.csv'


def read_csv_file(file_path):
    rows = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            rows.append(row)
    return rows


# 寻找id数据
vote_data = read_csv_file(file_path)
vote_data = pd.DataFrame(vote_data, columns=['vote_date', 'voter_id', 'story_id'])
grouped_data = vote_data.groupby('story_id')
filted_data_index = grouped_data.indices[story]
filted_data = vote_data.iloc[filted_data_index]
filted_data.to_csv('./digg_votes_filted_{0}.csv'.format(story))
users = filted_data['voter_id'].tolist()

# 寻找关注关系
file_path = './digg_friends.csv'
edge_data = read_csv_file(file_path)
edge_data = pd.DataFrame(edge_data, columns=['mutual', 'friend_date', 'user_id', 'friend_id'])
edge_data = edge_data.drop('friend_date', axis=1)
user_id = []
friend_id = []
for index, edge in tqdm(edge_data.iterrows()):
    if edge['user_id'] in users and edge['friend_id'] in users:
        user_id.append(edge['user_id'])
        friend_id.append(edge['friend_id'])
filted_edge = pd.DataFrame({'user_id': user_id,
                            'friend_id': friend_id})
filted_edge.to_csv('digg_friends_filted_{0}.csv'.format(story))
print(1)
