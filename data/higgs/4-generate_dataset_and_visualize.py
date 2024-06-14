import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
from datetime import timedelta, datetime
from scipy import sparse

story = '714'
file_path = './digg_votes_filted_{0}.csv'.format(story)
data = pd.read_csv(file_path)
day_add_record = {}
user_list = data['voter_id'].tolist()

for index, info in tqdm(data.iterrows()):
    vote_time = info['vote_date']
    vote_time = datetime.fromtimestamp(vote_time)
    vote_time = '{0}-{1}-{2}-{3}'.format(vote_time.year, vote_time.month, vote_time.day, vote_time.hour)
    if not vote_time in day_add_record:
        day_add_record[vote_time] = []
    day_add_record[vote_time].append(info['voter_id'])
sorted_day = sorted(day_add_record, key=lambda x: datetime.strptime(x, '%Y-%m-%d-%H'))
start_time = current_time = datetime.strptime(sorted_day[30], "%Y-%m-%d-%H")
end_time = datetime.strptime(sorted_day[-1], "%Y-%m-%d-%H")
diff = []
infected_people = []
sum_num = []
while current_time <= end_time:
    current_day = '{0}-{1}-{2}-{3}'.format(current_time.year, current_time.month, current_time.day, current_time.hour)
    if current_day in day_add_record:
        diff.append(len(day_add_record[current_day]))
        new_infected = list(set(day_add_record[current_day]))
        if not len(infected_people) == 0:
            new_infected = list(set(day_add_record[current_day] + infected_people[-1]))
        infected_people.append(new_infected)
    else:
        if not len(infected_people) == 0:
            infected_people.append(infected_people[-1])
    current_time += timedelta(hours=1)
for index, day in enumerate(infected_people):
    sum_num.append(len(day))
plt.plot(sum_num)
plt.legend()
plt.show()
plt.plot(diff)
plt.legend()
plt.show()

feature_graph = np.zeros((len(user_list), len(infected_people)))
for index, day in enumerate(infected_people):
    feature_graph[day, index] = 1
feature_graph = sparse.csr_matrix(feature_graph)
sparse.save_npz('./feature_graphs.npz', feature_graph)
