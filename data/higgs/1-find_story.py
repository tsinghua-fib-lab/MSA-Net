import numpy as np
import pandas as pd
import csv


def read_csv_file(file_path):
    rows = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            rows.append(row)
    return rows


# 示例文件路径
file_path = './digg_votes.csv'

# 逐行读取CSV文件
data = read_csv_file(file_path)
pd_data = pd.DataFrame(data, columns=['vote_date', 'voter_id', 'story_id'])
grouped_data = pd_data.groupby('story_id')
group_sizes = {}
for group in grouped_data.indices:
    group_sizes[group] = len(np.unique(grouped_data.indices[group]))


def sort_by_value(item):
    return item[1]  # 按照值进行排序


# 按照值排序，并返回一个元组的列表
sorted_data = sorted(group_sizes.items(), key=sort_by_value)
print(sorted_data[0], sorted_data[-1], sorted_data[int(len(sorted_data) / 2)])
