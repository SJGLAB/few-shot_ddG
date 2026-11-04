import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman', size=15)

df = pd.read_csv('result.csv')
df = df[df['num_sample']<=40]
df_random = pd.read_csv('result_random.csv')
df_random = df_random[df_random['num_sample']<=40]
nums = df['num_sample'].unique()

mean_lst = []
mean_random_lst = []
std_lst = []
std_random_lst = []

for num in nums:
    mean_lst.append(df[df['num_sample']==num]['val_ans'].values.mean())
    std_lst.append(df[df['num_sample']==num]['val_ans'].values.std())
    mean_random_lst.append(df_random[df_random['num_sample']==num]['val_ans'].values.mean())
    std_random_lst.append(df_random[df_random['num_sample']==num]['val_ans'].values.std())

mean_lst = np.array(mean_lst)
mean_random_lst = np.array(mean_random_lst)
std_lst = np.array(std_lst)
std_random_lst = np.array(std_random_lst)
plt.plot(nums,mean_lst,'-o')
#plt.fill_between(
#    nums,
#    mean_lst - std_lst,
#    mean_lst + std_lst,
#    alpha=0.2,
#    label="±1 SD"
#)
plt.plot(nums,mean_random_lst,'--s')
#plt.fill_between(
#    nums,
#    mean_random_lst - std_random_lst,
#    mean_random_lst + std_random_lst,
#    alpha=0.2,
#    label="±1 SD"
#)
plt.legend(['Active Learning','Random'])
plt.xlabel('Number of samples')
plt.ylabel('|SCC|')

plt.savefig('D-optimal.png',bbox_inches='tight',pad_inches=0.1)
