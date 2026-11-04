import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(22,12))

train_df = pd.read_csv('train_df.csv')
val_df = pd.read_csv('val_df.csv')
pdb_ids = train_df['pdb_id'].unique()

plt.subplot(221)
train_num_lst = []
val_num_lst = []
num_lst = []

train_pos_lst = []
val_pos_lst = []
pos_lst = []
icount = 0
for pdb_id in pdb_ids:
    pdb_df = train_df[train_df['pdb_id'] == pdb_id]
    pdb_val_df = val_df[val_df['pdb_id'] == pdb_id]
    print(pdb_id,pdb_df.shape,pdb_val_df.shape)

    train_pos_lst.append(icount)
    train_num_lst.append(pdb_df.shape[0])
    val_pos_lst.append(icount+1)
    val_num_lst.append(pdb_val_df.shape[0])

    pos_lst.append(icount)
    num_lst.append(pdb_df.shape[0])
    pos_lst.append(icount+1)
    num_lst.append(pdb_val_df.shape[0])
    icount += 3

bars = plt.barh(train_pos_lst, train_num_lst)
plt.bar_label(bars, fmt='%d', padding=3)
plt.yticks(ticks=train_pos_lst,labels=pdb_ids)
plt.xlim(180)

plt.subplot(223)
df = pd.read_csv('result.csv')
num_samples = df['num_sample'].unique()
num_lst = []
val_lst = []
for num_sample in num_samples:
    tp_df = df[df['num_sample'] == num_sample]
    val = tp_df['val_ans'].values

    num_lst.append(num_sample)
    val_lst.append(np.mean(val))
    print(num_sample,len(val),np.mean(val))

plt.plot(num_lst,val_lst,'-*')

df = pd.read_csv('result.csv')
df_other = pd.read_csv('result_by_other.csv')

max_indices = df.groupby('pdb_id')['num_sample'].idxmax()
result_max_df = df.loc[max_indices]
print(result_max_df)
print(df_other)

df_other['val_ans'] = 0.0
for idx,row in df_other.iterrows():
    df_other.at[idx, 'val_ans'] = result_max_df[result_max_df['pdb_id'] == row.pdb_id]['val_ans'].values[0]
print(df_other)

lst = ['val_ans','val_ans_other']
dat = []

ax_b = plt.subplot(222)
count = 0
colors = ['C0', 'C2']
for col in lst:
    plt.scatter(df_other[col].values,list(df_other[col].index+1),c=colors[count])
    dat.append(df_other[col].values)
    count += 1
plt.yticks(ticks=list(df_other['pdb_id'].index+1),labels=list(df_other['pdb_id'].values))
plt.legend(lst)

plt.subplot(224,sharex=ax_b)
bplot = plt.boxplot(dat, tick_labels=lst, vert=False, showfliers=False, patch_artist=True, medianprops={'color': 'C1', 'linewidth': 2})
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
yones = np.ones(df_other.shape[0])
plt.scatter(df_other['val_ans'].values,yones,c='C0')
plt.scatter(df_other['val_ans_other'].values,yones*2,c='C2')

plt.savefig('case-by-case_summary.png',bbox_inches='tight',pad_inches=0.1)
