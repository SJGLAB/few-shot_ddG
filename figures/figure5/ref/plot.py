import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv('train_df.csv')
val_df = pd.read_csv('val_df.csv')
pdb_ids = train_df['pdb_id'].unique()

plt.figure(figsize=(14,6))
plt.subplot(121)
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

plt.subplot(122)
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

plt.savefig('case-by-case.png',bbox_inches='tight',pad_inches=0.1)
