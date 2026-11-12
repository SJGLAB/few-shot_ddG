import numpy as np
import argparse
import pandas as pd
import scipy.stats
from scipy.optimize import differential_evolution,shgo
import pickle

np.set_printoptions(precision=3)
first_col = 6

def calculate_val(df,a,b,c,d,col1,col2,col3,col4):
    one = df['ddG_true'].values
    two = a * df[col1].values + b * df[col2].values + c * df[col3].values + d * df[col4].values
    val = scipy.stats.spearmanr(one, two)[0]
    ans = val
    return ans

df = pd.read_csv('train_df.csv')
big_val_df = pd.read_csv('val_df.csv')
pdb_ids = df['pdb_id'].unique()

col1 = 'esm2-650m'
col2 = 'FoldX'
col3 = 'MPNNddG'
col4 = 'Rasp'
bounds = [(-10, 10),(-10, 10),(-10, 10),(-10, 10)]

def objective(params,train_df,col1,col2,col3,col4):
    a, b, c, d = params
    ans = -calculate_val(train_df,a,b,c,d,col1,col2,col3,col4)
    return ans

num_samples = [5,10,20,30,40,50,60,70,80,90,100,120,150,200,250,300]
result_df = pd.DataFrame(columns=['pdb_id', 'num_sample', 'train_ans', 'val_ans'])
for num_sample in num_samples:
    for pdb_id in pdb_ids:
        pdb_df = df[df['pdb_id'] == pdb_id]
        if num_sample > pdb_df.shape[0]:
            continue
        train_df = pdb_df.sample(n=num_sample, random_state=42)
        result = differential_evolution(
            objective, 
            bounds,
            args=(train_df,col1,col2,col3,col4),
            maxiter=300, 
            popsize=15,
            seed=42
        )
        a,b,c,d = result.x
        train_ans = calculate_val(train_df,a,b,c,d,col1,col2,col3,col4)
        val_df = big_val_df[big_val_df['pdb_id'] == pdb_id]
        val_ans = calculate_val(val_df,a,b,c,d,col1,col2,col3,col4)
        print(train_df.shape,val_df.shape)
        print('Result:',num_sample,pdb_id,col1,col2,col3,col4,'%.5f'%train_ans,'%.5f'%val_ans,flush=True)
        result_df.loc[len(result_df)] = [pdb_id, num_sample, train_ans, val_ans]

print(result_df)
result_df.to_csv('result.csv')

