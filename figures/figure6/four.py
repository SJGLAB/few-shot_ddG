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

add_sample = 4

def objective(params,train_df,col1,col2,col3,col4):
    a, b, c, d = params
    ans = -calculate_val(train_df,a,b,c,d,col1,col2,col3,col4)
    return ans

def update_train_df_one(train_df, unsampled_df, feature_cols=[col1,col2,col3,col4], standardize=True):
    X = train_df[feature_cols].to_numpy(dtype=float)

    if standardize:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0

        X = (X - mean) / std
        X_pool = (unsampled_df[feature_cols].to_numpy(dtype=float) - mean) / std
    else:
        X_pool = unsampled_df[feature_cols].to_numpy(dtype=float)

    M = X.T @ X
    M_inv = np.linalg.pinv(M)

    scores = np.einsum("ij,jk,ik->i", X_pool, M_inv, X_pool)
    target_pos = np.argmax(scores)

    target_index = unsampled_df.index[target_pos]
    selected_row = unsampled_df.loc[[target_index]]
    #tpx = (selected_row[feature_cols].to_numpy(dtype=float) - mean) / std
    #score = np.einsum("ij,jk,ik->i", tpx, M_inv, tpx)
    #print(score)

    train_df = pd.concat([train_df, selected_row])
    unsampled_df = unsampled_df.drop(index=target_index)
    return train_df, unsampled_df

def update_train_df(train_df, unsampled_df):
    for i in range(add_sample):
        train_df, unsampled_df = update_train_df_one(train_df, unsampled_df)
    return train_df, unsampled_df

result_df = pd.DataFrame(columns=['pdb_id', 'num_sample', 'train_ans', 'val_ans'])

for pdb_id in pdb_ids:
    pdb_df = df[df['pdb_id'] == pdb_id]
    num_sample = 4

    if num_sample > pdb_df.shape[0]:
        continue
    train_df = pdb_df.sample(n=num_sample, random_state=42)
    unsampled_df = pdb_df.drop(train_df.index)

    for iloop in range(20):
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

        num_sample += add_sample
        train_df,unsampled_df = update_train_df(train_df,unsampled_df)

print(result_df)
result_df.to_csv('result.csv')

