import numpy as np
import argparse
import pandas as pd
import scipy.stats
from scipy.optimize import differential_evolution,shgo
import pickle

np.set_printoptions(precision=3)
first_col = 7

def calculate_val(df,a,b,c,d,col1,col2,col3,col4):
    one = df['ddG_true'].values
    two = a * df[col1].values + b * df[col2].values + c * df[col3].values + d * df[col4].values
    val = scipy.stats.spearmanr(one, two)[0]
    ans = abs(val)
    return ans

df = pd.read_csv('train_df.csv')
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

result_cof = {}
for pdb_id in pdb_ids:
    pdb_df = df[df['pdb_id'] == pdb_id]
    train_df = pdb_df.sample(n=500, random_state=42)

    result = differential_evolution(
        objective, 
        bounds,
        args=(train_df,col1,col2,col3,col4),
        maxiter=300, 
        popsize=15,
        seed=42
    )
    a,b,c,d = result.x
    result_cof[(col1,col2,col3,col4)] = (a,b,c,d)
    train_ans = calculate_val(train_df,a,b,c,d,col1,col2,col3,col4)
    print('Result:',pdb_id,col1,col2,col3,col4,'%.5f'%train_ans,flush=True)

with open('result_cof.pkl', 'wb') as f:
    pickle.dump(result_cof, f)

