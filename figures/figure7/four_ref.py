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

df = pd.read_csv('../../megascale_train_summary.csv')
apdbs = ['r10_437_TrROS_Hall','r10_437_TrROS_Hall','r10_437_TrROS_Hall']
num_pdbs = len(apdbs)

bdf = pd.read_csv('valtest.csv')
bpdbs = ['2M9E','2KXD','1TUC']

col1 = 'esm2-650m'
col2 = 'FoldX'
col3 = 'MPNNddG'
col4 = 'Rasp'
bounds = [(-10, 10),(-10, 10),(-10, 10),(-10, 10)]

def objective(params,obj_df,col1,col2,col3,col4):
    a, b, c, d = params
    ans = -calculate_val(obj_df,a,b,c,d,col1,col2,col3,col4)
    return ans

result_df = pd.DataFrame(columns=['apdb', 'bpdb', 'num_sample', 'train_ans', 'val_ans'])
for i in range(num_pdbs):
    apdb = apdbs[i]
    bpdb = bpdbs[i]

    pdb_df = df[df['pdb_id'] == apdb]
    val_df = bdf[bdf['pdb_id'] == bpdb]

    num_sample = 500
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
    val_ans = calculate_val(val_df,a,b,c,d,col1,col2,col3,col4)
    print(train_df.shape,val_df.shape)
    print('Result:',num_sample,apdb,bpdb,col1,col2,col3,col4,'%.5f'%train_ans,'%.5f'%val_ans,flush=True)
    result_df.loc[len(result_df)] = [apdb, bpdb, num_sample, train_ans, val_ans]

print(result_df)
result_df.to_csv('result_ref.csv')

