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

big_val_df = pd.read_csv('val_df.csv')
pdb_ids = big_val_df['pdb_id'].unique()

col1 = 'esm2-650m'
col2 = 'FoldX'
col3 = 'MPNNddG'
col4 = 'Rasp'
result_df = pd.DataFrame(columns=['pdb_id', 'val_ans_other'])
with open('../../four/result_cof.pkl', 'rb') as f:  # 二进制读取
    result_cof = pickle.load(f)

for pdb_id in pdb_ids:
    a,b,c,d = result_cof[(col1,col2,col3,col4)]
    val_df = big_val_df[big_val_df['pdb_id'] == pdb_id]
    val_ans = calculate_val(val_df,a,b,c,d,col1,col2,col3,col4)
    print('Result:',pdb_id,col1,col2,col3,col4,'%.5f'%val_ans,flush=True)
    result_df.loc[len(result_df)] = [pdb_id, val_ans]

print(result_df)
result_df.to_csv('result_by_other.csv')

