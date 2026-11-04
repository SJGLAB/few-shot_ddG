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
    ans = -abs(val)
    return ans

def cal_accuracy(df,pdb_ids):
    print(f"读取 {len(df)} 行数据")
    pdb_ids = df['pdb_id'].unique()
    cols = df.columns
    num_cols = len(cols)

    result_lst = []
    for i in range(first_col,num_cols):
        col1 = cols[i]
        col2 = cols[8]
        col3 = cols[9]
        col4 = cols[10]
        ans = calculate_val(df,1.0,0.0,0.0,0.0,col1,col2,col3,col4)
        result_lst.append(ans)
        print(col1,'%.5f'%(ans))
    return np.array(result_lst)

df = pd.read_csv('../megascale_val_summary.csv')
pdb_ids = df['pdb_id'].unique()
cols = df.columns
num_cols = len(cols)

num_train_pdb_ids = len(pdb_ids)
train_pdb_ids = pdb_ids[:num_train_pdb_ids]
train_last_index = df[df['pdb_id'] == train_pdb_ids[-1]].index[-1]
train_df = df.loc[:train_last_index]

print(df)
print(pdb_ids)
print(train_df)
print(train_pdb_ids)

total_result_lst = cal_accuracy(df,pdb_ids)
train_result_lst = cal_accuracy(train_df,train_pdb_ids)
print(total_result_lst)
print(train_result_lst)
print('\t'.join([f'{x:.5f}' for x in total_result_lst]))
print('\t'.join([f'{x:.5f}' for x in train_result_lst]))

def objective(params,col1,col2,col3,col4):
    a, b, c, d = params
    ans = calculate_val(train_df,a,b,c,d,col1,col2,col3,col4)
    return ans

effect_numcols = df.shape[1]-first_col
train_pairs_res = np.zeros((effect_numcols,effect_numcols,effect_numcols,effect_numcols))
with open('result_cof.pkl', 'rb') as f:  # 二进制读取
    result_cof = pickle.load(f)
for icol in range(first_col+1,num_cols):
    col1 = cols[icol]
    for jcol in range(icol+1,num_cols):
        col2 = cols[jcol]
        for kcol in range(jcol+1,num_cols):
            col3 = cols[kcol]
            for lcol in range(kcol+1,num_cols):
                col4 = cols[lcol]
                print(col1,col2,col3,col4)

                if (col1,col2,col3,col4) not in result_cof:
                    continue
                a,b,c,d = result_cof[(col1,col2,col3,col4)]
                train_ans = calculate_val(train_df,a,b,c,d,col1,col2,col3,col4)
                print(len(train_pdb_ids),train_ans)
                train_pairs_res[icol-first_col,jcol-first_col,kcol-first_col,lcol-first_col] = train_ans
                print('Result:',col1,col2,col3,col4,'%.5f'%train_ans,flush=True)

print(train_pairs_res)
for i in range(train_pairs_res.shape[0]):
    for j in range(train_pairs_res.shape[1]):
        for k in range(train_pairs_res.shape[2]):
            print('\t'.join([f'{x:.5f}' for x in train_pairs_res[i][j][k]]))
