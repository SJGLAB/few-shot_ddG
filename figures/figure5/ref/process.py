import numpy as np
import argparse
import pandas as pd
import scipy.stats
from scipy.optimize import differential_evolution,shgo
import pickle

df = pd.read_csv('../../../megascale_val_summary.csv')
pdb_ids = df['pdb_id'].unique()
cols = df.columns
num_cols = len(cols)

train_dfs = []
val_dfs = []
for pdb_id in pdb_ids:
    pdb_df = df[df['pdb_id'] == pdb_id]
    train_df = pdb_df.sample(n=pdb_df.shape[0]//2, random_state=42)
    val_df = pdb_df.drop(train_df.index)
    print(pdb_id,train_df.shape,val_df.shape)
    train_dfs.append(train_df)
    val_dfs.append(val_df)

big_train_df = pd.concat(train_dfs, ignore_index=True)
big_val_df = pd.concat(val_dfs, ignore_index=True)
print(big_train_df.shape,big_val_df.shape)
big_train_df.to_csv('train_df.csv')
big_val_df.to_csv('val_df.csv')
