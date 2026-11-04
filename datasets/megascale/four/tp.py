import numpy as np
import pandas as pd

df = pd.read_csv('../megascale_train_summary.csv')
pdb_ids = df['pdb_id'].unique()

print(df)
print(pdb_ids)

print(df.shape)
print(len(pdb_ids))
