import numpy as np
import pandas as pd

df = pd.read_csv('s571_summary.csv')
df = df[['pdb_id','pos','pre','post','chain','ddG_true','esm2-650m','FoldX','MPNNddG','Rasp']]
df.to_csv('tp.csv')
