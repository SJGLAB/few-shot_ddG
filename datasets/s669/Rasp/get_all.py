import numpy as np
import pandas as pd
import glob
import os

# 氨基酸字典
aa_dict = 'ACDEFGHIKLMNPQRSTVWYX'
all_data = []
output_dir = "./output/predictions/"

idx = 0
for csv_file in glob.glob(output_dir+"*.csv"):
    adf = pd.read_csv(csv_file)
    idx += 1
    print(idx,csv_file)
    if idx == 1:
        res_df = adf
    else:
        res_df = pd.concat([res_df,adf],axis=0)
res_df.to_csv('result.csv')
