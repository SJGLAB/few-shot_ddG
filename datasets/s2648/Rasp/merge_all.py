import pandas as pd

df = pd.read_csv('result_processed.csv')
cols_to_drop = ['Unnamed: 0']
df = df.drop(columns=cols_to_drop)
print(df)

exper = pd.read_csv('../s2648_summary.csv')
adf = df.merge(exper, on=['pdb_id', 'pos', 'pre', 'post', 'chain'], how='right')
print(adf)
cols_to_drop = ['Unnamed: 0','Unnamed: 0.1']
adf = adf.drop(columns=cols_to_drop)
adf = adf[[col for col in adf.columns if col != 'FoldX' and col != 'MPNNddG' and col != 'Rasp'] + ['FoldX','MPNNddG','Rasp']]
adf.to_csv("s2648_summary.csv")
print(f"完成！生成 {len(adf)} 行数据")

