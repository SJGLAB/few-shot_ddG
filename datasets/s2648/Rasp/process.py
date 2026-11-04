import pandas as pd

df = pd.read_csv('result.csv')
df['pdbid'] = df['pdbid'].str.rstrip('.pdb')
df['chain'] = df['pdbid'].str[-1]
df['pdbid'] = df['pdbid'].str[:4].map(lambda x: x.upper() if isinstance(x, str) else x)

cols_to_drop = ['Unnamed: 0','chainid','variant','wt_idx','mt_idx','wt_nlf','mt_nlf','score_ml_fermi','wt']
df = df.drop(columns=cols_to_drop)

df = df.rename(columns={'pdbid': 'pdb_id','score_ml': 'Rasp','wt_AA': 'pre','mt_AA': 'post'})
df.to_csv('result_processed.csv')
