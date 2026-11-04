import pandas as pd

rdf = pd.read_csv('ref.csv')
rdf['bpdb'] = rdf['pdb_id']
rdf['ref_ans'] = rdf['val_ans']
rdf = rdf.drop(columns=['Unnamed: 0','pdb_id','val_ans'])

df = pd.read_csv('result.csv')

df.merge(rdf,on=['bpdb'],how='left').to_csv('summary.csv')
