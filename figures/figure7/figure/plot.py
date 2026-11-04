import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman',size=15)

df = pd.read_csv('summary.csv')
print(df)

bpdb = df['bpdb'].values
val_ans = df['val_ans'].values
ref_ans = df['ref_ans'].values
num = val_ans.shape[0]
pos = np.ones(num*2)
vals = np.zeros(num*2)

apos = np.ones(num)
avals = np.zeros(num)
bpos = np.ones(num)
bvals = np.zeros(num)
for i in range(num):
    pos[2*i] = 3*i
    pos[2*i+1] = 3*i+1
    vals[2*i] = val_ans[i]
    vals[2*i+1] = ref_ans[i]

    apos[i] = 3*i
    bpos[i] = 3*i+1
    avals[i] = val_ans[i]
    bvals[i] = ref_ans[i]
print(bpdb,val_ans,ref_ans,vals)

#bars = plt.bar(pos, vals)
bars = plt.bar(apos, avals)
plt.bar_label(bars, fmt='%.3f', padding=3)
bars = plt.bar(bpos, bvals)
plt.bar_label(bars, fmt='%.3f', padding=3)

plt.ylim(0.65)
plt.xticks([0.5,3.5,6.5],bpdb)
plt.ylabel("|SCC|")
plt.savefig('transfer.png',bbox_inches='tight',pad_inches=0.1)
