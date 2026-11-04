import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman',size=15)

lst = ['ESM2-650M','FoldX','MPNNddG','Rasp','Mixing']
fig, axes = plt.subplots(1, 2, figsize=(23, 7))

plt.subplot(121)
df = pd.read_csv('spearman.csv',sep='\t')
print(df)
for col in lst:
    plt.scatter(df[col].values,list(df[col].index+1))
plt.yticks(ticks=list(df['DataSet'].index+1),labels=list(df['DataSet'].values))
#plt.legend(lst)
plt.xlabel('|SCC|')

plt.subplot(122)
df = pd.read_csv('pearson.csv',sep='\t')
print(df)
for col in lst:
    plt.scatter(df[col].values,list(df[col].index+1))
plt.yticks(ticks=list(df['DataSet'].index+1),labels=list(df['DataSet'].values))
#plt.legend(lst)
plt.xlabel('|PCC|')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,              # 横向排列数量
    bbox_to_anchor=(0.5, 1.02),
    frameon=False
)

plt.savefig('cof.png',bbox_inches='tight',pad_inches=0.1)


import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman', size=15)

lst = ['ESM2-650M','FoldX','MPNNddG','Rasp','Mixing']
fig, axes = plt.subplots(1, 2, figsize=(23, 7))

df = pd.read_csv('spearman.csv', sep='\t')
for col in lst:
    axes[0].scatter(df[col].values, df.index + 1, label=col)
axes[0].set_yticks(df.index + 1)
axes[0].set_yticklabels(df['DataSet'].values)
axes[0].set_xlabel('|SCC|')

df = pd.read_csv('pearson.csv', sep='\t')
for col in lst:
    axes[1].scatter(df[col].values, df.index + 1, label=col)
axes[1].set_yticks(df.index + 1)
axes[1].set_yticklabels(df['DataSet'].values)
axes[1].set_xlabel('|PCC|')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=len(labels),
    bbox_to_anchor=(0.5, 1.05),
    frameon=False
)
plt.tight_layout(rect=[0, 0, 1, 0.9])

plt.savefig('cof.png',bbox_inches='tight',pad_inches=0.1,dpi=300)
