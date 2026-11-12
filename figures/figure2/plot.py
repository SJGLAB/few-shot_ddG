import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')

lst = ['ESM2-650M','FoldX','ProteinMPNN-ddG','Rasp','Ours']
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

df = pd.read_csv('spearman.csv', sep='\t')
for col in lst:
    axes[0].scatter(df[col].values, df.index + 1, label=col)
axes[0].set_yticks(df.index + 1)
axes[0].set_yticklabels(df['DataSet'].values)
axes[0].set_xlabel('SCC')
axes[0].text(
    -0.20, 1.0,
    'a',
    transform=axes[0].transAxes,
    fontsize=18,
    fontweight='bold',
    va='top'
)

df = pd.read_csv('pearson.csv', sep='\t')
for col in lst:
    axes[1].scatter(df[col].values, df.index + 1, label=col)
axes[1].set_yticks(df.index + 1)
axes[1].set_yticklabels(df['DataSet'].values)
axes[1].set_xlabel('PCC')
axes[1].text(
    -0.20, 1.0,
    'b',
    transform=axes[1].transAxes,
    fontsize=18,
    fontweight='bold',
    va='top'
)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=len(labels),
    bbox_to_anchor=(0.5, 1.05),
    frameon=False
)
plt.tight_layout(rect=[0, 0, 1, 0.99])

plt.savefig('cof.png',bbox_inches='tight',pad_inches=0.1,dpi=300)
