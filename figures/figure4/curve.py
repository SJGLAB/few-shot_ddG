import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

plt.rc('font', family='Times New Roman', size=15)
df = pd.read_csv("../../mixed_val/mixed.csv")
ddG_true = df['ddG_true'].values
esm = df['esm2-650m'].values
foldx = df['FoldX'].values
mpnnddg = df['MPNNddG'].values
rasp = df['Rasp'].values
mixed = df['mixed_val'].values

methods = {
    'ESM2-650M': esm,
    'FoldX': -foldx,
    'ProteinMPNN-ddG': mpnnddg,
    'Rasp': -rasp,
    'Ours': -mixed
}


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
thresholds = [0.0, -0.5]
for ax, threshold in zip(axes, thresholds):
    y_true = (ddG_true <= threshold).astype(int)
    for name, pred in methods.items():
        precision, recall, _ = precision_recall_curve(y_true,pred)
        auc_val = auc(recall,precision)
        print(threshold,name,auc_val)
        ax.plot(recall,precision,label=name)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'ddG threshold = {threshold} kcal/mol')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

axes[0].text(
    -0.15, 0.985,
    'a',
    transform=axes[0].transAxes,
    fontsize=18,
    fontweight='bold'
)

axes[1].text(
    -0.15, 0.985,
    'b',
    transform=axes[1].transAxes,
    fontsize=18,
    fontweight='bold'
)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.05),
    ncol=len(labels),
    frameon=False,
    fontsize=15
)

plt.tight_layout(rect=[0, 0, 1, 0.99])

plt.savefig('curve.png',dpi=300,bbox_inches='tight',pad_inches=0.1)
