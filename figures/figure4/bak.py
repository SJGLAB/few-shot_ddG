import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

#df = pd.read_csv("../../megascale_train_summary.csv")
df = pd.read_csv("../../mixed_val/mixed.csv")
print(df)
ddG_true = df['ddG_true'].values
thermompnn = df['thermompnn'].values
esm = df['esm2-650m'].values
foldx = df['FoldX'].values
mpnnddg = df['MPNNddG'].values
rasp = df['Rasp'].values
mixed = df['mixed_val'].values

plt.figure(figsize=(12,5))
plt.subplot(121)
threshold = 0.0
y_true = (ddG_true <= threshold).astype(int)

precision_esm, recall_esm, _ = precision_recall_curve(y_true, esm)
plt.plot(recall_esm,precision_esm)

precision_foldx, recall_foldx, _ = precision_recall_curve(y_true, -foldx)
plt.plot(recall_foldx,precision_foldx)

precision_mpnnddg, recall_mpnnddg, _ = precision_recall_curve(y_true, mpnnddg)
plt.plot(recall_mpnnddg,precision_mpnnddg)

precision_rasp, recall_rasp, _ = precision_recall_curve(y_true, -rasp)
plt.plot(recall_rasp,precision_rasp)

precision_mixed, recall_mixed, _ = precision_recall_curve(y_true, -mixed)
plt.plot(recall_mixed,precision_mixed)

plt.legend(['esm','foldx','mpnnddg','rasp','mixed'])

plt.subplot(122)
threshold = -0.5
y_true = (ddG_true <= threshold).astype(int)

precision_esm, recall_esm, _ = precision_recall_curve(y_true, esm)
plt.plot(recall_esm,precision_esm)

precision_foldx, recall_foldx, _ = precision_recall_curve(y_true, -foldx)
plt.plot(recall_foldx,precision_foldx)

precision_mpnnddg, recall_mpnnddg, _ = precision_recall_curve(y_true, mpnnddg)
plt.plot(recall_mpnnddg,precision_mpnnddg)

precision_rasp, recall_rasp, _ = precision_recall_curve(y_true, -rasp)
plt.plot(recall_rasp,precision_rasp)

precision_mixed, recall_mixed, _ = precision_recall_curve(y_true, -mixed)
plt.plot(recall_mixed,precision_mixed)

plt.legend(['esm','foldx','mpnnddg','rasp','mixed'])

plt.savefig('curve.png',bbox_inches='tight',pad_inches=0.1)
