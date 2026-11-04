import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman',size=15)
plt.rc('axes', titlesize=18, labelsize=18)

df = pd.read_csv('mixed_train.csv')
ddG_true = df['ddG_true']

plt.figure(figsize=(24,10))

plt.subplot(251)
val = df['esm2-650m']
plt.scatter(ddG_true,val,s=1)
plt.title('ESM2-650M')
plt.ylabel('MegaScale-Train')

plt.subplot(252)
val = df['FoldX']
plt.scatter(ddG_true,val,s=1)
plt.title('FoldX')

plt.subplot(253)
val = df['MPNNddG']
plt.scatter(ddG_true,val,s=1)
plt.title('ProteinMPNN-ddG')

plt.subplot(254)
val = df['Rasp']
plt.scatter(ddG_true,val,s=1)
plt.title('Rasp')

plt.subplot(255)
val = df['mixed_val']
plt.scatter(ddG_true,val,s=1)
plt.title('Ours')


df = pd.read_csv('mixed_val.csv')
ddG_true = df['ddG_true']

plt.subplot(256)
val = df['esm2-650m']
plt.scatter(ddG_true,val,s=1)
plt.ylabel('MegaScale-Validation')

plt.subplot(257)
val = df['FoldX']
plt.scatter(ddG_true,val,s=1)

plt.subplot(258)
val = df['MPNNddG']
plt.scatter(ddG_true,val,s=1)

plt.subplot(259)
val = df['Rasp']
plt.scatter(ddG_true,val,s=1)

plt.subplot(2,5,10)
val = df['mixed_val']
plt.scatter(ddG_true,val,s=1)

plt.savefig('correlation.png',bbox_inches='tight',pad_inches=0.1)
