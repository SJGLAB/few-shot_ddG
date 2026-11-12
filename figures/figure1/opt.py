import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rc('font',family='Times New Roman',size=15)

with open('his_result.pkl', 'rb') as f:  # 二进制读取
    his_result = pickle.load(f)

energy = []
for i in range(len(his_result)):
    energy.append(his_result[i].population_energies)
#print(energy)
energy = np.array(energy)
num = energy.shape[0]
print(energy.shape)
x = np.linspace(1,num,num)
mean = np.mean(energy,axis=1)
minv = np.min(energy,axis=1)
print(mean)
print(minv)

plt.plot(x,mean,'-o')
plt.xlabel('Iteration')
plt.ylabel('-SCC')
#plt.plot(x,minv,'-o')

#for i in range(energy.shape[1]):
#    plt.plot(energy[:,i])
plt.savefig('opt.png',bbox_inches='tight',pad_inches=0.1)
