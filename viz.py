import matplotlib.pyplot as plt 
import pandas as pd 

global_model = pd.read_csv('global_acc.csv',header=None)
local_models = pd.read_csv('local_acc.csv',header=None)

plt.plot(global_model[0],label='global')
for i in range(5):
    plt.plot(local_models[i],label='local#%d'%(i+1))
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('n5_r50.png',bbox_inches='tight') # 24mins