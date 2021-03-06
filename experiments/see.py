import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

df = pd.read_csv('C:\\Users\\bnaik\\Desktop\\CSCI_5622_Machine_Learning_Project\\data\\train_simplified\\bee.csv')
examples = [ast.literal_eval(e) for e in df['drawing'][:8].values]
fig, ax = plt.subplots(1,8,figsize=(20,4))
for i, example in enumerate(examples[:8]):
    for x, y in example:
        ax[i].plot(x, y, marker='.', markersize=1, lw=3)
        # ax[i].axis('off')
    ax[i].invert_yaxis() 
plt.show()