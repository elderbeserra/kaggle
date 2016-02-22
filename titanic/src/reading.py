import pandas as pd
import socket
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import socket


computer = socket.gethostname()
if computer == 'somar33':
    ROOT = '/home/somar33/projetos/kaggle/titanic/data/'
else:
    ROOT = '/home/elder/projetos/kaggle/titanic/data/'

traindata = pd.read_csv(ROOT+'train.csv', delimiter=',')

for i in xrange(0, 2):
    print i, len(traindata[(traindata['Sex'] == 'male') & (
        traindata['Survived'] == i)])

traindata['Gender'] = traindata['Sex'].map({'female': 0, 'male': 1}).astype(int)

# traindata['Age'].hist()
#traindata['Age'].dropna().hist(bins=20, range=(0, 80), alpha=.5)
plt.show()
