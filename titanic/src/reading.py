import pandas as pd
import socket
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import socket


computer = socket.gethostname()
if computer == 'somar33':
    ROOT = '/home/somar33/projetos/titanic/titanic/data/'
else:
    ROOT = '/home/'+socket.gethostname()+'/projetos/titanic/data/'

traindata = pd.read_csv(ROOT+'train.csv', delimiter=',')

for i in xrange(0, 2):
    print i, len(traindata[(traindata['Sex'] == 'male') & (
        traindata['Survived'] == i)])

# traindata['Age'].hist()
traindata['Age'].dropna().hist(bins=20, range=(0, 80), alpha=.5)
plt.show()
