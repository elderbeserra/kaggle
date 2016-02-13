# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt

DIR = '/home/elder/projetos/kaggle/digit/'
dados = genfromtxt(open(DIR+'train.csv', 'r'), delimiter=',')[1:]

labels = [i[0] for i in dados]
treino = [i[1:] for i in dados]

teste = genfromtxt(open(DIR+'test.csv', 'r'), delimiter=',')[1:]

rforest = RandomForestClassifier(n_estimators=1000, n_jobs=4)
rforest.fit(treino, labels)

savetxt(DIR+'output4.csv', rforest.predict(teste), delimiter=',', fmt='%d')
