# -*- coding: utf-8 -*-

from numpy import genfromtxt, savetxt, chararray

DIR = '/home/elder/projetos/kaggle/digit/'
dados = genfromtxt(
    open('/home/somar33/github/kaggle/output2.csv', 'r'), delimiter=',')

# labels = [i[0] for i in dados]
# treino = [i[1:] for i in dados]
#
# teste = genfromtxt(open(DIR+'test.csv', 'r'), delimiter=',')[1:]
#
# savetxt(DIR+'output4.csv', rforest.predict(teste), delimiter=',', fmt='%d')
header = chararray(2, 7)
header[0] = 'ImageId'
header[1] = 'Label'
