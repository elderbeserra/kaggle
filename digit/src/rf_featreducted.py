# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd


#train_R = pd.read_csv('~/projetos/kaggle/digit/data/R_extraction/train_px22_cnt_zero22.csv')
train_R = np.genfromtxt(open('/home/elder/projetos/kaggle/digit/data/R_extraction/train_px22_cnt_zero22.csv', 'r'), delimiter=',')[1:]
labels = np.genfromtxt(open('/home/elder/projetos/kaggle/digit/data/R_extraction/labels.csv','r'))
test_R = np.genfromtxt(open('/home/elder/projetos/kaggle/digit/data/R_extraction/test_px22_cnt_zero22.csv', 'r'), delimiter=',')[1:]

rforest = RandomForestClassifier(n_estimators=1000, n_jobs=-1, verbose=1)
rforest.fit(train_R, labels)

#np.savetxt('/home/elder/projetos/kaggle/digit/data/rffr_noCV_1.csv', rforest.predict(test_R), delimiter=',', fmt='%d')
predict = rforest.predict(test_R).astype(int)

ImageId = []
for i in range(1,len(predict)+1):
    ImageId.append(i)
    
out = pd.DataFrame({"ImageId": ImageId, "Label": predict})
out.to_csv('~/projetos/kaggle/digit/data/rffr_noCV_3.csv', header=True, index=False, sep=',')
