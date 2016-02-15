# -*- coding: utf-8 -*-
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split


def findErrorValue(act, pred):
    if len(act) != len(pred):
        print('Length error!')
        exit()

    correct = 0
    for i in range(len(act)):
        if pred[i] == act[i]:
            correct += 1

    return correct/len(act)

def get_features(train):
    output = list(train.columns)
    output.remove('label')
    return output

train = pd.read_csv('~/projetos/kaggle/digit/data/R_extraction/train_px22_cnt_zero22.csv')
#train = np.genfromtxt(open('/home/elder/projetos/kaggle/digit/data/train.csv', 'r'), delimiter=',')[1:]
#labels = np.genfromtxt(open('/home/elder/projetos/kaggle/digit/data/R_extraction/labels.csv','r'))
test = pd.read_csv('~/projetos/kaggle/digit/data/R_extraction/test_px22_cnt_zero22.csv')
#test = np.genfromtxt(open('/home/elder/projetos/kaggle/digit/data/test.csv', 'r'), delimiter=',')[1:]
features = get_features(train)


# configura parametros do xgboost
param = {}
# classificação tipo multiclasse
param['objective'] = 'multi:softmax'
# tuning
param['eta'] = .05
param['max_depth'] = 4
param['subsample'] = 0.95
param['min_child_weight'] = 200
param['colsample_bytree'] = 0.3
param['num_class'] = 10

# num_boost_round ~= 50/eta
num_boost_round = 1000
nfold = 4

xtrain, xvalid = train_test_split(train, test_size=0.1, random_state=41)
ytrain = xtrain['label']
yvalid = xvalid['label']

dtrain = xgb.DMatrix(xtrain[features], ytrain)
dvalid = xgb.DMatrix(xvalid[features], yvalid)
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

gradient = xgb.train(param, dtrain, num_boost_round,
    evals=watchlist, early_stopping_rounds=num_boost_round, verbose_eval=True)

testepredict = gradient.predict(xgb.DMatrix(xvalid[features]), 
    ntree_limit=gradient.best_ntree_limit)

correct = findErrorValue(yvalid.values, testepredict)
print('Correct value: {:.6f}'.format(correct))

out = gradient.predict(xgb.DMatrix(test[features]), ntree_limit=gradient.best_ntree_limit)

f = open("/home/elder/projetos/kaggle/digit/data/xgb_1_yesdahmer" 
    + str(correct) + ".csv", 'w')
res_str = 'ImageId,Label\n'
total = 1
for i in range(len(out)):
    pstr = str(out[i].astype('int'))
    res_str += str(total) + ',' + pstr + '\n'
    total += 1
f.write(res_str)

