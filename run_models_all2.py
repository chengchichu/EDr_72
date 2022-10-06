#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 08:16:07 2021

@author: anpo

"""

import pandas as pd
import pickle
from ER_LG3 import run_models, parameter_selection
import numpy as np

def list_duplicates(seq):
    seen = set()
    seen_add = seen.add
    # adds all elements it doesn't know yet to seen and all other to seen_twice
    seen_twice = set( x for x in seq if x in seen or seen_add(x) )
    # turn the set into a list (as requested)
    return list( seen_twice )
      
##############################################################################################

# tune完之後的參數

data_root_folder = '/home/anpo/Desktop/pyscript/EDr_72/'
class_weight_apply = False
if class_weight_apply == False:
   set_bootstrap = True
else:   
   set_bootstrap = False
   
tune_voting_weight = False   

best_hyperparams = {}
a = {}
a = {'C': 0.01, 'penalty': 'l2', 'solver': 'lbfgs'}
best_hyperparams['lg'] = a

a = {}
a = {'n_estimators': 500, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': False}
best_hyperparams['rf'] = a

a = {}
a = {'colsample_bytree': 0.5336888312909173,
 'eta': 0.1341080263159646,
 'gamma': 2.5182603311285114,
 'max_depth': 10,
 'min_child_weight': 1,
 'n_estimators': 25,
 'reg_alpha': 10,
 'reg_lambda': 0.16386208990225784,
 'subsample': 0.643093387291623}
best_hyperparams['xgb'] = a

best_hyperparams_cp = best_hyperparams.copy()


# 讀存好的data

# with open('abdominal_pain_data_new639.pickle', 'rb') as handle:
#       data = pickle.load(handle)

# X_train_c = data['Xtrain'] 
# y_train_c = data['ytrain']  
# preprocessed_X_test = data['Xtest'] 
# y_test = data['ytest'] 
# encoding_head = data['head']

# filename
keyA = '腹痛-hypertuned+bestVote_0930'
keyB = keyA + '30'
seed_tag = 432


# 參數
PARAMs = parameter_selection(best_hyperparams_cp, class_weight_apply)
PARAMs['tune_vote'] = tune_voting_weight

# run model
# all feature model
data_out, clfs_out, imps, AUCS = run_models(X_train_c.values, y_train_c, preprocessed_X_test.values, y_test, encoding_head, PARAMs)
        
r = {}
r['lg'] = evaluate_model(data_out[0], clfs_out['lg'], 'LG', data_out[1])
r['rf']  = evaluate_model(data_out[0], clfs_out['rf'], 'RF', data_out[1])
r['xgb'] = evaluate_model(data_out[0], clfs_out['xgb'], 'XGB', data_out[1])
r['ec']  = evaluate_model(data_out[0], clfs_out['ec'], 'EC', data_out[1])

dims = []
dims.append(X_train_c.shape[0])
dims.append(preprocessed_X_test.shape[0])

save_data(data_root_folder, keyA, seed_tag, dims, r)


# top 30 feature model
best_hyperparams_cp = best_hyperparams.copy()
# top30 = ['Free_typing', 'ER_visit_365', 'Ketorolac','ER_visit_30','Xrayh_T','WEIGHT','DD_visit_365','dxh137','dxh45','atc78','age1','ER_LOS','Dr_VSy','PULSE','in_PULSE','BPS','in_BPB','WEIGHT','in_BPS','x0_I69','dxh131','x0_M32','dxh32','x0_K51','x0_K70','x0_C73','x0_R30','dxh237','x0_C67']
# from plotER72_model.py
# top30 = ['BPB', 'BPS', 'DBP2', 'Dr_VSy', 'ER_LOS', 'ER_visit_30', 'PULSE',
#         'SBP1', 'SBP2', 'TMP', 'WEIGHT', 'age1', 'atc20', 'dxh131',
#         'dxh200', 'dxh32', 'dxh80', 'in_BPB', 'in_BPS', 'in_PULSE',
#         'x0_C92', 'x0_I69', 'x0_K70', 'x0_L03', 'x0_R30']  
top30 = ['BPB', 'CTh_T', 'DBP1', 'DD_visit_365', 'ER_LOS', 'ER_visit_30',
        'ER_visit_365', 'Free_typing', 'Ketorolac', 'Primperan', 'SBP2',
        'Xrayh_T', 'age1', 'atc19', 'atc49', 'atc66', 'atc78', 'dxh131',
        'dxh84', 'in_BPS', 'ICD3_A09']
df_sub_train = pd.DataFrame(X_train_c, columns = encoding_head)
df_sub_test = pd.DataFrame(preprocessed_X_test, columns = encoding_head)

data_out30, clfs_out30, _ , _ = run_models(df_sub_train[top30].values, y_train_c, df_sub_test[top30].values, y_test, encoding_head, PARAMs)

r30 = {}
r30['lg'] = evaluate_model(data_out30[0], clfs_out30['lg'], 'LG', data_out30[1])
r30['rf']  = evaluate_model(data_out30[0], clfs_out30['rf'], 'RF', data_out30[1])
r30['xgb'] = evaluate_model(data_out30[0], clfs_out30['xgb'], 'XGB', data_out30[1])
r30['ec']  = evaluate_model(data_out30[0], clfs_out30['ec'], 'EC', data_out30[1])

save_data(data_root_folder, keyB, seed_tag, dims, r30)

# save trained models

# with open('ER72_trained_models_new639.pickle', 'wb') as handle:
#      pickle.dump(clfs_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('ER72_trained_models30new639.pickle', 'wb') as handle:
#      pickle.dump(clfs_out30, handle, protocol=pickle.HIGHEST_PROTOCOL)


#################### validation

with open('abdominal_pain_data_new639.pickle', 'rb') as handle:
      data = pickle.load(handle)

with open('abdominal_pain_data_valv2_new639.pickle', 'rb') as handle:
      val_data = pickle.load(handle)

for i,j in enumerate(val_data['head']):
    if j.startswith('INTY'):
       val_data['head'][i] = j+'.0'

# 比對新data與舊data的feature差異
new_head = list(set(val_data['head']) - set(data['head']))
old_head = list(set(data['head']) - set(val_data['head']))

X_train_c = val_data['Xtrain'] 
y_train_c = val_data['ytrain']  
preprocessed_X_test = val_data['Xtest'] 
y_test = val_data['ytest'] 

# 合在一起 做test
testX = pd.DataFrame(np.concatenate((X_train_c, preprocessed_X_test),axis=0),columns = val_data['head'])
testY = np.concatenate((y_train_c, y_test),axis=0)

# 做一個size跟就data一樣的matrix, 把新data放進去, default = 0
ept = pd.DataFrame(np.zeros((testX.shape[0],len(data['head']))),columns= data['head'])
for i,j in ept.items():
    if i in val_data['head']:
       ept[i] = testX[i].values

# load saved model and inference
with open('ER72_trained_models_new639.pickle', 'rb') as handle:
      saved_models = pickle.load(handle)

r = {}
r['lg'] = evaluate_model(testY, saved_models['lg'], 'LG', ept.values)
r['rf']  = evaluate_model(testY, saved_models['rf'], 'RF', ept.values)
r['xgb'] = evaluate_model(testY, saved_models['xgb'], 'XGB', ept.values)
r['ec']  = evaluate_model(testY, saved_models['ec'], 'EC', ept.values)

dims = []
dims.append(X_train_c.shape[0])  # 原來的train size
dims.append(ept.shape[0]) # validation data的size

save_data(data_root_folder, 'validation_2022_v2_new639', seed_tag, dims, r)


with open('ER72_trained_models30new639.pickle', 'rb') as handle:
      saved_models30 = pickle.load(handle)

r30 = {}
r30['lg'] = evaluate_model(testY, saved_models30['lg'], 'LG', ept[top30].values)
r30['rf']  = evaluate_model(testY, saved_models30['rf'], 'RF', ept[top30].values)
r30['xgb'] = evaluate_model(testY, saved_models30['xgb'], 'XGB', ept[top30].values)
r30['ec']  = evaluate_model(testY, saved_models30['ec'], 'EC', ept[top30].values)

save_data(data_root_folder, 'validation30_2022_v2_new639', seed_tag, dims, r30)


# compare old data 2018-2018 to new data

# from scipy.stats import ranksums

# for i in ERdata.cont_col:

#     print(i)
#     print(ranksums(ori_data[i], val_ori_data[i], nan_policy ='omit'))





















