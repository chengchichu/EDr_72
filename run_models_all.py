#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 08:16:07 2021

@author: anpo

"""

import pandas as pd
import pickle
from ER_LG2 import run_models
import numpy as np
# params = {}
# cls_constructor('rf',params)



      
##############################################################################################


data_root_folder = '/home/anpo/Desktop/pyscript/EDr_72/'
class_weight_apply = False
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

# with open('abdominal_pain_data.pickle', 'rb') as handle:
#      data = pickle.load(handle)

# X_train_c = data['Xtrain'] 
# y_train_c = data['ytrain']  
# preprocessed_X_test = data['Xtest'] 
# y_test = data['ytest'] 
# encoding_head = data['head']

# without dup y
# with open('dup_test_idx2.pickle', 'rb') as handle:
#      test_duplicates_idx = pickle.load(handle)

# fullidx = np.array(range(0,len(y_test)))
# uiidx = np.setdiff1d(fullidx, test_duplicates_idx)
# preprocessed_X_test = preprocessed_X_test[uiidx, :]
# y_test = y_test.values[uiidx]

keyA = '腹痛-hypertuned+bestVote'
keyB = keyA + '30'


models_out, bst_ec, bst_xgb, bst_rf, bst_lg = run_models(X_train_c, y_train_c, preprocessed_X_test, y_test, keyA, encoding_head, data_root_folder, best_hyperparams_cp, class_weight_apply, seed_tag)
        

# top 30 feature for prediction
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
       'dxh84', 'in_BPS', 'x0_A09']
df_sub_train = pd.DataFrame(X_train_c, columns = encoding_head)
df_sub_test = pd.DataFrame(preprocessed_X_test, columns = encoding_head)
models30_out, bst_ec2, bst_xgb2 = run_models(df_sub_train[top30].values, y_train_c, df_sub_test[top30].values, y_test, keyB, top30, data_root_folder, best_hyperparams_cp, class_weight_apply, seed_tag)


# with open('model_out_abdominal_pain_notestleak.pickle', 'wb') as handle:
#      pickle.dump(models_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
 

# with open('model_out_30_abdominal_pain.pickle', 'wb') as handle:
#      pickle.dump(models30_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('model_out_30_abdominal_pain_notestleak.pickle', 'wb') as handle:
#      pickle.dump(models30_out, handle, protocol=pickle.HIGHEST_PROTOCOL)




# s1 = df['IDCODE'].iloc[xtrain_idx]
# s2 = df['IDCODE'].iloc[xtest_idx]

# a,b,c = np.intersect1d(s2.values,s1.values,return_indices=True)
# test_duplicates_idx = []
# blist = s2.values[b]
# for i,j in enumerate(list(s2.values)):
#     if j in blist:
#        test_duplicates_idx.append(i) 
       
# with open('dup_test_idx2.pickle', 'wb') as handle:
#      pickle.dump(np.array(test_duplicates_idx), handle, protocol=pickle.HIGHEST_PROTOCOL)

####################
# def run_models(X_train_c, y_train_c, preprocessed_X_test, y_test, model_strat, encoding_head, data_root_folder, best_hyperparams, class_weight_apply):
    
#     # 調整class weight的話, 在training的時候就不bootstrap     
#     if class_weight_apply == False:
#        set_bootstrap = True
#     else:   
#        set_bootstrap = False
    
#     #
#     params = {} 
#     if class_weight_apply == False:     
#        if best_hyperparams:   
#           params = best_hyperparams['lg']              
#     elif class_weight_apply == True:    
#        if best_hyperparams:
#           params = best_hyperparams['lg']
#           params['class_weight'] = {0:0.05,1:0.95}
#        else: 
#           params['class_weight'] = {0:0.05,1:0.95}
          
#     clf1 = cls_constructor('lg', params)      
#     print('running LG')
#     bst_lg, _, _, aucs_lg = ml_model(clf1, X_train_c, y_train_c, set_bootstrap)
    
#     # LG imp
#     imp = pd.DataFrame(data = abs(bst_lg.coef_[0]),columns = ['lg_beta'])
#     head = pd.DataFrame(data = encoding_head,columns = ['features'])
#     imp_LG = pd.concat([imp, head],axis = 1)


#     #    
#     params = {} 
#     if class_weight_apply == False:     
#        if best_hyperparams:   
#           params = best_hyperparams['rf']              
#     elif class_weight_apply == True:    
#        if best_hyperparams:
#           params = best_hyperparams['rf']
#           params['class_weight'] = {0:0.05,1:0.95}
#        else: 
#           params['class_weight'] = {0:0.05,1:0.95}
          
#     clf2 = cls_constructor('rf', params)      
#     print('running RF')
#     bst_rf, _, _, aucs_rf = ml_model(clf2, X_train_c, y_train_c, set_bootstrap)
    
#     # 
#     imp = pd.DataFrame(data = abs(bst_rf.feature_importances_),columns = ['rf_importance'])
#     imp_RF = pd.concat([imp, head],axis = 1)
    
    
#     #
#     print('running XGB')  
#     if class_weight_apply == False:     
#        if best_hyperparams:   
#           bst_xgb, bst_xgb2 = model_xgb(xgb, X_train_c, y_train_c, best_hyperparams['xgb'])      
#        else:
#           bst_xgb, bst_xgb2 = model_xgb(xgb, X_train_c, y_train_c, best_hyperparams)
#     elif class_weight_apply == True:    
#        if best_hyperparams:
#           best_hyperparams['xgb']['scale_pos_weight'] = 16 
#           bst_xgb, bst_xgb2 = model_xgb(xgb, X_train_c, y_train_c, best_hyperparams['xgb'])  
#        else:
#           best_hyperparams['scale_pos_weight'] = 16   
#           bst_xgb, bst_xgb2 = model_xgb(xgb, X_train_c, y_train_c, best_hyperparams)
      
#     _, _, _, aucs_xgb = ml_model(bst_xgb2, X_train_c, y_train_c, set_bootstrap)

#     # different type of feature imp for xgb 
#     dictA = bst_xgb.get_score(importance_type = 'weight')
#     dictB = bst_xgb.get_score(importance_type = 'gain')
#     dictC = bst_xgb.get_score(importance_type = 'cover')
#     dictD = bst_xgb.get_score(importance_type = 'total_gain')
#     dictE = bst_xgb.get_score(importance_type = 'total_cover')
#     impXGB = [dictA, dictB, dictC, dictD, dictE]
    
    
#     #     
#     clf4 = LinearSVC(random_state=0, tol=1e-5, dual=False, max_iter = 10000) 
#     print('running SVC')
#     bst_svm, _, _, aucs_svm = ml_model(clf4, X_train_c, y_train_c, set_bootstrap)

#     eclf1 = VotingClassifier(estimators=[('lg', bst_lg), ('rf', bst_rf), ('xgb', bst_xgb2)], voting='soft', weights = [2.5,5,5])
#     bst_eclf, models, kidx, aucs_eclf = ml_model(eclf1, X_train_c, y_train_c, set_bootstrap)
    
    
    
#     cm_lg, cp_lg, others_lg = model_result(y_test, bst_lg, 'LG', preprocessed_X_test)
#     cm_rf, cp_rf, others_rf  = model_result(y_test, bst_rf, 'RF', preprocessed_X_test)
#     cm_xg, cp_xg, others_xg  = model_result(y_test, bst_xgb, 'XGB', preprocessed_X_test)
#     cm_sv, cp_sv, others_sv  = model_result(y_test, bst_svm, 'SVM', preprocessed_X_test)
#     cm_ec, cp_ec, others_ec  = model_result(y_test, bst_eclf, 'ECLF', preprocessed_X_test)

#     models_ = {}
#     lg_auc, lgprc, lg_yscore, fpr, tpr = model_auc(bst_lg, preprocessed_X_test, y_test)
#     models_['lg'] = (fpr, tpr, lg_auc, aucs_lg, lg_yscore)
#     rf_auc, rfprc, rf_yscore, fpr, tpr = model_auc(bst_rf, preprocessed_X_test, y_test)
#     models_['rf'] = (fpr, tpr, rf_auc, aucs_rf, rf_yscore)
#     svm_auc, svmprc, svm_yscore, fpr, tpr = model_auc(bst_svm, preprocessed_X_test, y_test)
#     models_['svm'] = (fpr, tpr, svm_auc, aucs_svm, svm_yscore)
#     ec_auc, ecprc, ec_yscore, fpr, tpr = model_auc(bst_eclf, preprocessed_X_test, y_test)
#     print(ec_auc)
#     models_['ec'] = (fpr, tpr, ec_auc, aucs_eclf, ec_yscore)
#      # xgb test part 跟別人不同分開寫
#     dtest = xgb.DMatrix(preprocessed_X_test , label = y_test)
#     xgb_yscore = bst_xgb.predict(dtest)
#     fpr, tpr, _ = roc_curve(y_test, xgb_yscore)
#     xgb_auc = auc(fpr, tpr)
#     models_['xgb'] = (fpr, tpr,xgb_auc, aucs_xgb, xgb_yscore)
    
#     imps = {}
#     imps['lg'] = imp_LG
#     imps['rf'] = imp_RF
#     imps['xgb'] = impXGB    
#     models_['feature_imp'] = imps
     
#     # if not model_strat:
#     print(model_strat)
#     print('LG')
#     tb1 = table_r(cp_lg,cm_lg,lg_auc,others_lg)
#     print('RF')
#     tb2 = table_r(cp_rf,cm_rf,rf_auc,others_rf)
#     print('XGB')
#     tb3 = table_r(cp_xg,cm_xg,xgb_auc,others_xg)
#     print('SVM')
#     tb4 = table_r(cp_sv,cm_sv,svm_auc,others_sv)
#     print('EC')
#     tb5 = table_r(cp_ec,cm_ec,ec_auc,others_ec)
     
#     filename = 'M'+model_strat+'result.txt'
#     ftb = 'LG' + '\n' + tb1 + '\n' +'RF' + '\n' + tb2 + '\n' +'XGB' + '\n' + tb3 + '\n' +'SVM' + '\n' + tb4 + '\n' +'EC' + '\n' + tb5 + '\n'
                
#     ftb2 = ftb+'\n'+'train_size:'+str(X_train_c.shape[0])+'\n'+'test_size:'+str(preprocessed_X_test.shape[0])
    
#     with open(data_root_folder+filename, 'w') as f:
#          f.write(ftb2)















