#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 15:12:23 2021

@author: anpo
"""
from matplotlib_venn import venn3_unweighted
import pickle
def fun_and(a,b,c):
    d = np.logical_and(a,b)
    e = np.logical_and(c,d)
    return e

def class_venn(data):
    Abc = fun_and(data['lg'],~data['rf'],~data['xgb'])
    aBc = fun_and(data['rf'],~data['lg'],~data['xgb'])
    ABc = fun_and(data['lg'],data['rf'],~data['xgb'])
    abC = fun_and(data['xgb'],~data['lg'],~data['rf'])
    AbC = fun_and(data['lg'],data['xgb'],~data['rf'])
    aBC = fun_and(~data['lg'],data['xgb'],data['rf'])
    ABC = fun_and(data['lg'],data['rf'],data['xgb'])
    
    setvalues = (sum(Abc).astype(int), sum(aBc).astype(int), sum(ABc).astype(int), sum(abC).astype(int), sum(AbC).astype(int), sum(aBC).astype(int), sum(ABC).astype(int))
    print(setvalues)
    venn3_unweighted(subsets = setvalues , set_labels = ('LG', 'RF', 'XGB'))

# save the error case from each classifier
  
lg_wrong = [y_test != bst_lg.predict(preprocessed_X_test)]  
rf_wrong = [y_test != bst_rf.predict(preprocessed_X_test)]
xgb_wrong = [y_test != bst_xgb.predict(preprocessed_X_test)]

wrongs = {}
wrongs['rf'] = rf_wrong[0].values
wrongs['lg'] = lg_wrong[0].values
wrongs['xgb'] = xgb_wrong[0].values
wrongs['test_data_index'] = lg_wrong[0].index
wrongs['cols'] = cols
 
lg_corrects = [y_test == bst_lg.predict(preprocessed_X_test)]  
rf_corrects = [y_test == bst_rf.predict(preprocessed_X_test)]
xgb_corrects = [y_test == bst_xgb.predict(preprocessed_X_test)]

corrects = {}
corrects['rf'] = rf_corrects[0].values
corrects['lg'] = lg_corrects[0].values
corrects['xgb'] = xgb_corrects[0].values
corrects['test_data_index'] = lg_corrects[0].index
corrects['cols'] = cols

 # with open('correct_class_cases.pkl', 'wb') as f:
 #      pickle.dump(corrects,f) 

file = open('correct_class_cases.pkl', 'rb')
data = pickle.load(file)
class_venn(data)
# plot the venn diagram





# cm = np.logical_and((rf_wrong[0].values), (lg_wrong[0].values), (xgb_wrong[0].values))

# cm2 = np.logical_and(~(rf_wrong[0].values), ~(lg_wrong[0].values), ~(xgb_wrong[0].values))

# wrong_case = ind[cm]

# correct_case = ind[cm2]

# WCs = df.loc[wrong_case,:]

# CCs = df.loc[correct_case,:]


# def ranksumXY(CCs, WCs, cols2):
#     ps = []
#     for i in cols2:  
#         p = ranksums(CCs[i], WCs[i])
#         ps.append(p)
#     return ps

# from scipy.stats import ranksums

# from scipy.stats import chi2_contingency
# from ER_LG2 import assert_number, add_cut

# def chi2test(case_x, case_y, cols2):
#     chips = []
#     contingency_tables = []
#     for i in cols2:  
#         print(i)
#         # print(case_x)
#         data_crosstab = pd.crosstab(case_x[i], np.squeeze(case_y), margins = False)                                       
#         stat, p, dof, expected = chi2_contingency(data_crosstab) 
#         chips.append([i,p])    
#         contingency_tables.append(data_crosstab)
#     return chips, contingency_tables



# cols = {} 

# # cols['SEX'] = 0 
# # cols['ANISICCLSF_C'] = 2
# # cols['INTY'] = 0
# # cols['week'] = 0
# # cols['weekday'] = 2
# # cols['indate_time_gr'] = 0
# # #cols['ER_visit_30'] = 2
# # #cols['ER_visit_365'] = 2
# # cols['TMP'] = 0	
# # cols['PULSE'] = 0
# # cols['BPS'] = 0
# # cols['GCSE'] = 2
# # cols['GCSV'] = 2
# # cols['GCSM'] = 2
# # cols['BRTCNT'] = 0
# # cols['SPAO2'] = 2
# # #cols['DD_visit_30'] = 2
# # cols['ct'] = 2
# # cols['MRI'] = 2
# # cols['xray'] = 2
# # cols['EKG'] = 2
# # cols['Echo'] = 2
# # #cols['DD_visit_365'] = 2
# # cols['indate_month'] = 0
# # #cols['exam_TOTAL'] = 2
# # #cols['lab_TOTAL'] = 2
# # cols['ANISICMIGD'] = 2
# # cols['ANISICMIGD_1'] = 2
# # cols['ANISICMIGD_2'] = 2
# # cols['ANISICMIGD_3'] = 2
# # cols['Creatine_rslt'] = 2
# # cols['Hb_rslt'] = 2
# # cols['Hct_rslt'] = 2
# # cols['RBC_rslt'] = 2
# # cols['WBC_rslt'] = 2


# cols['ER_LOS'] = 1
# cols['age1'] = 1
# cols['Dr_VSy'] = 1
# cols['WEIGHT'] = 1
# cols['SBP'] = 1
# cols['DBP'] = 1

# case_x = pd.concat([CCs,WCs])
# case_y = np.concatenate((np.ones((len(CCs),1) ), np.zeros((len(WCs),1))))

# case_x['INTY'].fillna(value=6, inplace=True)
# case_x = assert_number(case_x, 'ER_LOS'	)
# case_x = assert_number(case_x, 'age1')
# case_x = assert_number(case_x, 'Dr_VSy'	)
# case_x = assert_number(case_x, 'WEIGHT'	)
# case_x = assert_number(case_x, 'SBP'	)
# case_x = assert_number(case_x, 'DBP'	)
          
# fs_to_imp = ['ER_LOS','TMP','PULSE','BPS','BRTCNT','SPAO2','Dr_VSy','WEIGHT','SBP','DBP']
# impdata = imp.transform(case_x[fs_to_imp])
 
# cnt = 0
# for i in fs_to_imp:
#     print(i)
#     case_x[i] = impdata[:,cnt]
#     cnt+=1
 
# case_x = add_cut(case_x, 'TMP', [36, 37.5, 39]) # 體溫
# case_x = add_cut(case_x, 'PULSE', [60, 100]) # 
# case_x = add_cut(case_x, 'BPS', [90, 130]) # 收縮壓
# case_x = add_cut(case_x, 'BRTCNT', [12, 20]) #
# case_x = add_cut(case_x, 'SPAO2', [94]) #

# cols2 = cols.keys()

# chips, contingency_tables = chi2test(case_x, case_y, cols2)

# rankps = ranksumXY(CCs, WCs, cols2)


