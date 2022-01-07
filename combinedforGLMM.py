#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 15:11:36 2021

@author: chengchichu
"""
# # after running the main code, this is for GLMM

# combinedX = np.concatenate((preprocessed_X, preprocessed_X_test), axis=0)

# combinedy = np.concatenate((y_train, y_test), axis=0)

# ErData = np.concatenate((combinedX, combinedy.reshape(-1,1)), axis=1)

# encoding_head_flat.append('re72')

# erDataPd = pd.DataFrame(ErData,columns = encoding_head_flat)

# erDataPd.to_csv('/home/anpo/Desktop/pyscript/EDr_72/er72_processed_DATA_v10.csv')



# ################# parse ccs code, for running specific ccs model


# import pandas as pd
# out = pd.read_csv('/home/anpo/Desktop/pyscript/EDr_72/er72_processed_DATA_v10.csv')


# with open('/home/anpo/Desktop/pyscript/EDr_72/ccs_distri.txt', 'r') as f:
#       ccs_ids = f.read().splitlines()

# del out['Unnamed: 0']

# df = out[ccs_ids]

# x = np.where(df == 1, df.columns, '')

# ccs_ = []
# for irow in range(df.shape[0]):  
#     a = x[irow,:].flatten().tolist()
#     ccs_id = [i for i in a if len(i)>0]
#     if not ccs_id:
#         ccs_.append('null')
#     else:
#         ccs_.append(ccs_id[0])


# ccs_ids = set(ccs_ids)
# ccs_not = [item for item in out.columns if item not in ccs_ids]

# # ccs_not = list(set(out.columns) - set(ccs_ids))

# erData = out[ccs_not]

# erData['ccs'] = ccs_

# Is = np.concatenate((X_train.index, X_test.index))

# ISpd = pd.DataFrame(Is,columns = ['newID'])

# erData2 = pd.concat([erData,ISpd],axis=1)

# erData3 = erData2.set_index('newID')

# erData3=erData3.sort_index()

# erData3.to_csv('/home/anpo/Desktop/pyscript/EDr_72/er72_processed_DATA_v10_ccs_converted.csv')


## #####
import pandas as pd

def reverse_onehot(ErData, ccs_ids, newname):

    df = ErData[ccs_ids]
    x = np.where(df == 1, df.columns, '')
    
    ccs_ = []
    for irow in range(df.shape[0]):  
        a = x[irow,:].flatten().tolist()
        ccs_id = [i for i in a if len(i)>0]
        if not ccs_id:
            ccs_.append('null')
        else:
            ccs_.append(ccs_id[0])

    ccs_ids = set(ccs_ids)
    ccs_not = [item for item in ErData.columns if item not in ccs_ids]

    erData = ErData[ccs_not]
    erData[newname] = ccs_
    
    return erData


combinedX = pd.concat((X,X_test))

combinedy = pd.concat((y_train,y_test))

ErData = pd.concat((combinedX,combinedy), axis = 1)


with open('/home/anpo/Desktop/pyscript/EDr_72/ccs_distri.txt', 'r') as f:
      ccs_ids = f.read().splitlines()

out = reverse_onehot(ErData, ccs_ids, 'ccs')

# with open('/home/anpo/Desktop/pyscript/EDr_72/ccsh_distri.txt', 'r') as f:
#       cchs_ids = f.read().splitlines()

# out2 = reverse_onehot(out, cchs_ids, 'ccsh')

# with open('/home/anpo/Desktop/pyscript/EDr_72/atc_distri.txt', 'r') as f:
#       atc_ids = f.read().splitlines()

# out3 = reverse_onehot(out2, atc_ids, 'atc')


# Is = np.concatenate((X_train.index, X_test.index))

# ISpd = pd.DataFrame(Is,columns = ['newID'])

# erData2 = pd.concat([erData,ISpd],axis=1)

# erData3 = erData2.set_index('newID')

erData3=out.sort_index()

erData3.to_csv('/home/anpo/Desktop/pyscript/EDr_72/er72_processed_DATA_v10_ccs_converted_indate_corr.csv', index = False)


























