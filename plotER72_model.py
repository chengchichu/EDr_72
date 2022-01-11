#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# plot ER72, roc curve, prcurve

from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot
from sklearn.metrics import roc_curve, auc, classification_report, average_precision_score
import pickle
from scipy.stats import sem

# output from ER_LG2.py
with open('model_out_abdominal_pain.pickle', 'rb') as handle:
      out = pickle.load(handle)
     
# with open('abdominal_pain_data.pickle', 'rb') as handle:
#      data = pickle.load(handle)     
     
# with open('model_out_30_abdominal_pain_notestleak.pickle', 'rb') as handle:
#       out = pickle.load(handle)

# with open('model_out_30_abdominal_pain.pickle', 'rb') as handle:
#       out = pickle.load(handle)
     

# y_test = data['ytest']

# with open('dup_test_idx.pickle', 'rb') as handle:
#      test_duplicates_idx = pickle.load(handle)
     
# fullidx = np.array(range(0,len(y_test)))
# uiidx = np.setdiff1d(fullidx, test_duplicates_idx)
# label = y_test.values[uiidx]

label = data['ytest'].values


cols = ['lg', 'rf', 'xgb', 'svm', 'ec'] # 0.15 0.18 0.2, 0.15, 0.2
colors = ['black','red','blue','green','orange']



mAP = []

fig, (ax1, ax2) = plt.subplots(1, 2)

for idx, i in enumerate(cols):
   
    # pr curve
    precision, recall, thresholds = precision_recall_curve(label, out[i][4], pos_label = 1)
    mAP.append(average_precision_score(label, out[i][4]))
    # fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    # ix = np.nanargmax(fscore)
    # f1_max = np.nanmax(fscore)
    ax1.plot(recall, precision, marker='.', label=i, markersize=0.3, color=colors[idx])
    ax1.legend(loc = 'upper right')
    # ax1.scatter(recall[ix], precision[ix], marker='o', color=colors[idx], label='Best', s =50)
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')

    ax1.axis('scaled')
    ax1.axis(xmin=0,xmax=1)
    ax1.axis(ymin=0,ymax=1)
    # roc curve for the model
    # no_skill = len(testy[testy==1]) / len(testy)
    # pyplot.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
    
    ax2.plot(out[i][0], out[i][1], colors[idx], label = i+'AUC = %0.2f' % out[i][2])
    ax2.legend(loc = 'lower right')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    
    ax2.axis('scaled')
    ax2.axis(xmin=0,xmax=1)
    ax2.axis(ymin=0,ymax=1)
    # 
    
    
    
    ax2.axline((1, 1), slope=1, linestyle = '--')  

# ax2.plot([0, 1], [0, 1], transform=ax2.transAxes)    
# axis labels
# pyplot
# pyplot.ylabel('Precision')
# pyplot.legend()
# # show the plot

# plt.savefig('/home/anpo/Desktop/pyscript/EDr_72/model.eps', format='eps')
plt.savefig('/home/anpo/Desktop/pyscript/EDr_72/model30.eps', format='eps')
pyplot.show()



## plot feature importance

# # if XGB
# dictA = bst_xgb.get_score(importance_type = 'weight')
# dictB = bst_xgb.get_score(importance_type = 'gain')
# dictC = bst_xgb.get_score(importance_type = 'cover')
# dictD = bst_xgb.get_score(importance_type = 'total_gain')
# dictE = bst_xgb.get_score(importance_type = 'total_cover')

def sort_dict_value(dict_in):
    dictA_sorted = { k: v for k, v in sorted(dict_in.items(), key=lambda item: item[1], reverse=True) }
    return dictA_sorted

xgb_weight_imp = out['feature_imp']['xgb'][0]

dictA_ = sort_dict_value(xgb_weight_imp)
# dictB_ = sort_dict_value(dictB)
# dictC_ = sort_dict_value(dictC)
# dictD_ = sort_dict_value(dictD)
# dictE_ = sort_dict_value(dictE)

# # plot
encoding_head = data['head']
bv = list(dictA_.keys())
xh = [encoding_head[int(u[1:])] for u in bv]
yh = list(dictA_.values())

xh = xh[0:50]
yh = yh[0:50]
xgb10 = xh[0:10]

fig, ax = plt.subplots(figsize=(12, 6))
plt.bar(np.arange(0,len(yh)),yh)
ax.set_xticks(np.arange(0, len(xh), step=1)) 
ax.set_xticklabels(xh) 
plt.setp(ax.get_xticklabels(), rotation=90)
plt.ylabel("Xgboost Feature Importance")
plt.xlabel("Features")

plt.savefig('/home/anpo/Desktop/pyscript/EDr_72/xgb_f_imp.eps', format='eps')


lg_weight_imp = out['feature_imp']['lg']

imp2 = lg_weight_imp.sort_values(by=['lg_beta'],ascending = False)
yh = imp2['lg_beta'].values[0:50]
xh = imp2['features'].values[0:50]
lg10 = xh[0:10]

fig, ax = plt.subplots(figsize=(12, 6))
plt.bar(np.arange(0,len(yh)),yh)
ax.set_xticks(np.arange(0, len(xh), step=1)) 
ax.set_xticklabels(xh) 
plt.setp(ax.get_xticklabels(), rotation=90)
plt.ylabel("abs(beta_coefficient)")
plt.xlabel("Features")

plt.savefig('/home/anpo/Desktop/pyscript/EDr_72/lg_f_imp.eps', format='eps')


rf_weight_imp = out['feature_imp']['rf']

imp2 = rf_weight_imp.sort_values(by=['rf_importance'],ascending = False)
yh = imp2['rf_importance'].values[0:50]
xh = imp2['features'].values[0:50]
rf10 = xh[0:10]
fig, ax = plt.subplots(figsize=(12, 6))
plt.bar(np.arange(0,len(yh)),yh)
ax.set_xticks(np.arange(0, len(xh), step=1)) 
ax.set_xticklabels(xh) 
plt.setp(ax.get_xticklabels(), rotation=90)
plt.ylabel("mean decrease in Gini purity")
plt.xlabel("Features")

plt.savefig('/home/anpo/Desktop/pyscript/EDr_72/rf_f_imp.eps', format='eps')

top30 = np.unique( np.concatenate((xgb10,lg10,rf10)) )

# estimate auc ci




n_bootstraps = 1000
rng_seed = 42  # control reproducibility
bootstrapped_scores = []

for idx, i in enumerate(cols):
    
    y_pred = out[i][4]
    y_true = label

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
    
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
        
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
        confidence_lower, confidence_upper))





# explainer = shap.TreeExplainer(bst_xgb)
# pX = pd.DataFrame(preprocessed_X)
# shap_values = explainer.shap_values(pX)
# shap.summary_plot(shap_values, pX)
# #results = model.evals_result()
# fig, ax = plt.subplots(figsize=(12, 6))
# p = shap.force_plot(explainer.expected_value, shap_values[1,:], pX.iloc[1,:])
# plt.savefig('tmp.png')
# plt.close()

















