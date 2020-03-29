#!/usr/bin/env python
# coding: utf-8

# In[232]:


import random
import pickle
import time
import warnings
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn_crfsuite
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn_crfsuite import metrics

pd.options.display.float_format = '{:20,.5f}'.format

SEED = random.seed(0)
EXPERIMENTS = Path('experiments')
dir_name = 'CRF_allfeats'
PIO = {"participants", "interventions", "outcomes"}

# # Dataset

# In[233]:


train_val = pd.read_parquet('datasets/train_1000.parquet')
# df.drop('par_form', axis=1)


# In[234]:


print('token' in train_val.columns)
print('lemma' in train_val.columns)
print('word' in train_val.columns)


# ## Feature selection

# Create some subsets of processing to iterate over when generating results.

# In[235]:


get = lambda pat, s: {v for v in s if pat in v}

labels = PIO.union({'aggregate'})
features = {}

features['all'] = set(train_val.columns.difference(PIO))
features['fasttext'] = get('PMFT', features['all'])
features['lag'] = get('LAG', features['all']) 
features['deprel'] = {'deprel', 'dist_to_parent', 'par_form', 'par_lemma', 'par_upos', 'par_xpos'}
del get

print(list(features.items())[0])


# ## Validation Split

# Split off an additional set for validation, static for now. Validation could be implemented in the loop by creating folds of test indices here and calling those in the validation loop.

# In[236]:


k_folds = 5

train_idx, val_idx = train_test_split(train_val.index.unique('doc'), 
                                      train_size=1-(1/k_folds)) # default 80:20

print(f'splitting data {len(train_idx)}:{len(val_idx)}')

train_set = train_val.loc[(train_idx, slice(None)),:]
val_set   = train_val.loc[(val_idx, slice(None)),:]

X_train_df, y_train_df = train_set[features['all']], train_set[PIO]
X_test_df, y_test_df = val_set[features['all']], val_set[PIO]


# ## Formatting

# Reformat the data to a condensed dict version. Specify whether to aggregate the labels into a single multi-column (the procedure is arbitrary and needs refinement or explanation), or to keep the label columns separate and to train a model for each target.

# In[237]:


def aggregate_labels(lab_df, level='sent'):
    """Can aggregate the labels if desired. Currently not (properly) implemented."""
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        lab_df['none'] = (lab_df.sum(axis=1) == 0).astype(int)
        lab_df['label'] = ['none'] * len(lab_df)

        for label in list(PIO):
            lab_df.loc[lab_df[label] == 1, 'aggregate'] = label

    return lab_df


def dictify(docs_df):
    x = docs_df.groupby(level=['doc', 'sent']).apply(
        lambda df: list(df.xs(df.name).reset_index(drop=True).T.to_dict().values())
        )
    
    return list(x)

# for df in [X_train_df, X_test_df]:
#     df['lemma'][df['lemma'].isna()] = df['token']
#     df['par_lemma'][df['par_lemma'].isna()] = df['token']

y_train_df, y_test_df = [aggregate_labels(df) for df in [y_train_df, y_test_df]]

X_train, y_train, X_test, y_test = [dictify(df) for df in [X_train_df, y_train_df, X_test_df, y_test_df]]


# # Model

# Open issues:
# 
# - class imbalance: how to handle [https://elitedatascience.com/imbalanced-classes]
# - Scoring over multi
# - Label aggregation (not needed?)

# In[ ]:


def get_label(df, label):
    return df[label].apply(str).groupby(['doc', 'sent']).apply(list)

labels = PIO
params = {'algorithm':'lbfgs', 'c1':0.1, 'c2':0.1, 'max_iterations': 80, 'all_possible_transitions': True}
###############
# 'c1': 1.0,  #
# 'c2': 1e-3, #
###############
score_metrics = ['accuracy', 'precision', 'recall', 'f_score', 'support']


# In[ ]:


# for feature_set, feats in processing.items():
#     if feature_set != 'all': print('done'); break
       
#     predictions = X_test_df['lemma'].to_frame() #index=y_test.index)#, columns=labels)
    
#     start_time = time.time()
    
#     for label in labels:
        
#         print(f'running for label {label}')
#         y_train, y_test = [get_label(df, label) for df in [y_train_df, y_test_df]]
            
#         crf = sklearn_crfsuite.CRF(**params)
        
#         crf = crf.fit(X_train, y_train)
#         score = crf.score(X_test, y_test)
        
#         targets = crf.classes_
#         targets.remove('0')
        
#         y_pred = crf.predict(X_test)

#         report = pd.DataFrame(
#             metrics.flat_classification_report(y_test, y_pred, labels=targets, 
#                                                digits=3, output_dict=True)
#             ).rename({'1': label}, axis='columns')
        
        
#         print(f"Saving data for run with processing '{feature_set}' and target '{label}' in '{dir_name}'")
        
#         Path(f'models/{dir_name}').mkdir(exist_ok=True, parents=True)
#         dump(crf, f'models/{dir_name}/{"crf_bro"}.joblib')
              
#         model_name = label[0].upper() + f'_{feature_set}'
#         save_model = True
        
#         # save the model
#         if save_model:
#             Path(f'models/{dir_name}').mkdir(exist_ok=True, parents=True)
#             dump(crf, f'models/{dir_name}/{model_name}_{feature_set}.joblib')
        
#         exp_folder = EXPERIMENTS / dir_name / model_name
#         exp_folder.mkdir(exist_ok=True, parents=True)
  
#         # save predicted vs true labels table

# #         pred_results = pd.DataFrame([X_test_df['token'].values, c(y_test), c(y_pred)], 
# #                                     index=['token', f'{label}_true', f'{label}_pred']).T

#         report.to_csv(exp_folder / 'report.csv')
# #         pred_results.to_csv(exp_folder / 'predictions.csv')
             
#         c = lambda x: list(itertools.chain.from_iterable(x)) 
#         predictions[f'{label}_true'] = c(y_test)
#         predictions[f'{label}_pred'] = c(y_pred)
             
#         print(report)
#         print(predictions)
             
#         print(f'\n{time.time() - start_time:.2f}')
#         predictions.to_csv(exp_folder / 'predictions.csv')
# print('Done')    
        
# # print prompt/notification for label
# # print prompt/notification for 


# In[ ]:


words = X_train_df['lemma']

words.head()

# X_train_df.xs('11683484', level='doc')['token'].values.tolist()


# In[ ]:





y_train_df = y_train_df.fillna('none')
y_train_df = y_train_df.groupby(level=[0])['aggregate'].apply(list)


# In[1]:

crf = sklearn_crfsuite.CRF(**params)
y_train, y_test = [get_label(df, 'interventions') for df in [y_train_df, y_test_df]]
crf = crf.fit(X_train, get_label(y_train_df, label))
crf = crf.fit(X_train, labels)
score = crf.score(X_test, y_test)
targets = crf.classes_
targets.remove('0')
y_pred = crf.predict(X_test)



# In[ ]:


report = pd.DataFrame(
            metrics.flat_classification_report(y_test, y_pred, labels=targets, 
                                               digits=3, output_dict=True)
            ).rename({'1': label}, axis='columns')


# In[ ]:


report


# In[ ]:





# In[ ]:




