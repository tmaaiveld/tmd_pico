# params = {
#     'verbosity': 2,
#     'tree_method': 'gpu_hist',
#     'n_estimators': 10000,
#     'subsample': 0.8,
#     # 'early_stopping_rounds': 2,
#     'max_depth': 1,  # default 6
#     'num_parallel_tree': 10, # default 1
#     # 'gamma': ,
#     # '
#     # 'scale_pos_weight': 100, # set in loop
#     # 'disable_default_eval_metric': 1,
#     # 'objective':'binary:logistic'
#     # 'n_jobs': 2,
# }


params = {
    'n_estimators': None, #set in script
    'criterion': 'entropy',
    'max_depth': None,
    'min_samples_split': 2,
    'max_features': 'auto',
    'bootstrap': True,
    'verbose': 3,
    'n_jobs': 8,
    'class_weight': None,
}