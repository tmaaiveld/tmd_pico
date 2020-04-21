import sys
import random
import time
import warnings
import pickle
from pathlib import Path
import scipy
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from model_params import params

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import xgboost as xgb

pd.options.display.float_format = '{:20,.5f}'.format

EXPERIMENTS = Path('../../experiments/data_dumps/')
DIR_NAME = sys.argv[1] if len(sys.argv) > 1 else f"XGB_{''.join(np.random.randint(0,9,4).astype(str))}"

PIO = ["participants", "interventions", "outcomes"]
TOTAL_DOCS = 4993
POS_WEIGHT_FACTOR_ADJUST = 0.5
N_TRAIN = ''


def hotcode(df, dropcols=False):
    cat_cols = set(df.select_dtypes(include=['category']).columns)
    print(f"Generating one-hot coding for columns {', '.join(list(cat_cols))}")

    if len(cat_cols) > 0:
        dummies = pd.get_dummies(df[cat_cols].copy(), columns=cat_cols, dtype=bool)

        df = downcast(pd.concat([df.drop(cat_cols, axis=1), dummies], axis=1))
        print('hotcode complete.')

    #     dropcols = df.loc[:,(df.sum() == 0)].columns

    if dropcols:
        df = df.drop(dropcols, axis=1)

    return df


def downcast(df):
    unsigned = df.select_dtypes(include=['uint8', 'uint16', 'int8', 'int16', 'int32', 'int64']).columns
    for col in unsigned:
        df[col] = df[col].apply(pd.to_numeric, downcast='signed')

    return df


def print_attrs(obj, only_attrs=False, only_names=False):
    attrs = [s for s in dir(obj) if not s[:2] == '__']

    for attr in attrs:
        if not callable(getattr(obj, attr)):
            print(f'\n*** ATTR: xgtrain.{attr} ***')
            if not only_names: print(getattr(obj, attr))

        elif not only_attrs:
            print(f'\n*** METHOD: xgtrain.{attr} ***')
            try:
                if not only_names: print(getattr(obj, attr)())
            except:
                print('failed')


def write_lines(path, collection):
    with (path).open('w') as f:
        namestr = '\n'.join(list(collection))
        f.write(namestr)


# def clean_processing():
#     label = 'interventions'
#     df = pd.read_parquet('data/split/train3_100.parquet')
#
#     labels = df.copy()[PIO]
#     features = hotcode(
#         df.copy().drop(PIO, axis=1)
#     )
#
#     num_pos = labels[label].sum()
#     num_neg = len(labels[label]) - num_pos
#     weight = num_neg / num_pos
#
#     weights = ((labels[label] * weight) + 1)
#     weights = weights / max(weights)
#
#     # features.insert(0, label, df[label])
#     # features.insert(1, 'class_weight', weights)
#

def sparsify(df):
    csr_mat = scipy.sparse.csr_matrix(df.values)

    return csr_mat


def read_inds(path):
    with open(path, 'r') as f:
        text = f.readlines()

    inds = zip([line.split(' ') for line in text])
    return inds


def delete(var):
    if 'val' in globals().keys():
        del var


def write_dict(path, dict_obj):
    with open(path, 'w') as f:
        f.write('\n'.join([f'{k}: {v},' for k, v in dict_obj.items()]))
    print(f'dict saved to {path}')


def get_pos_weights(y_train):
    return (len(y_train) - sum(y_train)) / sum(y_train)


def prepare_data(data_path='../../data/xgb_split', k_folds=10):


    data_path = sys.argv[1] if len(sys.argv) > 1 else data_path
    data_path = Path(data_path)
    data = pd.read_parquet(data_path).drop(['par_lemma', 'lemma'], axis=1)

    print(data)

    print(len(data.columns))
    data = data.drop(data.columns[data.columns.str.contains('PMFT')],axis=1)

    print(len(data.columns))
    n_docs = len(data.index.unique('doc'))
    n_string = f'_{n_docs}' if n_docs != TOTAL_DOCS else ''

    print(f'preparing dataset with {n_docs} documents.')

    X = hotcode(data.copy().drop(PIO, axis=1))
    y = data.copy()[PIO]

    print(X.shape)
    print(X.info())
    print(list(X.columns))

    train_idx, val_idx = train_test_split(y.index.unique('doc'), train_size=1 - (1 / k_folds)) # default 80:20
    print(f'Performing a {len(train_idx)}:{len(val_idx)} train/test split.')

    X_train, X_test = [X.loc[(idx, slice(None), slice(None)), :] for idx in [train_idx, val_idx]]
    y_train, y_test = [y.loc[(idx, slice(None), slice(None)), :] for idx in [train_idx, val_idx]]

    folder = Path(f'../../data/xgb_split{n_string}/')
    folder.mkdir(exist_ok=True, parents=True)

    print(f'Split completed. Saving data to {str(folder)}.csv')

    words = pd.read_parquet('../../data/labels_sent.parquet')['Word'].loc[y_test.index]
    y_test = pd.concat([words, y_test], axis=1)

    [df.to_parquet(str(folder / f'{str(name)}')) for df, name in [(X_train, f'X_train{n_string}.parquet'), (X_test, f'X_test{n_string}.parquet'),
                                                                   (y_train, f'y_train{n_string}.parquet'), (y_test, f'y_test{n_string}.parquet')]]


    words.to_csv(str(folder / 'words.csv'))

    return len(y.index.unique('doc'))


def main(load_folder=f'../../data/xgb_split/', label='interventions', dir_name=DIR_NAME):
    # fixes:
    # - reduce n workers
    # - less features
    # - smaller dataset

    load_folder = Path(load_folder)

    overwrite = False
    if not load_folder.exists() or overwrite:
        print('Data path not found. Creating data.')
        n_docs = prepare_data()
        print('processing complete. Recommended to rerun the script.')
        quit()
        load_folder = (Path('/'.join(load_folder.name.split('_')[:-1] + [n_docs]))).mkdir(exist_ok=True, parents=True) # todo: doesn't work

    n_string = f"{load_folder.name.split('_')[-1]}"
    dir_name = f'{random.randint(0,999)}_{label}' if not dir_name else dir_name

    start_time = time.time()
    label = sys.argv[2] if len(sys.argv) > 2 else label

    # Load X and y
    print(f"Loading datasets {', '.join([str(p.name) for p in list(load_folder.glob('*.parquet'))])} from {load_folder}")

    X_train = pd.read_parquet(load_folder / f'X_train.parquet')
    features = X_train.columns
    X_train = X_train.values

    y_train_df = pd.read_parquet(load_folder / f'y_train.parquet')
    X_test  = pd.read_parquet(load_folder / f'X_test.parquet').values
    y_test_df  = pd.read_parquet(load_folder / f'y_test.parquet')

    test_words = y_test_df.copy()['Word']
    train_index = pd.read_parquet(load_folder / f'X_train.parquet').index

    y_train_df['all'] = ~(y_train_df.sum(axis=1) == 0)
    y_test_df['all'] = ~(y_test_df.sum(axis=1) == 0)

    print('\ntrain docs:\n', train_index)
    print('\n Docs in validation set:\n', y_train_df.index.unique('doc'))

    n_est = 500
    params['n_estimators'] = n_est

    print(f"\nTraining a GBC model to predict label '{label}'.")
    print('input shapes: ', [df.shape for df in [X_train, X_test, y_train_df, y_test_df]])
    print('Starting SKLRF.')

    # PIO.insert(0, 'all')
    print(PIO)

    # Set the parameters by cross-validation
    # for n_est in n_estimators:
    # for i, label in enumerate(PIO):

    y_train = np.where(y_train_df[label].values, label, f'not_{label}')
    y_test = np.where(y_test_df[label].values, label, f'not_{label}')
    print(np.unique(y_train, return_counts=True))

    label_counts = np.unique(y_train, return_counts=True)[1]

    params['class_weight'] = {label: label_counts[1] / label_counts[0], f'not_{label}': 1.0}

    print(params['class_weight'])

    print(f'\nTesting n_estimators {n_est} for label {label}')

    clf = RandomForestClassifier(**params)

    clf.fit(X=X_train, y=y_train)

    y_pred = clf.predict(X_test)

    with open('dump1.pickle', 'wb') as f:
        pickle.dump(y_pred, f)

    print("\n******************************\n")
    report = pd.DataFrame(
        classification_report(y_test.flatten(), y_pred.flatten(),
                              digits=3, output_dict=True)
    )
    print(report.head())
    print("\n******************************\n")

    print(report)
    time.sleep(3)

    print(f"Saving data...")

    exp_folder = EXPERIMENTS / dir_name / label / str(n_est)
    print('exp folder is here:', exp_folder.resolve())
    exp_folder.mkdir(exist_ok=True, parents=True)

    probs = clf.predict_proba(X_test)
    preds = {'token': test_words,
            f'not_{label}': probs[:,0],
            f'{label}': probs[:,1],
            f'pred': y_pred,
             'true': y_test
    }
    pd.DataFrame(preds, index=test_words.index).to_csv(exp_folder / 'predict.csv')

    feat_imps = clf.feature_importances_
    pd.DataFrame(feat_imps, index=features).to_csv(exp_folder / 'feature_importances.csv')

    report.to_csv(exp_folder / 'class_report.csv')
    # pd.to_csv(evals_result, exp_folder / 'evals_result.csv')
    # pd.DataFrame({'pred': prediction*1, 'true': sets['val'][1]*1}, index=val_index).to_csv(exp_folder / 'prediction.csv')


    if hasattr(clf, 'clf.best_score'):
        print(f"No score improvement detected over {params['early_stopping_rounds']} steps, terminating.")

    write_dict((exp_folder / '.params'), params)

    print(f'\n Total computation time: {time.time() - start_time:.2f}s')
    print(f"Saved data for run with target '{label}' in '{dir_name}'")


if __name__ == '__main__':
    main()
