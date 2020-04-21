import sys
import random
import warnings
from pathlib import Path
import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import metrics

pd.options.display.float_format = '{:20,.5f}'.format

EXPERIMENTS = Path('experiments')
dir_name = 'CRF_allfeats'
PIO = ["participants", "interventions", "outcomes"]
DATA_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else 'data/split_CRF')


def dictify(docs_df):
    x = docs_df.groupby(level=['doc', 'sent']).apply(
        lambda df: list(df.xs(df.name).reset_index(drop=True).T.to_dict().values())
    )

    return x


def get_label(df, label):
    return list(df[label].apply(str).groupby(['doc', 'sent']).apply(list))


def aggregate_labels(lab_df, level='sent'):
    """Can aggregate the labels if desired. Currently not (properly) implemented."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        lab_df['none'] = (lab_df.sum(axis=1) == 0).astype(int)
        lab_df['label'] = ['none'] * len(lab_df)

        for label in list(PIO):
            lab_df.loc[lab_df[label] == 1, 'aggregate'] = label

    return lab_df


train = pd.read_parquet(DATA_PATH / f'another_train_4500.parquet').iloc[1000:]
test = pd.read_parquet(DATA_PATH / f'another_test.parquet').iloc[1000:]
feature_cols = [col for col in train.columns if col not in PIO]

assert train.isna().sum().sum() == 0, 'Missing values detected in features'

X_train_df, X_test_df = train[feature_cols], test[feature_cols]
y_train_df, y_test_df = train[PIO], test[PIO]

test_words = X_test_df.copy()['token']
train_index = train.index

del train
del test

print('Data successfully loaded, converting to dict format.')
X_train, X_test = [dictify(df) for df in [X_train_df, X_test_df]]

labels = PIO
params = {'algorithm':'lbfgs', 'c1':1.0, 'c2':10**-3, 'max_iterations': 5, 'all_possible_transitions': True}
score_metrics = ['accuracy', 'precision', 'recall', 'f_score', 'support']

for label in labels:
    print(f'Training model to predict label {label}')

    crf = sklearn_crfsuite.CRF(**params)
    y_train, y_test = [get_label(df, label) for df in [y_train_df, y_test_df]]

    crf = crf.fit(X_train, get_label(y_train_df, label))
    crf = crf.fit(X_train, labels)

    score = crf.score(X_test, y_test)
    targets = crf.classes_
    targets.remove('0')
    y_pred = crf.predict(X_test)

    exp_folder = EXPERIMENTS / f'final_crf_runs/run0/{label}'
    report = pd.DataFrame(
                metrics.flat_classification_report(y_test, y_pred, labels=targets,
                                                   digits=3, output_dict=True)
                ).rename({'1': label}, axis='columns')
    report.to_csv(exp_folder / 'report.csv')

    predictions = pd.DataFrame({'token': test_words,
                                f'y_true_{label}': y_test,
                                f'y_pred_{label}': y_pred})

    predictions.to_csv(exp_folder / 'predictions.csv')


