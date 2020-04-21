import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pd.options.display.float_format = '{:20,.5f}'.format

dir_name = 'CRF_allfeats'
# PIO = {"participants", "interventions", "outcomes"}


def convert_strings(df):
    converted_df = pd.DataFrame()
    for col in df.columns:
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:
            converted_df.loc[:, col] = df[col].astype('category')
        else:
            converted_df.loc[:, col] = df[col]
    return converted_df


def get_top_words(df, col, n=1000):
    return df.drop(df.index[df[col].duplicated()]) \
               .sort_values('tfidf', ascending=False).iloc[:n][col]


def pca_transform(X, cutoff=0.9):
    pca = PCA()
    pca.fit(X)
    features = pca.fit_transform(X)

    var_ratios = pca.explained_variance_ratio_
    var_ratios.sort()
    var_ratios = var_ratios[::-1]

    n_vars = 0
    c = 0.0

    while c < cutoff:
        c += var_ratios[n_vars]
        n_vars += 1

    selected_var_ratios = var_ratios[:n_vars]

    plt.plot(var_ratios)
    plt.title('PCA scores')
    plt.axvline(n_vars, color='r')
    plt.savefig('figures/PCA_scores.svg')

    print(f'{n_vars} remaining after PCA analysis.')

    colnames = [f'PCA_{i}' for i in range(n_vars)]
    features = pd.DataFrame(features[:,:n_vars], index=X.index, columns=colnames)

    return features

def tree_optimize(features, keep_lemmas=False, n_drop_lemmas=None):
    df = features
    # df = pd.read_parquet('data\\dataset.parquet')

    print(list(df.columns))
    start_time = time.time()

    top_lemmas = set(get_top_words(df, 'lemma'))
    replace_lemmas = {k: '_' for k in set(df['lemma']) - top_lemmas}

    print('replacing lemmas')

    for col in ['lemma', 'par_lemma']:
        df[col] = df[col].where(~(df['is_int']), 'xNUMx')
        df[col] = df[col].where(~(df['is_dec']), 'xDECx')
        print(df[col])

        if not keep_lemmas:
            df[col] = df[col].apply(replace_lemmas, n_drop_lemmas).fillna(df[col])
            df[col] = df[col].fillna(df['token'])

    # processing = set(df.columns)
    drop_feats = {'token', 'feats', 'stem', 'head', 'par_form', 'CoNNL_ID',
                  'IS_INT_LAG-2', 'IS_INT_LAG-1', 'IS_INT_LAG1', 'IS_INT_LAG2', 'IS_DEC_LAG-2',
                  'IS_DEC_LAG-1', 'IS_DEC_LAG1', 'IS_DEC_LAG2', 'FIRST_WORD_LAG-1', 'FIRST_WORD_LAG-2',
                  'LAST_WORD_LAG2', 'LAST_WORD_LAG1', 'LEMMA_LAG-2', 'LEMMA_LAG-1', 'LEMMA_LAG1', 'LEMMA_LAG2',
                  'XPOS_LAG-2', 'XPOS_LAG-1', 'XPOS_LAG1', 'XPOS_LAG2', 'DEPREL_LAG-2', 'DEPREL_LAG-1',
                  'DEPREL_LAG1', 'DEPREL_LAG2', 'FORM_LAG-2', 'FORM_LAG-1', 'FORM_LAG1', 'FORM_LAG2', 'form', 'par_form'}

    to_drop = set(df.columns).intersection(drop_feats)

    df = df.drop(to_drop, axis=1)
    print('converting dtypes')
    int_dc = df.select_dtypes(include=['int32']).copy().apply(pd.to_numeric,downcast='signed') \
                      .join(df.select_dtypes(include=['int64']).copy().apply(pd.to_numeric,downcast='signed'))
    float_dc = df.select_dtypes(include=['float64']).apply(pd.to_numeric,downcast='float')
    str_dc = convert_strings(df.select_dtypes(include=['object']).copy())

    print('Replacing numerical columns with downcast equivalents.')
    drop_cols = list(int_dc.columns) + list(float_dc.columns) + list(str_dc.columns)
    print([df.drop(drop_cols, axis=1), int_dc, float_dc, str_dc])

    df = pd.concat([df.drop(drop_cols, axis=1), int_dc, float_dc, str_dc], axis=1)

    unsigned = df.select_dtypes(include=['uint8', 'uint16']).columns
    for col in unsigned:
        df[col] = df[col].apply(pd.to_numeric, downcast='signed')

    print(time.time()-start_time)

    obj_cols = df.select_dtypes(include='object').columns
    df[obj_cols] = df[obj_cols].fillna('_')

    return df