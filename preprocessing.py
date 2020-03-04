import tarfile
import zipfile
from pathlib import Path
import pandas as pd
from util import c, get_id, make_dirs

# these will end up in a params dict somewhere
DATA_PATH = Path('data')
STANFORD_COLS = ["Sent_pos", "Word", "Lemma", "UPOS", "XPOS", "HEAD", "DEPREL"]
PHASE = 'starting_spans' # 'hierarchical_labels'
CATS = ['interventions', 'outcomes', 'participants']


def import_raw(path, phase, verbose=True):
    """
    Function extract EBM-NLP data directly from tarfile.
    :param path: a Path object like Path('path/to/tarfile.tar.gz')
    :param phase: desired phase of the experiment
    :param verbose: whether to print debug information
    :return: An indexed DF of unprocessed text entries and attached labels.
    """
    temp = Path('data/temp')
    root = path.name.split('.')[0]

    try:
        tar = tarfile.open(path.resolve())
    except:
        raise FileNotFoundError('Place the tarfile in the /data/raw directory.')

    if verbose: print(f'Extracting {path.name}. This may take a few minutes.')

    tar.extractall(path=temp)
    temp = temp / root

    doc_fnames = list((temp / 'documents/').glob('*.tokens'))

    if verbose: print(f"parsing {len(doc_fnames)} documents from {path / 'documents/'}.")

    tokens = {f.name.split('.')[0]: f.open(encoding='latin-1').read().split() for f in doc_fnames}
    token_s = prep_tokens(tokens)
    token_s.name = 'Word'
    token_s.index.names = ['doc', 'idx']
    token_s.value_counts().to_csv('data/raw_vocab.csv')

    pio = dict.fromkeys(CATS)
    n_imported_rows = 0

    for cat in pio:
        train_fnames = list(temp.glob(f'annotations/aggregated/{phase}/{cat}/train/*.ann'))
        test_fnames = list(temp.glob(f'annotations/aggregated/{phase}/{cat}/test/crowd/*.ann'))

        # split parameters differ for ebm_nlp_1_00 ['_'] and ebm_nl_2_00
        # split_params = ['_', ','] if path.name == 'ebm_nlp_1_00' else ['.','\n'] # todo: outdated

        pio[cat] = {f.name.split('_')[0]: f.open().read().split(',')
                    for f in train_fnames + test_fnames}

        n_imported_rows += sum([len(l) for l in pio[cat].values()])

        print(f'Successfully parsed {cat}.')

    tar.close()
    # token_s = token_s.loc[(list(pio[cat].keys()),slice(None))] # todo: necessary for ebm_nlp_2

    try:
        assert len(token_s) == n_imported_rows / len(pio)

    except AssertionError:
        print(f'length of token series: {len(token_s)}')
        print(f'number of imported label rows: {n_imported_rows / len(pio)}')
        raise AssertionError('Merge failed. Difference detected in the number of tokens and labels.\n',
                             'Note: this should not be considered an error for ebm_nlp_2_00.tar.gz')

    labels_df = pd.concat(
        [pd.DataFrame.from_dict(pio[cat], orient='index').stack() for cat in pio],
        axis=1)

    labels_df.columns = CATS
    labels_df.index.names = ['doc', 'idx']

    df = pd.merge(token_s, labels_df, how='inner', on=['doc','idx']) # 'inner' for ebm_nlp_2, differing number of docs

    if verbose:
        print(f'Imported {len(df.index)} tokens.')
        print(f'{df.isna().sum().sum()} values were missing from the labels.')

    return df


def prep_tokens(tok_dict):
    s = pd.DataFrame.from_dict(tok_dict, orient='index').stack()

    return s


def import_CoNNL(dir_path, verbose=True):
    """
    Preprocessing for the CoNNL features.
    """

    sep_dfs = []
    doc_ids = []

    for filepath in dir_path.rglob('*.zip'):
        with zipfile.ZipFile(filepath) as z:
            paths = z.namelist()
            print(f'parsing {len(paths)} documents from {filepath}')

            for doc_id in paths:
                doc_ids.append(get_id(doc_id))
                with z.open(doc_id) as f:
                    sep_dfs.append(pd.read_csv(f, delimiter='\t', header=None))

    doc_keys = c([[str(doc_ids[i])] * df.shape[0] for i, df in enumerate(sep_dfs)])
    tok_keys = c([range(len(df)) for df in sep_dfs])

    index = pd.MultiIndex.from_tuples(zip(doc_keys, tok_keys), names=['doc', 'idx'])

    df = pd.concat(sep_dfs)
    df.index = index
    df.columns = STANFORD_COLS

    # rewrite index column # todo: not implemented

    # process variables where needed todo: not implemented

    return df


if __name__ == '__main__':
    make_dirs(DATA_PATH)
    make_dirs(*[(DATA_PATH / d).mkdir() for d in ['raw','CoNNL', 'word2vec', 'temp']])

    if (DATA_PATH / 'raw.pkl').exists():
        raw = pd.read_pickle(DATA_PATH / 'raw.pkl')
        print(f"Importing data from {DATA_PATH / 'raw.pkl'}")

    else:
        try:
            raw = import_raw(DATA_PATH / 'raw/ebm_nlp_1_00.tar.gz', PHASE, verbose=True)
            # stanford = import_CoNNL(DATA_PATH / 'CoNNL', verbose=False)

            assert len(raw) > 0 # and len(stanford) > 0

        except AssertionError:
            raise Exception('Import failed. Ensure ebm_nlp_*_00.tar.gz is placed in data/raw.')

    doc_ids = len(raw.index.get_level_values('doc'))

    # todo: add sentence number column (sentence tagging)


    raw.to_pickle('data/raw.pkl')
    print('Preprocessing complete.')



    # stanford.to_pickle('data/stanford.pkl')
    # test_merge = pd.merge(raw, stanford, how='left', on=['doc','idx'])
    #
    # print(test_merge)
    # print(f'\nMissing values: \n\n {test_merge.isna().sum()} ')
    # print(f"\ndocuments remaining after merge: {}")
    #
    # test_merge.to_csv('data/test_merge.csv')
    # test_merge.to_pickle('data/test_merge.pkl')
    #
    # # delete temp folder contents
