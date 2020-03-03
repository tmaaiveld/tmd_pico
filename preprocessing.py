import tarfile
import zipfile
from pathlib import Path
import pandas as pd
from util import c, get_id, create_folders

# these will end up in a params dict somewhere
RAW_DATA_PATH = Path('data/raw')
STANFORD_DATA_PATH = Path('data/CoNNL')
STANFORD_COLS = ["Sent_pos", "Word", "Lemma", "UPOS", "XPOS", "HEAD", "DEPREL"]
EBM_PHASES = ['hierarchical_labels', 'starting_spans']
# SET = 'train' # 'test/gold'
CATS = ['interventions', 'outcomes', 'participants']


def import_raw(dir_path):
    """

    - extracting directly from tarfile more user-friendly
    - only testing for EBM-NLP 1. Authors use 2?
    """

    dir_path = Path('data/raw/ebm_nlp_1_00') # todo: update

    doc_fnames = (dir_path / 'documents/').glob('*.tokens')

    rows = {f.name.split('.')[0]: f.open().read().split() for f in doc_fnames}
    token_s = pd.DataFrame.from_dict(rows, orient='index').stack()
    token_s.name = 'Word'
    token_s.index.names = ['doc', 'idx']

    pio = dict.fromkeys(CATS)

    for cat in pio:
        train_fnames = c([dir_path.glob(f'annotations/aggregated/{phase}/{cat}/train/*.ann')
                         for phase in EBM_PHASES])
        test_fnames = c([dir_path.glob(f'annotations/aggregated/{phase}/{cat}/test/gold/*.ann')
                         for phase in EBM_PHASES])

        pio[cat] = {
            **{f.name.split('_')[0]: f.open().read().split(',') for f in train_fnames},
            **{f.name.split('_')[0]: f.open().read().split(',') for f in test_fnames}
        }

        # todo: imported both datasets for now. train/test split < discussion topic

    # todo: figure out how to handle test/train split maintained:
        # option 1: maintain split in two dfs
        # option 2: group dataset and add a test/train column
        # option 3: create our own train/test split using the document indices

    # todo: initial match looks alright
    # print(pio['interventions']['9989713'])
    # print(token_s.xs('9989713'))
    #
    # print(len(pio['interventions']['9989713']))
    # print(len(token_s.xs('9989713')))

    # todo: some assert testing to check if data are properly indexed

    df = pd.concat([pd.DataFrame.from_dict(pio[cat], orient='index').stack() for cat in CATS], axis=1)
    df.columns = CATS
    df.index.names = ['doc', 'idx']

    df = pd.merge(token_s, df, how='left', on=['doc','idx'])

    # todo: some missing values...

    print(f'{df.isna().sum()} values were missing.')

    return df


def import_CoNNL(dir_path):
    """
    Preprocessing for the CoNNL features. Could use some refinement.
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

    return df


if __name__ == '__main__':

    create_folders(RAW_DATA_PATH, STANFORD_DATA_PATH)

    labels = import_raw(RAW_DATA_PATH)
    stanford = import_CoNNL(STANFORD_DATA_PATH)

    test_merge = pd.merge(labels, stanford, how='left', on=['doc','idx'])

    print(test_merge)

    print(f'{test_merge.isna().sum()} values were missing from the test merge.')
    print(test_merge['Word_x', 'Word_y'])

    # todo: small amount of data loss, BUT data are currently not properly aligned! (i.e. 5000+ mismatched rows). < discussion topic
    # import the  data from csv in run.py
    # count na's in each doc, filter docs with na's, these ones have crappy alignment
    # figure out which characters/tokens are causing the problem, implement a fix
        # wordwise alignment would be difficult...

    test_merge.to_csv('data/test_merge.csv')
    test_merge.to_pickle('data/test_merge.pkl')

    # raw_path = pre.process_data(RAW_DATA_PATH, 'raw.csv', preprocessing=pre.raw)
