import os
import tarfile
import zipfile
from pathlib import Path
from glob import glob
import pandas as pd
from util import c, get_id
import codecs

EBM_COLS = ["ID", "Word", "Lemma", "UPOS", "XPOS", "HEAD", "DEPREL"]

def import_data(path, filename, preprocessing):

    path = Path(path)

    if not path.exists():
        path.mkdir()

        input("Extract EBM archive into the /parsed directory and press enter.")
        # todo: add license and code reference for SCNLP?

    elif (path / filename).exists(): return path / filename

    df = preprocessing(path)

    # todo: module to add labels

    df.to_csv(path / filename)

    print('Imported Stanford CoreNLP CoNNL features.')
    return path / filename


def raw(path):
    """
    todo: read tarfile with tarfile module, extract files and preprocess. See
    """

    raise NotImplementedError()

    # # todo: read these in properly
    # test_path = RAW_DATA_PATH / 'ebm_nlp_1_00.tar.gz'
    # with tarfile.open(test_path) as tar:
    #     names = tar.getnames()
    #     for path in names:
    #         file = pd.read_csv(tar.extractfile(path),
    #                            compression='gzip', delimiter='\t', header=None)
    #         print(file)
    #         break


def CoNNL(path):

    dfs = []
    doc_ids = []

    for filepath in path.rglob('*.zip'):
        with zipfile.ZipFile(filepath) as z:
            paths = z.namelist()
            print(f'parsing {len(paths)} documents from {filepath}')

            for doc_id in paths:
                doc_ids.append(get_id(doc_id))
                with z.open(doc_id) as f:
                    dfs.append(pd.read_csv(f, delimiter='\t', header=None))

    doc_keys = c([[doc_ids[i]] * df.shape[0] for i, df in enumerate(dfs)])
    tok_keys = c([df.iloc[:,0] for df in dfs])

    index = pd.MultiIndex.from_tuples(zip(doc_keys, tok_keys), names=['PMID', 'token'])

    df = pd.concat(dfs)
    df.index = index
    df.columns = EBM_COLS

    # rewrite index column # todo: not implemented

    return df


def merge():
    """
    Function to align and merge raw tokens and labels into dataframe for keras model
    """
    raise NotImplementedError()
