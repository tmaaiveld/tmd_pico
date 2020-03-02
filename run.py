from pathlib import Path
from util import c, get_id, create_folders
import pandas as pd
import preprocessing as pre


RAW_DATA_PATH = Path('data/raw')
STANFORD_DATA_PATH = Path('data/CoNNL')

### PRE-PROCESSING ###

create_folders(RAW_DATA_PATH, STANFORD_DATA_PATH)

# todo: refine, extraction should be made simpler at some point.
# input("place the required data in the subfolders of '/data' and press Enter.")
# place ebm_nlp_1_00.tar.gz in raw and extract
# drop ebm_1_parsed into /data/CoNNL (ebm_2_parsed is optional for now)

# todo: not implemented, see raw() in preprocessing

raw_path = pre.import_data(RAW_DATA_PATH, 'raw.csv', preprocessing=pre.raw)
# stanford_path = pre.import_data(STANFORD_DATA_PATH, 'connl.csv', preprocessing=pre.CoNNL)

# df = pre.merge() # todo: not implemented

# raw = pd.read_csv(raw_path)
# stanford_df = pd.read_csv(stanford_df, header=0,
#                           index_col=['PMID', 'token']).drop('ID', axis=1)

# print(df)


