from pathlib import Path
from util import c, get_id, create_folders
import pandas as pd
import preprocessing as pre


RAW_DATA_PATH = Path('data/raw')
STANFORD_DATA_PATH = Path('data/CoNNL')

# todo: refine, extraction should be made simpler at some point.
# input("place the required data in the subfolders of '/data' and press Enter.")
# place ebm_nlp_1_00.tar.gz in raw and extract
# drop ebm_1_parsed into /data/CoNNL (ebm_2_parsed is optional for now)


# stanford_path = pre.import_data(STANFORD_DATA_PATH, 'connl.csv', preprocessing=pre.CoNNL)

