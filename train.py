import os
import pandas as pd
from utils.path_utils import project_root

data = pd.read_csv(os.path.join(project_root(), 'data', 'processed', 'librispeech-gender-feats.csv'))


