import os
import tqdm
import soundfile as sf
import pandas as pd

from utils.path_utils import project_root
from utils.get_librispeech_paths import get_librispeech_paths


def extract_features(audio):
    return feat


if __name__ == '__main__':

    raw_data_root = os.path.join(project_root(), 'data', 'raw')
    results_filepath = os.path.join(project_root(), 'data', 'processed', 'librispeech-gender-feats.csv')
    speakers_filepath = os.path.join(raw_data_root, 'SPEAKERS.TXT')

    audio_paths, labels = get_librispeech_paths(raw_data_root, speakers_filepath)

    results = pd.DataFrame()
    tq = tqdm.tqdm(enumerate((zip(audio_paths, labels))))
    for i, (path, label) in tq:
        audio = sf.read(path)
        feats = extract_features(audio)
        print()

    results.to_csv(results_filepath, index=False)
