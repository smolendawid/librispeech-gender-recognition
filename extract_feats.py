import os
import tqdm
import soundfile as sf
import pandas as pd
import numpy as np
from scipy import stats

from utils.path_utils import project_root
from utils.get_librispeech_paths import get_librispeech_paths
from frequency_feats import freq_feats


def extract_features(audio, fs):

    bin_means = freq_feats(audio, fs)
    mean = bin_means.mean()
    std = bin_means.std()
    median = np.median(bin_means)
    kurt = stats.kurtosis(bin_means)
    skew = stats.skew(bin_means)
    p25 = np.percentile(bin_means, 25)
    p75 = np.percentile(bin_means, 75)
    iqr = p75 - p25
    ent = stats.entropy(bin_means)

    return mean, std, median, kurt, skew, p25, p75, iqr, ent


if __name__ == '__main__':
    chosen_set = 'train-clean-100'

    raw_data_root = os.path.join(project_root(), 'data', 'raw', 'LibriSpeech')
    speakers_filepath = os.path.join(raw_data_root, 'SPEAKERS.TXT')

    results_filepath = os.path.join(project_root(), 'data', 'processed', f'librispeech-gender-feats-{chosen_set}.csv')

    audio_paths, labels = get_librispeech_paths(raw_data_root, speakers_filepath, contains=chosen_set)

    tq = tqdm.tqdm(enumerate((zip(audio_paths, labels))), total=len(audio_paths))
    feats_rows = []
    for i, (path, label) in tq:
        audio, fs = sf.read(path)
        row = extract_features(audio, fs)
        feats_rows.append(row)

    results = pd.DataFrame(feats_rows, columns=['mean', 'std', 'median', 'kurt', 'skew', 'p25', 'p75', 'iqr', 'ent'])
    results['path'] = ['/'.join(p.split('/')[-4:]) for p in audio_paths]
    results['label'] = labels

    results.to_csv(results_filepath, index=False)
