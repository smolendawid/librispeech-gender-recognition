import os
import pandas as pd
import numpy as np
from utils.path_utils import project_root
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import plotly.express as px


def get_outliers(lof, nbr_of_outliers=10, threshold=-1.5):
    # TO DO
    pass

if __name__ == '__main__':

    train_data = pd.read_csv(os.path.join(project_root(), 'data', 'processed',
                                          'librispeech-gender-feats-train-clean-100.csv'))

    dev_data = pd.read_csv(os.path.join(project_root(), 'data', 'processed',
                                        'librispeech-gender-feats-dev-clean.csv'))

    data = dev_data.dropna()

    # columns = ['mean', 'std', 'median', 'kurt', 'skew', 'p25', 'p75', 'iqr', 'ent', 'meanfun', 'maxfun', 'minfun']
    columns = ['mean', 'std', 'median', 'kurt', 'skew', 'p25', 'p75', 'iqr', 'ent', 'meanfun']

    values = data[columns].values

    clf = LocalOutlierFactor(n_neighbors=25) #, contamination=0.01
    lof = clf.fit_predict(values)
    lof_values = clf.negative_outlier_factor_
    data['outlier'] = np.where(lof == -1, 'True', 'False')
    data['lof_factor'] = lof_values
    data_sorted = data.sort_values('lof_factor')
    print((data['outlier'].value_counts()))

    plt.hist(data['duration'], bins=30, facecolor='blue', alpha=0.5,edgecolor='black')
    plt.xlabel('duration, s')
    plt.show()
    print(min(data.duration))

    perplexity = 20
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    feats2d = tsne.fit_transform(values)

    data['y'] = feats2d[:, 1]
    data['x'] = feats2d[:, 0]
    fig = px.scatter(data, x='x', y='y', title=f'train data', color="outlier",
                     hover_data=['path'], symbol='label', symbol_sequence=['star', 'diamond']);fig.show()
    fig.write_html(os.path.join(project_root(), "data", "processed", f"outliers_{perplexity}.html"))

    data_mean = data.mean()
    data_3std = 3 * data.std()
    for col in ['mean']:
        print(data_mean[col])
        print(data_3std[col])
        print((data[~(np.abs(data[col] - data_mean[col]) <= data_3std[col])]))
        print(len(data[~(np.abs(data[col] - data_mean[col]) <= data_3std[col])]))
        pass
