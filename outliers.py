import os
import pandas as pd
import numpy as np
from utils.path_utils import project_root
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

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

    columns = ['mean', 'std', 'median', 'kurt', 'skew', 'p25', 'p75', 'iqr', 'ent', 'meanfun', 'maxfun', 'minfun']

    values = data[columns].values

    clf = LocalOutlierFactor(n_neighbors=10) #, contamination=0.01
    lof = clf.fit_predict(values)
    lof_values = clf.negative_outlier_factor_
    data['outlier'] = np.where(lof == -1, 'True', 'False')
    data['lof_factor'] = lof_values
    data_sorted = data.sort_values('lof_factor')
    print((data['outlier'].value_counts()))


    perplexity = 20
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    feats2d = tsne.fit_transform(values)

    data['y'] = feats2d[:, 1]
    data['x'] = feats2d[:, 0]
    fig = px.scatter(data, x='x', y='y', title=f'train data', color="outlier",
                     hover_data=['path'], symbol='label', symbol_sequence=['star', 'diamond']);fig.show()
    fig.write_html(os.path.join(project_root(), "data", "processed", f"outliers_{perplexity}.html"))
