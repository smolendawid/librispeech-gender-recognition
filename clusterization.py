import os
import pandas as pd
from utils.path_utils import project_root
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from sklearn.manifold import TSNE
import plotly.express as px


if __name__ == '__main__':

    train_data = pd.read_csv(os.path.join(project_root(), 'data', 'processed',
                                          'librispeech-gender-feats-train-clean-100.csv'))

    dev_data = pd.read_csv(os.path.join(project_root(), 'data', 'processed',
                                        'librispeech-gender-feats-dev-clean.csv'))

    data = dev_data.dropna()

    columns = ['mean', 'std', 'median', 'kurt', 'skew', 'p25', 'p75', 'iqr', 'ent', 'meanfun', 'maxfun', 'minfun']

    values = data[columns].values

    scaler = StandardScaler()
    scaler.fit(values)
    values = scaler.transform(values)

    model = DBSCAN(eps=1.4, min_samples=5)
    clusters = model.fit_predict(values)

    data['clusters'] = clusters

    perplexity = 20
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    feats2d = tsne.fit_transform(values)

    data['x'] = feats2d[:, 0]
    data['y'] = feats2d[:, 1]
    fig = px.scatter(data, x='x', y='y', title=f'train data', color="clusters",
                     hover_data=['path'], symbol='label', symbol_sequence=['star', 'diamond']);fig.show()
    fig.write_html(os.path.join(project_root(), "data", "processed", f"clusters_{perplexity}.html"))
