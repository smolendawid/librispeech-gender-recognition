import os
import pandas as pd
from utils.path_utils import project_root
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans

from sklearn.manifold import TSNE
import plotly.express as px


if __name__ == '__main__':

    train_data = pd.read_csv(os.path.join(project_root(), 'data', 'processed',
                                          'librispeech-gender-feats-train-clean-100.csv'))

    dev_data = pd.read_csv(os.path.join(project_root(), 'data', 'processed',
                                        'librispeech-gender-feats-dev-clean.csv'))

    data = dev_data.dropna()

    columns = ['mean', 'std', 'median', 'kurt', 'skew', 'p25', 'p75', 'iqr', 'ent', 'meanfun', 'maxfun', 'minfun']
    # columns = ['iqr', 'meanfun']


    for i in range(7, len(columns)-1):
        for j in range(i+1, len(columns)):

            tmp_columns = [columns[i], columns[j]]
            values = data[tmp_columns].values

            scaler = StandardScaler()
            scaler.fit(values)
            values = scaler.transform(values)

            # model = DBSCAN(eps=0.8, min_samples=4)
            model = KMeans(n_clusters=2)
            clusters = model.fit_predict(values)

            data['clusters'] = clusters

            fig = px.scatter(data, x=tmp_columns[0], y=tmp_columns[1], title=f'train data', color="clusters",
                             hover_data=['path'], symbol='label', symbol_sequence=['star', 'diamond']);fig.show()
            fig.write_html(os.path.join(project_root(), "data", "processed",
                                        f"clusters_{tmp_columns[0]}_{tmp_columns[1]}.html"))
