import os

from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd

from utils.path_utils import project_root


if __name__ == '__main__':
    path = os.path.join(project_root(), 'data', 'processed', 'librispeech-gender-feats-test-clean.csv')
    data = pd.read_csv(path)
    columns = ['mean', 'std', 'median', 'kurt', 'skew', 'p25', 'p75', 'iqr', 'ent']

    data = data.dropna()

    perplexity = 20
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    feats2d = tsne.fit_transform(data[columns].values)

    data['x'] = feats2d[:, 0]
    data['y'] = feats2d[:, 1]
    fig = px.scatter(data, x='x', y='y', title=f'TSNE 2d projection of data<br>{path}', color="label", hover_data=['path']);fig.show()
    fig.write_html(os.path.join(project_root(), "data", "processed", f"2d_{perplexity}.html"))
