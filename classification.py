import os
import pandas as pd
from utils.path_utils import project_root
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score


if __name__ == '__main__':

    train_data = pd.read_csv(os.path.join(project_root(), 'data', 'processed',
                                          'librispeech-gender-feats-train-clean-100.csv'))

    dev_data = pd.read_csv(os.path.join(project_root(), 'data', 'processed',
                                        'librispeech-gender-feats-dev-clean.csv'))

    train_data = train_data.dropna()
    dev_data = dev_data.dropna()

    columns = ['mean', 'std', 'median', 'kurt', 'skew', 'p25', 'p75', 'iqr', 'ent']

    train_y = train_data['label']

    le = LabelEncoder()
    le.fit(train_data['label'])

    y_train = le.transform(train_data['label'])
    y_dev = le.transform(dev_data['label'])

    x_train = train_data[columns].values
    x_dev = dev_data[columns].values

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_dev = scaler.transform(x_dev)

    model = LogisticRegression(C=1)
    model = RandomForestClassifier(n_estimators=20, max_depth=None, min_samples_leaf=8)
    model.fit(x_train, y_train)

    preds_train = model.predict(x_train)
    preds_dev = model.predict(x_dev)

    print(f"Train f1: {f1_score(y_train, preds_train)}")
    print(f"Dev f1: {f1_score(y_dev, preds_dev)}")
