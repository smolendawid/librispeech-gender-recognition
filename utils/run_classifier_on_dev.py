import tqdm
import pandas as pd
import soundfile as sf
from sklearn.metrics import f1_score

from voice_gender_classifier import VoiceGenderClassifier


probas = []
df = pd.read_csv('data/processed/librispeech-gender-feats-dev-clean.csv')
paths = df["path"]
raw_labels = df["label"]
model_path = 'model_store/RandomForest.joblib'
voice_gender_classifier = VoiceGenderClassifier(model_path, fs=16000)

labels = []

for audio_path, raw_label in tqdm.tqdm(zip(paths[:200], raw_labels[:200])):
    try:
        audio, fs = sf.read(audio_path)
        proba = voice_gender_classifier.predict(audio, fs)
        labels.append(1 if raw_label == 'M' else 0)
    except:
        print("skipped", audio_path)
        continue
    probas.append(proba)

print("Dev score: ", f1_score(labels, probas))
