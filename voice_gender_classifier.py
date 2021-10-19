import os

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import f1_score
import tqdm

from extract_feats import extract_features


class VoiceGenderClassifier:
    def __init__(self, model_path, fs):
        self.model = joblib.load(model_path)
        self.fs = fs

    def _normalize_audio(self, audio, fs):
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        if fs != self.fs:
            audio = librosa.resample(audio, orig_sr=fs, target_sr=self.fs)
            fs = self.fs
        return audio, fs

    def predict(self, audio, fs):
        audio, fs = self._normalize_audio(audio, fs)
        feats = extract_features(audio, fs)

        feats = np.array(feats)
        feats = feats[np.newaxis, :-1]
        result = self.model.predict_proba(feats)
        if result[0, 1] >= 0.5:
            return "Male", result[0, 1]
        if result[0, 1] < 0.5:
            return "Female", result[0, 1]
        


if __name__ == '__main__':
    # path = 'data/raw/LibriSpeech/dev-clean/251/136532/251-136532-0000.flac'
    # audio_path = 'data/raw/other/Radek.wav'
    # model_path = 'model_store/RandomForest.joblib'
    # result = VoiceGenderClassifier(model_path, fs=16000)
    # audio, fs = sf.read(audio_path)
    # proba = result.predict(audio, fs)
    # print(f'Man probability: {proba}')

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
