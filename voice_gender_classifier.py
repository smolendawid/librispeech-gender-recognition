import joblib
import librosa
import numpy as np
import soundfile as sf
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

    def prediction(self, audio_path):
        audio, fs = sf.read(audio_path)
        audio, fs = self._normalize_audio(audio, fs)
        feats = extract_features(audio, fs)

        feats = np.array(feats)
        feats = feats[np.newaxis, :-1]
        result = self.model.predict_proba(feats)

        return result[0, 1]


if __name__ == '__main__':
    # path = 'data/raw/LibriSpeech/dev-clean/251/136532/251-136532-0000.flac'
    audio_path = 'data/raw/other/Gabo.wav'
    model_path = 'model_store/RandomForest.joblib'
    result = VoiceGenderClassifier(model_path, fs=16000)
    proba = result.prediction(audio_path)
    print(f'Man probability: {proba}')
