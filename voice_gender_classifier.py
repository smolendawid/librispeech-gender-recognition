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
    audio_path = 'data/raw/LibriSpeech/dev-clean/174/50561/174-50561-0000.flac'
    model_path = 'model_store/RandomForest.joblib'
    result = VoiceGenderClassifier(model_path, fs=16000)
    audio, fs = sf.read(audio_path)
    proba = result.predict(audio, fs)
    print(f'Result: {proba}')
