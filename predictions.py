import joblib
import librosa
import soundfile as sf
from extract_feats import extract_features


def prediction(audio, fs, model):
    feats = extract_features(audio, fs)
    result = model.predict(feats)
    return result


if __name__ == '__main__':
    # path = 'data/raw/LibriSpeech/dev-clean/251/136532/251-136532-0000.flac'
    path = 'data/raw/other/Dawid.wav'
    audio, fs = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if fs != 16000:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=16000)
        fs = 16000
    # model = joblib.load('data/model_store')
    result = prediction(audio, fs, model=None)
    print(f'Prediction result: {result}')
