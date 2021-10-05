import connexion
import librosa
import numpy as np
import soundfile as sf
from checks import check_duration, check_format
from voice_gender_classifier import VoiceGenderClassifier


def recognize(audioFile):
    check_format(audioFile)
    audio, fs = sf.read(audioFile)
    check_duration(audio, fs)
    result, proba = vgc.predict(audio, fs)
    response = {"class": result, "probability": proba}
    return response


# if __name__ == '__main__':
model_path = 'model_store/RandomForest.joblib'
vgc = VoiceGenderClassifier(model_path, fs=16000)
app = connexion.FlaskApp(__name__, port=9090, specification_dir='openapi/')
app.add_api('gender_recognition-openapi.yaml')
app.run()

# ? stereo

# unit tests for all outputs and possibe errors