import connexion
import numpy as np
import soundfile as sf
from voice_gender_classifier import VoiceGenderClassifier


def recognize(audioFile):
    audio, fs = sf.read(audioFile)
    result, proba = vgc.predict(audio, fs)
    response = {"class": result, "probability": proba}
    return response


# if __name__ == '__main__':
model_path = 'model_store/RandomForest.joblib'
vgc = VoiceGenderClassifier(model_path, fs=16000)
app = connexion.FlaskApp(__name__, port=9090, specification_dir='openapi/')
app.add_api('gender_recognition-openapi.yaml')
app.run()

# {
#     "class": "man",
#     "probability": 0.43
# }

# {
#     "class": "female",
#     "probability": 0.43
# }


# "Too long recording"
# "Too short "
# "Wrong format "

# ? stereo

# unit tests for all outputs and possibe errors