import connexion
from voice_gender_classifier import VoiceGenderClassifier
from api.audio_file_processing import processing
from flask_cors import CORS


def recognize(audioFile):
    response = processing(audioFile, vgc)
    return response


model_path = 'model_store/RandomForest.joblib'
vgc = VoiceGenderClassifier(model_path, fs=16000)
app = connexion.FlaskApp(__name__, specification_dir='api/openapi/')
app.add_api('gender_recognition-openapi.yaml')
CORS(app.app)
