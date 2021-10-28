import connexion
import flask
from voice_gender_classifier import VoiceGenderClassifier
from audio_file_processing import processing


def recognize(audioFile):
    response = processing(audioFile, vgc)
    return response


# if __name__ == '__main__':
model_path = 'model_store/RandomForest.joblib'
vgc = VoiceGenderClassifier(model_path, fs=16000)
app = connexion.FlaskApp(__name__, specification_dir='openapi/')
app.add_api('gender_recognition-openapi.yaml')
