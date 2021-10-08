import pytest
import requests
from requests.api import patch

def voice_gender_client(path):

    headers = {
        'accept': 'application/json',
        'Content-Type': 'multipart/form-data',
    }

    files = {
        'audioFile': (path, open(path, 'rb'), 'audio/wav'),
    }

    response = requests.post('http://localhost:9090/v1.0/recognition', files=files)
    return response


def test(): 
    path = 'data/raw/other/Gabi.wav'

    response = voice_gender_client(path=path)

    # response = requests.post('http://localhost:9090/v1.0/recognition', files=dict(audioFile='data\raw\other\Gabi.wav'))

    assert response.status_code == 200
