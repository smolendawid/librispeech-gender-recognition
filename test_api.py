import pytest
import requests


def voice_gender_client(path):

    headers = {
        'accept': 'application/json',
        'Content-Type': 'multipart/form-data',
    }

    file_format = path.split("/")[3].split(".")[1]

    files = {
        'audioFile': (path, open(path, 'rb'), 'audio' + '/' + file_format),
    }

    response = requests.post('https://enigmatic-badlands-41342.herokuapp.com/v1.0/recognition', files=files)
    return response


def test_normal_wave(): 
    path = 'data/raw/other/Dawid2.wav'
    response = voice_gender_client(path=path)
    assert response.status_code == 200


def test_too_long_wave(): 
    path = 'data/raw/other/Dawid.wav'
    response = voice_gender_client(path=path)
    assert response.status_code == 412


def test_wrong_format():
    path = 'data/raw/other/Gabi flac.flac'
    response = voice_gender_client(path=path)
    assert response.status_code == 415


def test_too_short_wave(): 
    path = 'data/raw/other/Short.wav'
    response = voice_gender_client(path=path)
    assert response.status_code == 412
