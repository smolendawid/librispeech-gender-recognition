from posixpath import abspath
from flask import abort


def check_format(audioFile):
    if audioFile.content_type == 'audio/wav':
        return None
    else:
        abort(415, "Wrong format")


def check_duration(audio, fs):
    duration = audio.shape[0] / fs
    if duration > 4:
        abort(412, "Too long recording")
    if duration < 2:
        abort(412, "Too short recording")
    else:
        return None