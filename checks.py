from flask import abort
from logging_messages import info, error


def check_format(audioFile):
    info("Format check")
    if audioFile.content_type == 'audio/wav':
        info("Format correct")
        return None
    else:
        error("Error message")
        abort(415, "Wrong format")


def check_duration(audio, fs):
    info("Duration check")
    duration = audio.shape[0] / fs
    if duration > 4:
        error("Error message")
        abort(412, "Too long recording")
    if duration < 2:
        error("Error message")
        abort(412, "Too short recording")
    else:
        info("Duration correct")
        return None