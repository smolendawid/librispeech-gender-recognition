import soundfile as sf
from api.logging_messages import info
from api.checks import check_duration, check_format


def processing(audioFile, vgc):
    info("Audio file in")
    info("Audio processing")
    check_format(audioFile)
    audio, fs = sf.read(audioFile)
    check_duration(audio, fs)
    info("Audio processed")
    result, proba = vgc.predict(audio, fs)
    info("Result done successfully")
    response = {"class": result, "probability": proba}
    return response