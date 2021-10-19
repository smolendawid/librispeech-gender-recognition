from logging_messages import info
import soundfile as sf
from checks import check_duration, check_format


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