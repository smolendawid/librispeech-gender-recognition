
from utils.parse_speakers_txt import parse_speakers_txt


def get_librispeech_paths(raw_data_root, speakers_filepath):
    """

    :param raw_data_root: path to all data with folders train-100-clean dev-clean etc
    :param speakers_filepath: path to SPEAKERS.txt
    :return: paths, labels
    """
    sex_mapping = parse_speakers_txt(speakers_path=speakers_filepath)

    audio_paths = []
    labels = []
    for path, subdirs, files in os.walk(raw_data_root):
        for name in files:
            if name.endswith('.flac'):
                audio_paths.append(os.path.join(path, name))
                u_id = name.split('-')[0]
                labels.append(sex_mapping[u_id])

    return audio_paths, labels
