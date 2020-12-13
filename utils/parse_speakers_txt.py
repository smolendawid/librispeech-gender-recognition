
def parse_speakers_txt(speakers_path):
    """ Parse Libbrispeech original file with info about speakes"""
    mapping = {}
    with open(speakers_path) as f:
        lines = f.read().splitlines()[12:]
        for line in lines:
            splitted = line.split('|')
            u_id = splitted[0].replace(' ', '')
            sex = splitted[1].replace(' ', '')
            mapping[u_id] = sex
    print(f"Found {len(mapping)} speakers")
    return mapping
