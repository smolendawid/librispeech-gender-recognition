import logging


def info(message):
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info(message)


def error(message):
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logging.info(message)
