from os import listdir
from os.path import join, isfile, dirname, abspath
import sys

ROOT_DIR = dirname(abspath(__file__))


def get_file_path(file_path):
    return join(ROOT_DIR + file_path)


def get_files_in_directory(directory_path):
    return [f for f in listdir(join(ROOT_DIR + directory_path)) if isfile(join(ROOT_DIR + directory_path, f))]
