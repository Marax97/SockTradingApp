import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_file_path(filePath):
    return os.path.join(ROOT_DIR + filePath)