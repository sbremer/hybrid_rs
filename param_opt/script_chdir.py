import os

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

import sys
sys.path.insert(0, base_path)

import __main__
if hasattr(__main__, '__file__'):
    # Go to base path to load data
    os.chdir(base_path)
