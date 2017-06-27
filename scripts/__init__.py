import os

import __main__
if hasattr(__main__, '__file__'):
    # Go to base path to load data
    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    os.chdir(base_path)
