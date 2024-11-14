"""
Following are utility functions created for listing files in a directory and generate a usable key when random values are present in column names

"""
import os
import re

def slugify(s):
    s = s.lower().strip()
    s = re.sub(r'[^A-Za-z0-9]+', '-', string=s)
    return s

def list_files(directory):
    absolute_paths = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            absolute_paths.append(os.path.abspath(filepath))
    return absolute_paths
# print(f)
