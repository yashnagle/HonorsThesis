import os
import wget
import tarfile
import re
from nltk.tokenize import word_tokenize
import collections
import pandas as pd
import pickle
import numpy as np



# with tarfile.open("dbpedia_csv.tar.gz") as tar:
#         tar.open()

tar = tarfile.open('dbpedia_csv.tar.gz', 'r:gz')
tar.extractall('./my_folder')
tar.close()
        
print('Done')
        
