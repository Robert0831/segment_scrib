from pathlib import Path
import pickle
import random
import numpy as np
import json
import cv2
from copy import deepcopy


path="C:/Users/acvlab/Desktop/EMC-Click-main/lvis_v1_val.json"
f = open(path)
data = json.load(f)
print(np.array(data['annotations'][0]['segmentation']).shape)