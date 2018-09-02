from IPython.core.debugger import set_trace
import types
import numpy as np
from numpy.random import rand
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import pickle
import torch
from torch import nn
import torch.nn.functional as F
import sys
from pathlib import Path
from itertools import chain
import collections
import re
import cv2
from PIL import Image
import torch.utils.model_zoo as model_zoo
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from ipykernel.kernelapp import IPKernelApp
def in_notebook(): return IPKernelApp.initialized()

import tqdm as tq
from tqdm import tnrange

def tqdm(*args, **kwargs):
    return tq.tqdm(*args, file=sys.stdout, **kwargs)

if not in_notebook():
    from tqdm import trange
    tnrange = trange

from .functional import *