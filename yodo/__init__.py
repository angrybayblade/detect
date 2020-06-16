import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

from .anchors import BoxGrid, AnchorGrid, Anchors
from .utils import JSON,XMLParser
from .net import YODO

"""
YODO : You only detect one (or more)

A minimal object detection framework for single class object detection
"""