import sys
import numpy as np
import matplotlib.pyplot as plt
import utils
import build_datasets
from skimage import io
import cv2
import argparse

if __name__ == "__main__":
    # Generam dataseturile de spectrograme
    build_datasets.generate_all()

