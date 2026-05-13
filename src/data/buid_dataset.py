import os
import numpy as np
import torch
import mne
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from scipy.io import loadmat