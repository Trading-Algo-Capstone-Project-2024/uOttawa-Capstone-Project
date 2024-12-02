import torch
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns

'''

This document is to make sure that both PyTorch and the device's CUDA cores can be used.

CUDA is proprietary to NVIDIA, if you do not have an NVIDIA GPU, this will not work for you.


'''



print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Seaborn version:", sns.__version__)