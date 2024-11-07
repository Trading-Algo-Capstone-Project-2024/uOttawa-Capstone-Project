import torch
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Seaborn version:", sns.__version__)