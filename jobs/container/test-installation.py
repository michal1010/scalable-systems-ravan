import torch
import transformers
import sklearn
import pandas
import matplotlib
import numpy

from federated.lora import LoRALinear
from federated.ravan import RavanLinear

linear = torch.nn.Linear(8, 8, bias=False)
_ = LoRALinear(linear, rank=2)
_ = RavanLinear(torch.nn.Linear(8, 8, bias=False), heads=2, rank=2)

print("Python imports OK")
print(f"NumPy: {numpy.__version__}")
print(f"Pandas: {pandas.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
