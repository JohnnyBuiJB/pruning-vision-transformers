# Install these libraries
#!pip install transformers
#!pip install thop
#!pip install torchprofile
#!pip install torchinfo

# General Imports
import torch
from transformers import ViTForImageClassification
from torchprofile import profile_macs
from torchinfo import summary
from thop import profile

# Define constant for FLOPs
billion = 1000000000
million = 1000000

# Define model (DeiT base)
model = ViTForImageClassification.from_pretrained('facebook/deit-base-patch16-224')

# Define dumpy input (image of size 224x224 with 3 channels, batch size of 1)
input_tensor = torch.randn(1, 3, 224, 224)

# Zhu, et al. estimated:
# FLOPS (B) - 17.6
# PARAMS (M) - 86.4

# ------------------------------------------ USING THOP LIBRARY ------------------------------------------

# Run input through model and get FLOPs/params
flops, params = profile(model, inputs=(input_tensor,), verbose=False)

# Print estimated FLOPs and Parameters
print(f"FLOPs (B): {round(flops/billion,2)}") # 16.86 (B) - (slightly less - could be from update?)
print(f"Parameters (M): {round(params/million, 2)}") # 86.42 (M) - (matches - rounding difference)

# ------------------------------------------ USING TORCHPROFILE LIBRARY ------------------------------------------

# Run input through model and get FLOPs/params
flops = profile_macs(model, input_tensor)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print estimated FLOPs and Parameters
print(f"FLOPs (B): {round(flops/billion,2)}") # 17.57 (B) - (matches - rounding difference)
print(f"Parameters (M): {round(params/million, 2)}") # 86.57 (M) - (slightly higher, +0.17 million)

# ------------------------------------------ USING TORCHINFO LIBRARY ------------------------------------------

# Good for printing out parameters for each part of model, size, operations, etc. (no FLOPs)
# Seems to be the most accurate for counting parameters
summary(model, input_size=(1, 3, 224, 224))

